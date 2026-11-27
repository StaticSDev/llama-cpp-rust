//! Port of `tools/server/server-queue.{h,cpp}`.
//!
//! Task queue, response fan-out and the response reader used by handlers.

use crate::core::Result;
use crate::tools::server::task::{ServerTask, TaskResult};
use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

type NewTaskCallback = Arc<dyn Fn(ServerTask, bool) -> bool + Send + Sync>;
type SleepingCallback = Arc<dyn Fn(bool) + Send + Sync>;
type UpdateSlotsCallback = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct QueueState {
    id: i32,
    running: bool,
    sleeping: bool,
    req_stop_sleeping: bool,
    n_inflight: usize,
    tasks: VecDeque<ServerTask>,
    deferred: VecDeque<ServerTask>,
    unhandled: VecDeque<ServerTask>,
}

/// `server_queue`.
pub struct ServerQueue {
    state: Mutex<QueueState>,
    cv: Condvar,
    new_task: Mutex<Option<NewTaskCallback>>,
    sleeping_callbacks: Mutex<Vec<SleepingCallback>>,
    update_slots: Mutex<Option<UpdateSlotsCallback>>,
    terminated: AtomicBool,
}

impl Default for ServerQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerQueue {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(QueueState::default()),
            cv: Condvar::new(),
            new_task: Mutex::new(None),
            sleeping_callbacks: Mutex::new(Vec::new()),
            update_slots: Mutex::new(None),
            terminated: AtomicBool::new(false),
        }
    }

    pub fn on_new_task<F>(&self, callback: F)
    where
        F: Fn(ServerTask, bool) -> bool + Send + Sync + 'static,
    {
        *self.new_task.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(callback));
    }

    pub fn has_new_task_callback(&self) -> bool {
        self.new_task.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    pub fn on_update_slots<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        *self.update_slots.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(callback));
    }

    pub fn on_sleeping_state<F>(&self, callback: F)
    where
        F: Fn(bool) + Send + Sync + 'static,
    {
        self.sleeping_callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::new(callback));
    }

    /// `server_queue::get_new_id()`.
    pub fn get_new_id(&self) -> i32 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.id += 1;
        state.id
    }

    /// `server_queue::post()`.
    pub fn post(&self, mut task: ServerTask, front: bool) -> i32 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if task.id < 0 {
            state.id += 1;
            task.id = state.id;
        }
        let id = task.id;
        if front {
            state.tasks.push_front(task);
        } else {
            state.tasks.push_back(task);
        }
        self.cv.notify_one();
        id
    }

    fn post_many(&self, tasks: Vec<ServerTask>, front: bool) -> Vec<i32> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut ids = Vec::with_capacity(tasks.len());
        for mut task in tasks {
            if task.id < 0 {
                state.id += 1;
                task.id = state.id;
            }
            ids.push(task.id);
            if front {
                state.tasks.push_front(task);
            } else {
                state.tasks.push_back(task);
            }
        }
        self.cv.notify_all();
        ids
    }

    /// `server_queue::defer()`.
    pub fn defer(&self, mut task: ServerTask) -> i32 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if task.id < 0 {
            state.id += 1;
            task.id = state.id;
        }
        let id = task.id;
        state.deferred.push_back(task);
        id
    }

    /// `server_queue::pop_deferred_task()`.
    pub fn pop_deferred_task(&self, id_slot: i32) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let position = state
            .deferred
            .iter()
            .position(|t| t.id_slot == id_slot)
            .or_else(|| state.deferred.front().map(|_| 0));
        if let Some(idx) = position {
            if let Some(task) = state.deferred.remove(idx) {
                state.tasks.push_back(task);
                self.cv.notify_one();
                return true;
            }
        }
        false
    }

    /// `server_queue::is_sleeping()`.
    pub fn is_sleeping(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).sleeping
    }

    pub fn deferred_len(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).deferred.len()
    }

    /// `server_queue::wait_until_no_sleep()`.
    pub fn wait_until_no_sleep(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.sleeping {
            return;
        }
        state.req_stop_sleeping = true;
        state.n_inflight += 1;
        self.cv.notify_all();
        while state.sleeping && !self.terminated.load(Ordering::Relaxed) {
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// `server_queue::release_inflight()`.
    pub fn release_inflight(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.n_inflight = state.n_inflight.saturating_sub(1);
        self.cv.notify_all();
    }

    /// `server_queue::terminate()`.
    pub fn terminate(&self) {
        self.terminated.store(true, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.running = false;
        state.req_stop_sleeping = true;
        self.cv.notify_all();
    }

    fn notify_sleeping(&self, is_sleeping: bool) {
        let callbacks = self.sleeping_callbacks.lock().unwrap_or_else(|e| e.into_inner());
        if is_sleeping {
            for cb in callbacks.iter() {
                cb(true);
            }
        } else {
            for cb in callbacks.iter().rev() {
                cb(false);
            }
        }
    }

    fn enter_sleep<'a>(&self, mut state: MutexGuard<'a, QueueState>) -> MutexGuard<'a, QueueState> {
        state.sleeping = true;
        self.notify_sleeping(true);
        while !state.req_stop_sleeping && !self.terminated.load(Ordering::Relaxed) {
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.sleeping = false;
        self.notify_sleeping(false);
        state.req_stop_sleeping = false;
        state
    }

    fn process_new_tasks(&self, state: &mut QueueState, is_yielding: bool) -> bool {
        let callback = self.new_task.lock().unwrap_or_else(|e| e.into_inner()).clone();
        while let Some(task) = state.tasks.pop_front() {
            let accepted = match &callback {
                Some(cb) => cb(task.clone(), is_yielding),
                None => false,
            };
            if !accepted && is_yielding {
                state.unhandled.push_back(task);
            }
        }
        false
    }

    /// `server_queue::start_loop()`.
    pub fn start_loop(&self, idle_sleep_ms: i64) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.running = true;
        }
        let timeout = if idle_sleep_ms < 0 {
            None
        } else {
            Some(Duration::from_millis(idle_sleep_ms.max(1) as u64))
        };
        loop {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.running || self.terminated.load(Ordering::Relaxed) {
                break;
            }
            while !state.tasks.is_empty() {
                self.process_new_tasks(&mut state, false);
                if let Some(cb) = self.update_slots.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                    cb();
                }
            }
            if state.sleeping {
                state = self.enter_sleep(state);
                continue;
            }
            state = match timeout {
                Some(d) => self.cv.wait_timeout(state, d).unwrap_or_else(|e| e.into_inner()).0,
                None => self.cv.wait(state).unwrap_or_else(|e| e.into_inner()),
            };
            if timeout.is_some() && state.tasks.is_empty() && state.n_inflight == 0 && !self.terminated.load(Ordering::Relaxed) {
                state = self.enter_sleep(state);
            }
        }
    }

    /// `server_queue::yield_to_queue()`.
    pub fn yield_to_queue<F>(&self, work: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let handle = std::thread::spawn(work);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.process_new_tasks(&mut state, true);
        let unhandled: Vec<_> = state.unhandled.drain(..).collect();
        for task in unhandled {
            state.tasks.push_front(task);
        }
        drop(state);
        handle.join().map_err(|_| crate::core::Error::new("queue worker panicked"))
    }
}

#[derive(Default)]
struct ResponseState {
    running: bool,
    waiting: BTreeSet<i32>,
    results: VecDeque<TaskResult>,
}

/// `server_response`.
pub struct ServerResponse {
    state: Mutex<ResponseState>,
    cv: Condvar,
}

impl Default for ServerResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerResponse {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ResponseState { running: true, ..ResponseState::default() }),
            cv: Condvar::new(),
        }
    }

    pub fn add_waiting_task_id(&self, id: i32) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).waiting.insert(id);
    }

    pub fn add_waiting_task_ids(&self, ids: &BTreeSet<i32>) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).waiting.extend(ids.iter().copied());
    }

    pub fn remove_waiting_task_id(&self, id: i32) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).waiting.remove(&id);
    }

    pub fn remove_waiting_task_ids(&self, ids: &BTreeSet<i32>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for id in ids {
            state.waiting.remove(id);
        }
    }

    /// `server_response::send()`.
    pub fn send(&self, result: TaskResult) {
        let id = result_id(&result);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.waiting.contains(&id) {
            state.results.push_back(result);
            self.cv.notify_all();
        }
    }

    /// `server_response::broadcast()`.
    pub fn broadcast(&self, result: TaskResult) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.results.push_back(result);
        self.cv.notify_all();
    }

    fn recv_locked<'a>(&self, mut state: MutexGuard<'a, ResponseState>, ids: &BTreeSet<i32>) -> Option<TaskResult> {
        loop {
            if let Some(pos) = state.results.iter().position(|r| ids.contains(&result_id(r))) {
                return state.results.remove(pos);
            }
            if !state.running {
                return None;
            }
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// `server_response::recv()`.
    pub fn recv(&self, ids: &BTreeSet<i32>) -> Option<TaskResult> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.recv_locked(state, ids)
    }

    /// `server_response::recv_with_timeout()`.
    pub fn recv_with_timeout(&self, ids: &BTreeSet<i32>, timeout: Duration) -> Option<TaskResult> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(pos) = state.results.iter().position(|r| ids.contains(&result_id(r))) {
                return state.results.remove(pos);
            }
            if !state.running {
                return None;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let (next, _) = self
                .cv
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }
    }

    /// `server_response::terminate()`.
    pub fn terminate(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.running = false;
        self.cv.notify_all();
    }
}

/// Batch aggregation result of `ResponseReader::wait_for_all()`.
pub struct BatchResponse {
    pub is_terminated: bool,
    pub results: Vec<TaskResult>,
    pub error: Option<TaskResult>,
}

fn result_id(result: &TaskResult) -> i32 {
    match result {
        TaskResult::Completion(r) => r.id,
        TaskResult::CompletionPartial(r) => r.index as i32,
        TaskResult::Error { id, .. }
        | TaskResult::Metrics { id, .. }
        | TaskResult::Slots { id, .. }
        | TaskResult::SlotSaveLoad { id, .. }
        | TaskResult::SlotErase { id, .. }
        | TaskResult::Control { id, .. }
        | TaskResult::GetLora { id, .. }
        | TaskResult::ApplyLora { id }
        | TaskResult::Embedding { id, .. }
        | TaskResult::Rerank { id, .. }
        | TaskResult::Decision { id, .. }
        | TaskResult::Realtime { id, .. }
        | TaskResult::Tts { id, .. }
        | TaskResult::Router { id, .. } => *id,
    }
}

/// `server_response_reader`.
pub struct ResponseReader {
    pub id_tasks: BTreeSet<i32>,
    pub received_count: usize,
    pub cancelled: bool,
    pub polling_interval: Duration,
    queue: Arc<ServerQueue>,
    responses: Arc<ServerResponse>,
}

impl ResponseReader {
    pub fn new(queue: Arc<ServerQueue>, responses: Arc<ServerResponse>, polling_seconds: i32) -> Self {
        Self {
            id_tasks: BTreeSet::new(),
            received_count: 0,
            cancelled: false,
            polling_interval: Duration::from_secs(polling_seconds.max(0) as u64),
            queue,
            responses,
        }
    }

    pub fn get_new_id(&self) -> i32 {
        self.queue.get_new_id()
    }

    pub fn post_task(&mut self, mut task: ServerTask, front: bool) {
        if task.id < 0 {
            task.id = self.get_new_id();
        }
        self.id_tasks.insert(task.id);
        self.responses.add_waiting_task_id(task.id);
        self.queue.post(task, front);
    }

    pub fn post_tasks(&mut self, tasks: Vec<ServerTask>, front: bool) {
        let mut ids = BTreeSet::new();
        let mut prepared = Vec::new();
        for mut task in tasks {
            if task.id < 0 {
                task.id = self.get_new_id();
            }
            ids.insert(task.id);
            prepared.push(task);
        }
        self.responses.add_waiting_task_ids(&ids);
        self.id_tasks.extend(ids);
        self.queue.post_many(prepared, front);
    }

    pub fn has_next(&self) -> bool {
        !self.id_tasks.is_empty() && !self.cancelled
    }

    /// `server_response_reader::next()`.
    pub fn next(&mut self, should_stop: &dyn Fn() -> bool) -> Option<TaskResult> {
        loop {
            if should_stop() {
                self.stop();
                return None;
            }
            if self.id_tasks.is_empty() {
                return None;
            }
            if let Some(result) = self.responses.recv_with_timeout(&self.id_tasks, self.polling_interval) {
                self.received_count += 1;
                if result.is_error() {
                    self.id_tasks.clear();
                    self.cancelled = true;
                }
                return Some(result);
            }
        }
    }

    pub fn wait_for_all(&mut self, should_stop: &dyn Fn() -> bool) -> BatchResponse {
        let mut out = BatchResponse { is_terminated: false, results: Vec::new(), error: None };
        while !self.id_tasks.is_empty() {
            match self.next(should_stop) {
                Some(result) => {
                    if result.is_error() {
                        out.error = Some(result);
                        break;
                    }
                    out.results.push(result);
                }
                None => {
                    out.is_terminated = !self.id_tasks.is_empty();
                    break;
                }
            }
        }
        out
    }

    pub fn stop(&mut self) {
        self.cancelled = true;
        let ids = std::mem::take(&mut self.id_tasks);
        self.responses.remove_waiting_task_ids(&ids);
    }
}

impl Drop for ResponseReader {
    fn drop(&mut self) {
        self.stop();
    }
}
