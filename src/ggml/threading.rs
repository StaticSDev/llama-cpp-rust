//! Port of `ggml-threading.h` / `ggml-threading.cpp`.
//!
//! The reference source only defines the global critical-section mutex. The
//! thread pool API declared in `ggml.h` / `ggml-cpu.h` is ported here too:
//! callback jobs run on a fixed set of `std::thread` workers. CPU affinity,
//! scheduling priority and graph barriers are represented by parameters but
//! not applied, since std has no API for them.

use crate::core::{Error, Result};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

/// `GGML_MAX_N_THREADS`.
pub const GGML_MAX_N_THREADS: usize = 512;

/// `GGML_DEFAULT_N_THREADS`.
pub const GGML_DEFAULT_N_THREADS: usize = 4;

/// Default polling level set by `ggml_threadpool_params_init`.
pub const DEFAULT_POLL: u32 = 50;

// ---------------------------------------------------------------------------
// critical section
// ---------------------------------------------------------------------------

static CRITICAL_SECTION: Mutex<()> = Mutex::new(());

thread_local! {
    // The C++ API has no token to return a guard, so keep it in thread-local
    // storage for the duration of the section.
    static SECTION_GUARD: RefCell<Option<MutexGuard<'static, ()>>> = RefCell::new(None);
}

/// `ggml_critical_section_start`.
///
/// A nested start reuses the held guard instead of deadlocking as
/// `std::mutex::lock` would.
pub fn critical_section_start() {
    SECTION_GUARD.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            // C++ std::mutex has no poison state; recover as lock() would.
            *slot = Some(CRITICAL_SECTION.lock().unwrap_or_else(|e| e.into_inner()));
        }
    });
}

/// `ggml_critical_section_end`.
pub fn critical_section_end() {
    SECTION_GUARD.with(|slot| {
        drop(slot.borrow_mut().take());
    });
}

// ---------------------------------------------------------------------------
// thread pool params
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedPriority {
    Low = -1,
    Normal = 0,
    Medium = 1,
    High = 2,
    Realtime = 3,
}

/// `ggml_threadpool_params`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadpoolParams {
    pub cpumask: Vec<bool>,
    pub n_threads: usize,
    pub prio: SchedPriority,
    pub poll: u32,
    pub strict_cpu: bool,
    pub paused: bool,
}

impl ThreadpoolParams {
    /// `ggml_threadpool_params_default`.
    pub fn new(n_threads: usize) -> Self {
        Self {
            cpumask: vec![false; GGML_MAX_N_THREADS],
            n_threads,
            prio: SchedPriority::Normal,
            poll: DEFAULT_POLL,
            strict_cpu: false,
            paused: false,
        }
    }

    /// `ggml_threadpool_params_init`.
    pub fn init(&mut self, n_threads: usize) {
        *self = Self::new(n_threads);
    }

    /// `ggml_threadpool_params_match`. `paused` is not part of the C++ match.
    pub fn matches(&self, other: &Self) -> bool {
        self.n_threads == other.n_threads
            && self.prio == other.prio
            && self.poll == other.poll
            && self.strict_cpu == other.strict_cpu
            && self.cpumask == other.cpumask
    }
}

/// Best guess for the CPU count, falling back to `GGML_DEFAULT_N_THREADS`.
pub fn max_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(GGML_DEFAULT_N_THREADS)
}

// ---------------------------------------------------------------------------
// thread pool
// ---------------------------------------------------------------------------

/// One callback per job; the argument is the worker index.
pub type Job = Box<dyn FnOnce(usize) + Send>;

struct State {
    jobs: VecDeque<Job>,
    pending: usize,
    paused: bool,
    stop: bool,
}

struct Inner {
    state: Mutex<State>,
    ready: Condvar,
    done: Condvar,
}

/// `ggml_threadpool` with a generic callback queue in place of a compute graph.
pub struct ThreadPool {
    n_threads: usize,
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// `ggml_threadpool_new`.
    pub fn new(params: &ThreadpoolParams) -> Result<Self> {
        if params.n_threads == 0 || params.n_threads > GGML_MAX_N_THREADS {
            return Err(Error::new("n_threads out of range"));
        }

        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                pending: 0,
                paused: params.paused,
                stop: false,
            }),
            ready: Condvar::new(),
            done: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(params.n_threads);
        for ith in 0..params.n_threads {
            let inner = Arc::clone(&inner);
            match thread::Builder::new()
                .name(format!("ggml-{ith}"))
                .spawn(move || worker(inner, ith))
            {
                Ok(handle) => workers.push(handle),
                Err(e) => {
                    stop_workers(&inner);
                    for handle in workers {
                        let _ = handle.join();
                    }
                    return Err(Error::new(format!("failed to spawn worker: {e}")));
                }
            }
        }

        Ok(Self {
            n_threads: params.n_threads,
            inner,
            workers,
        })
    }

    /// Build a pool from a thread count with default params.
    pub fn with_n_threads(n_threads: usize) -> Result<Self> {
        Self::new(&ThreadpoolParams::new(n_threads))
    }

    /// `ggml_threadpool_get_n_threads`.
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Queue jobs and wake all workers. Call `wait` for completion.
    pub fn run<I>(&self, jobs: I)
    where
        I: IntoIterator<Item = Job>,
    {
        let mut state = lock(&self.inner.state);
        for job in jobs {
            state.jobs.push_back(job);
            state.pending += 1;
        }
        drop(state);
        self.inner.ready.notify_all();
    }

    /// Run one callback on every worker, with its worker index.
    pub fn run_all<F>(&self, callback: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let callback = Arc::new(callback);
        self.run((0..self.n_threads).map(|ith| {
            let callback = Arc::clone(&callback);
            Box::new(move |_| callback(ith)) as Job
        }));
    }

    /// Block until all queued jobs have finished.
    pub fn wait(&self) {
        let mut state = lock(&self.inner.state);
        while state.pending > 0 {
            state = self.inner.done.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Queue work and block until it finishes.
    pub fn run_and_wait<I>(&self, jobs: I)
    where
        I: IntoIterator<Item = Job>,
    {
        self.run(jobs);
        self.wait();
    }

    /// `ggml_threadpool_pause`.
    pub fn pause(&self) {
        lock(&self.inner.state).paused = true;
    }

    /// `ggml_threadpool_resume`.
    pub fn resume(&self) {
        lock(&self.inner.state).paused = false;
        self.inner.ready.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        lock(&self.inner.state).paused
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        stop_workers(&self.inner);
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn stop_workers(inner: &Inner) {
    let mut state = lock(&inner.state);
    state.stop = true;
    state.pending = 0;
    state.jobs.clear();
    drop(state);
    inner.ready.notify_all();
    inner.done.notify_all();
}

fn worker(inner: Arc<Inner>, ith: usize) {
    loop {
        let mut state = lock(&inner.state);
        while !state.stop && (state.paused || state.jobs.is_empty()) {
            state = inner.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        if state.stop {
            return;
        }
        let job = state.jobs.pop_front().expect("worker woke with an empty queue");
        drop(state);

        // A panicking job must not leave `wait` blocked forever.
        let _ = catch_unwind(AssertUnwindSafe(|| job(ith)));

        let mut state = lock(&inner.state);
        // Drop may have cleared the queue while this job was already running.
        state.pending = state.pending.saturating_sub(1);
        if state.pending == 0 {
            inner.done.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn params_match_ignores_paused() {
        let params = ThreadpoolParams::new(4);
        assert_eq!(params.prio, SchedPriority::Normal);
        assert_eq!(params.poll, DEFAULT_POLL);
        assert!(!params.strict_cpu);
        assert_eq!(params.cpumask.len(), GGML_MAX_N_THREADS);

        let mut other = params.clone();
        other.paused = true;
        assert!(params.matches(&other));

        other.poll = 0;
        assert!(!params.matches(&other));
    }

    #[test]
    fn critical_section_start_end_pairs() {
        let total = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let total = Arc::clone(&total);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    critical_section_start();
                    total.fetch_add(1, Ordering::Relaxed);
                    critical_section_end();
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker panicked");
        }
        assert_eq!(total.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn threadpool_runs_one_job_per_worker() {
        let pool = ThreadPool::with_n_threads(4).expect("pool creation failed");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let count = Arc::new(AtomicUsize::new(0));
        {
            let seen = Arc::clone(&seen);
            let count = Arc::clone(&count);
            pool.run_all(move |ith| {
                seen.lock().unwrap().push(ith);
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        pool.wait();
        assert_eq!(count.load(Ordering::Relaxed), 4);
        let mut ids = seen.lock().unwrap().clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn threadpool_pause_and_resume() {
        let pool = ThreadPool::with_n_threads(2).expect("pool creation failed");
        pool.pause();
        assert!(pool.is_paused());
        let count = Arc::new(AtomicUsize::new(0));
        {
            let count = Arc::clone(&count);
            pool.run(vec![Box::new(move |_| {
                count.fetch_add(1, Ordering::Relaxed);
            }) as Job]);
        }
        pool.resume();
        pool.wait();
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
