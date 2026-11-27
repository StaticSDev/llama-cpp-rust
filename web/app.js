const form = document.getElementById('composer');
const textarea = document.getElementById('prompt');
const sendButton = document.getElementById('send');
const messages = document.getElementById('messages');
const empty = document.getElementById('empty');
const statusText = document.getElementById('status-text');
const dot = document.getElementById('dot');
const model = document.getElementById('model');
const headNote = document.getElementById('head-note');

const ui = {
  prompt: document.getElementById('t-prompt'),
  gen: document.getElementById('t-gen'),
  speed: document.getElementById('t-speed'),
  elapsed: document.getElementById('t-elapsed'),
  trace: document.getElementById('trace-fill'),
  mPrompt: document.getElementById('m-prompt'),
  mGen: document.getElementById('m-gen'),
  mPrefill: document.getElementById('m-prefill'),
  mDecode: document.getElementById('m-decode'),
};

const options = { n_predict: 256, temp: 0, top_p: 0.9 };

let waiting = false;
let ready = false;
let ticker = 0;

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function autoGrow() {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
}

function rate(count, seconds) {
  if (!(seconds > 0.05)) return '0.0';
  return (count / seconds).toFixed(1);
}

function secs(ms) {
  return (ms / 1000).toFixed(1) + 's';
}

function setWaiting(value) {
  waiting = value;
  sendButton.disabled = value;
  form.setAttribute('aria-busy', String(value));
  dot.classList.toggle('busy', value);
  dot.classList.toggle('ready', !value && ready);
  statusText.textContent = value ? 'Working' : ready ? 'Ready' : 'Loading';
}

function addUser(text) {
  empty.remove();
  const item = document.createElement('article');
  item.className = 'message user';
  const body = document.createElement('div');
  body.className = 'text';
  body.textContent = text;
  item.append(body);
  messages.appendChild(item);
  scrollToBottom();
}

function startReply() {
  empty.remove();
  const item = document.createElement('article');
  item.className = 'message assistant';

  const meta = document.createElement('div');
  meta.className = 'meta';
  const name = document.createElement('span');
  name.textContent = 'Assistant';
  const run = document.createElement('span');
  run.className = 'run';
  run.textContent = 'Reading prompt';
  meta.append(name, run);

  const reasoning = document.createElement('div');
  reasoning.className = 'reasoning';
  reasoning.hidden = true;

  const body = document.createElement('div');
  body.className = 'text';

  item.append(meta, reasoning, body);
  messages.appendChild(item);
  scrollToBottom();
  return { item, run, reasoning, body, raw: '' };
}

function renderReply(view) {
  const raw = view.raw;
  if (raw.startsWith('<think>')) {
    const end = raw.indexOf('</think>');
    if (end === -1) {
      view.reasoning.textContent = raw.slice(7).replace(/^\n+/, '');
      view.body.textContent = '';
    } else {
      view.reasoning.textContent = raw.slice(7, end).replace(/^\n+/, '');
      view.body.textContent = raw.slice(end + 8).replace(/^\n+/, '');
    }
  } else {
    view.body.textContent = raw;
  }
  view.reasoning.hidden = !view.reasoning.textContent.trim();
  scrollToBottom();
}

function resetTelemetry() {
  ui.prompt.textContent = '0 tok';
  ui.gen.textContent = '0 tok';
  ui.speed.textContent = '— tok/s';
  ui.elapsed.textContent = '0.0s';
  ui.trace.style.width = '0%';
  headNote.textContent = 'Reading prompt';
}

async function sendPrompt(prompt) {
  if (waiting || !prompt) return;

  setWaiting(true);
  addUser(prompt);
  textarea.value = '';
  resetTelemetry();

  const view = startReply();
  const start = performance.now();
  let firstTokenAt = 0;
  let tokens = 0;

  ticker = window.setInterval(() => {
    const base = firstTokenAt || start;
    ui.elapsed.textContent = ((performance.now() - base) / 1000).toFixed(1) + 's';
  }, 250);

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, ...options }),
    });

    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => null);
      throw new Error(data && data.error ? data.error : 'Request failed (' + response.status + ')');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const apply = (event) => {
      if (event.type === 'prompt') {
        const elapsed = (performance.now() - start) / 1000;
        const percent = Math.round((event.done / event.total) * 100);
        ui.trace.style.width = percent + '%';
        ui.prompt.textContent = event.done + ' tok';
        ui.speed.textContent = rate(event.done, elapsed) + ' tok/s';
        ui.elapsed.textContent = elapsed.toFixed(1) + 's';
        view.run.textContent = 'Reading prompt ' + event.done + '/' + event.total;
        headNote.textContent = 'Reading prompt ' + percent + '%';
      } else if (event.type === 'token') {
        if (!firstTokenAt) {
          firstTokenAt = performance.now();
          ui.trace.style.width = '100%';
        }
        tokens += 1;
        view.raw += event.text;
        renderReply(view);
        const elapsed = (performance.now() - firstTokenAt) / 1000;
        ui.gen.textContent = tokens + ' tok';
        ui.speed.textContent = rate(tokens, elapsed) + ' tok/s';
        ui.elapsed.textContent = elapsed.toFixed(1) + 's';
        view.run.textContent = tokens + ' tok · ' + rate(tokens, elapsed) + ' tok/s';
        headNote.textContent = 'Generating';
      } else if (event.type === 'done') {
        ui.prompt.textContent = event.prompt_tokens + ' tok';
        ui.gen.textContent = event.tokens + ' tok';
        ui.speed.textContent = event.tok_s.toFixed(1) + ' tok/s';
        ui.elapsed.textContent = secs(event.gen_ms);
        ui.trace.style.width = '100%';
        ui.mPrompt.textContent = event.prompt_tokens + ' tok';
        ui.mGen.textContent = event.tokens + ' tok';
        ui.mPrefill.textContent = event.prompt_tok_s.toFixed(1) + ' tok/s';
        ui.mDecode.textContent = event.tok_s.toFixed(1) + ' tok/s';
        view.run.textContent = 'Prefill ' + event.prompt_tok_s.toFixed(1) + ' · decode ' + event.tok_s.toFixed(1) + ' tok/s';
        headNote.textContent = 'Done in ' + secs(event.gen_ms);
      } else if (event.type === 'error') {
        throw new Error(event.error || 'Generation failed.');
      }
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, cut).trim();
        buffer = buffer.slice(cut + 1);
        if (line) apply(JSON.parse(line));
      }
    }

    if (!view.body.textContent.trim() && !view.reasoning.textContent.trim()) {
      throw new Error('The model returned an empty reply.');
    }
  } catch (error) {
    view.item.classList.add('error');
    view.run.textContent = '';
    view.reasoning.hidden = true;
    view.reasoning.textContent = '';
    view.body.textContent = error.message || 'Something went wrong.';
    headNote.textContent = 'Stopped';
  } finally {
    window.clearInterval(ticker);
    ticker = 0;
    setWaiting(false);
    scrollToBottom();
    textarea.focus();
    loadStatus();
  }
}

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) return;
    const data = await response.json();
    if (typeof data.model === 'string' && data.model) {
      model.textContent = data.model.split('/').pop().replace(/[^.0-9A-Za-z_-]/g, ' ');
      model.title = data.model;
    }
    ready = data.ready === true;
    dot.classList.toggle('ready', ready && !waiting);
    if (!waiting) {
      statusText.textContent = ready ? 'Ready' : 'Loading';
    }
  } catch (_) {
    // Keep the gray dot: status is unknown until the backend answers.
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const prompt = textarea.value.trim();
  if (prompt) {
    sendPrompt(prompt);
  }
});

textarea.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

textarea.addEventListener('input', autoGrow);

autoGrow();
loadStatus();
