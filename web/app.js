const form = document.getElementById('composer');
const textarea = document.getElementById('prompt');
const sendButton = document.getElementById('send');
const messages = document.getElementById('messages');
const empty = document.getElementById('empty');
const statusText = document.getElementById('status-text');
const dot = document.getElementById('dot');
const model = document.getElementById('model');

const options = { n_predict: 256, temp: 0, top_p: 0.9 };

let waiting = false;
let ready = false;

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function autoGrow() {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  messages.style.paddingBottom = form.offsetHeight + 32 + 'px';
}

function setWaiting(value) {
  waiting = value;
  sendButton.disabled = value;
  form.setAttribute('aria-busy', String(value));
  dot.classList.toggle('busy', value);
  statusText.textContent = value ? 'Thinking' : ready ? 'Ready' : 'Loading';
}

function addMessage(role, text, isError = false) {
  empty.remove();

  const item = document.createElement('article');
  item.className = 'message ' + role + (isError ? ' error' : '');

  const meta = document.createElement('div');
  meta.className = 'meta';
  const name = document.createElement('span');
  name.textContent = isError ? 'Error' : role === 'user' ? 'You' : 'Assistant';
  meta.append(name);

  const body = document.createElement('div');
  body.className = 'text';
  body.textContent = text;

  item.append(meta, body);
  messages.appendChild(item);
  scrollToBottom();
  return { item, meta, body };
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

  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('i');
  bar.append(fill);

  const body = document.createElement('div');
  body.className = 'text';

  item.append(meta, bar, body);
  messages.appendChild(item);
  scrollToBottom();
  return { item, run, bar, fill, body };
}

function rate(count, seconds) {
  if (!(seconds > 0.05)) return '0.0 tok/s';
  return (count / seconds).toFixed(1) + ' tok/s';
}

function seconds(ms) {
  return (ms / 1000).toFixed(1) + 's';
}

function fail(view, message) {
  view.item.classList.add('error');
  view.run.textContent = '';
  view.bar.hidden = true;
  view.body.textContent = message;
}

async function sendPrompt(prompt) {
  if (waiting || !prompt) return;

  setWaiting(true);
  addMessage('user', prompt);
  textarea.value = '';
  autoGrow();

  const view = startReply();
  const start = performance.now();
  let firstTokenAt = 0;
  let tokens = 0;
  let prefillDone = false;

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
        prefillDone = false;
        const elapsed = (performance.now() - start) / 1000;
        view.fill.style.width = Math.round((event.done / event.total) * 100) + '%';
        view.run.textContent =
          'Reading prompt ' + event.done + '/' + event.total + ' · ' + rate(event.done, elapsed);
      } else if (event.type === 'token') {
        if (!prefillDone) {
          prefillDone = true;
          firstTokenAt = performance.now();
          view.bar.hidden = true;
        }
        tokens += 1;
        view.body.textContent += event.text;
        const elapsed = (performance.now() - firstTokenAt) / 1000;
        view.run.textContent = 'Writing · ' + tokens + ' tok · ' + rate(tokens, elapsed);
        scrollToBottom();
      } else if (event.type === 'done') {
        view.bar.hidden = true;
        view.run.textContent =
          'prefill ' +
          event.prompt_tokens +
          ' tok · ' +
          (event.prompt_tok_s || 0).toFixed(1) +
          ' tok/s · ' +
          seconds(event.prompt_ms) +
          ' — gen ' +
          event.tokens +
          ' tok · ' +
          (event.tok_s || 0).toFixed(1) +
          ' tok/s · ' +
          seconds(event.gen_ms);
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

    if (!view.body.textContent.trim()) {
      throw new Error('The model returned an empty reply.');
    }
  } catch (error) {
    fail(view, error.message || 'Something went wrong.');
  } finally {
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
      model.title = data.model;
    }
    ready = data.ready === true;
    dot.classList.toggle('ready', ready);
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
