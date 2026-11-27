const form = document.getElementById('composer');
const textarea = document.getElementById('prompt');
const sendButton = document.getElementById('send');
const messages = document.getElementById('messages');
const empty = document.getElementById('empty');
const statusText = document.getElementById('status-text');
const dot = document.getElementById('dot');
const model = document.getElementById('model');

const options = { n_predict: 64, temp: 0, top_p: 0.9 };

let waiting = false;
let ready = false;

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function autoGrow() {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
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
  meta.textContent = isError ? 'Error' : role === 'user' ? 'You' : 'Assistant';

  const body = document.createElement('div');
  body.className = 'text';
  body.textContent = text;

  item.append(meta, body);
  messages.appendChild(item);
  scrollToBottom();
}

function showTyping() {
  const indicator = document.createElement('div');
  indicator.className = 'typing';
  indicator.setAttribute('aria-hidden', 'true');
  indicator.innerHTML = '<span></span><span></span><span></span>';
  messages.appendChild(indicator);
  scrollToBottom();
  return indicator;
}

async function sendPrompt(prompt) {
  if (waiting || !prompt) return;

  setWaiting(true);
  addMessage('user', prompt);
  textarea.value = '';
  autoGrow();
  const typing = showTyping();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, ...options }),
    });
    const data = await response.json().catch(() => null);

    if (!response.ok || !data) {
      throw new Error(data && data.error ? data.error : 'Request failed (' + response.status + ')');
    }
    if (data.error) {
      throw new Error(data.error);
    }
    addMessage('assistant', typeof data.text === 'string' ? data.text : '');
  } catch (error) {
    addMessage('assistant', error.message || 'Something went wrong.', true);
  } finally {
    typing.remove();
    setWaiting(false);
    scrollToBottom();
    textarea.focus();
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
