(() => {
  'use strict';

  const STORE_KEY = 'local-inference.chats.v1';
  const OPTIONS = { n_predict: 256, temp: 0, top_p: 0.9 };
  const TITLE_MAX = 36;

  const $ = (id) => document.getElementById(id);
  const el = {
    sidebar: $('sidebar'),
    scrim: $('scrim'),
    drawerToggle: $('drawer-toggle'),
    newChat: $('new-chat'),
    chatList: $('chat-list'),
    model: $('model'),
    statusText: $('status-text'),
    deviceSquare: $('device-square'),
    gpu: $('gpu'),
    headTitle: $('head-title'),
    headNote: $('head-note'),
    messages: $('messages'),
    empty: $('empty'),
    form: $('composer'),
    textarea: $('prompt'),
    send: $('send'),
    segments: $('segments'),
    tPrefill: $('t-prefill'),
    tGen: $('t-gen'),
    tSpeed: $('t-speed'),
    tElapsed: $('t-elapsed'),
    tState: $('t-state'),
    tFinal: $('t-final'),
  };
  const segmentEls = Array.from(el.segments.children);

  let state = loadState() || freshState();
  let waiting = false;
  let ready = false;
  let confirmingId = null;
  let messageViews = new Map();
  let live = blankTelemetry();
  let ticker = 0;
  let saveTimer = 0;

  function uid() {
    return Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
  }

  function freshChat() {
    const now = Date.now();
    return { id: uid(), title: 'New chat', createdAt: now, updatedAt: now, messages: [], telemetry: null };
  }

  function freshState() {
    const chat = freshChat();
    return { activeId: chat.id, chats: [chat] };
  }

  function clamp01(n) {
    return Math.min(1, Math.max(0, n));
  }

  function blankTelemetry() {
    return {
      phase: 'idle', prefillPct: 0, promptTokens: 0, promptTotal: 0,
      tokens: 0, tokS: 0, promptTokS: 0, promptMs: 0, genMs: 0, elapsedMs: 0,
    };
  }

  function normalizeTelemetry(t) {
    const phases = ['idle', 'prefill', 'decode', 'done', 'error'];
    return {
      phase: phases.includes(t.phase) ? t.phase : 'idle',
      prefillPct: clamp01(Number(t.prefillPct) || 0),
      promptTokens: Math.max(0, Number(t.promptTokens) || 0),
      promptTotal: Math.max(0, Number(t.promptTotal) || 0),
      tokens: Math.max(0, Number(t.tokens) || 0),
      tokS: Math.max(0, Number(t.tokS) || 0),
      promptTokS: Math.max(0, Number(t.promptTokS) || 0),
      promptMs: Math.max(0, Number(t.promptMs) || 0),
      genMs: Math.max(0, Number(t.genMs) || 0),
      elapsedMs: Math.max(0, Number(t.elapsedMs) || 0),
    };
  }

  function loadState() {
    let data;
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return null;
      data = JSON.parse(raw);
    } catch (_) {
      return null;
    }
    if (!data || !Array.isArray(data.chats)) return null;

    const chats = data.chats.filter((c) => c && typeof c === 'object').map((c) => ({
      id: typeof c.id === 'string' ? c.id : uid(),
      title: typeof c.title === 'string' ? c.title : 'New chat',
      createdAt: Number(c.createdAt) || Date.now(),
      updatedAt: Number(c.updatedAt) || Date.now(),
      messages: Array.isArray(c.messages) ? c.messages.filter((m) => m && typeof m === 'object').map((m) => ({
        id: typeof m.id === 'string' ? m.id : uid(),
        role: m.role === 'user' || m.role === 'error' ? m.role : 'assistant',
        text: typeof m.text === 'string' ? m.text : '',
        reasoning: typeof m.reasoning === 'string' ? m.reasoning : '',
        createdAt: Number(m.createdAt) || Date.now(),
      })) : [],
      telemetry: c.telemetry && typeof c.telemetry === 'object' ? normalizeTelemetry(c.telemetry) : null,
    }));
    if (!chats.length) chats.push(freshChat());
    const activeId = chats.some((c) => c.id === data.activeId) ? data.activeId : chats[0].id;
    return { activeId, chats };
  }

  function save() {
    window.clearTimeout(saveTimer);
    saveTimer = 0;
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch (_) {
      // Storage may be full or blocked; the in-memory state still works.
    }
  }

  function scheduleSave() {
    if (!saveTimer) saveTimer = window.setTimeout(save, 400);
  }

  function activeChat() {
    return state.chats.find((c) => c.id === state.activeId) || null;
  }

  function rate(count, seconds) {
    return seconds > 0.05 ? (count / seconds).toFixed(1) : '0.0';
  }

  function secs(ms) {
    return (ms / 1000).toFixed(1) + 's';
  }

  function stripLead(s) {
    return s.replace(/^\r?\n+/, '');
  }

  function splitReply(raw) {
    const open = raw.indexOf('<think>');
    if (open === -1) return { reasoning: '', answer: stripLead(raw) };
    const close = raw.indexOf('</think>', open + 7);
    if (close === -1) return { reasoning: stripLead(raw.slice(open + 7)), answer: '' };
    return {
      reasoning: stripLead(raw.slice(open + 7, close)),
      answer: stripLead(raw.slice(close + 8)),
    };
  }

  function titleFrom(text) {
    const one = text.replace(/\s+/g, ' ').trim();
    if (!one) return 'New chat';
    return one.length > TITLE_MAX ? one.slice(0, TITLE_MAX - 1) + '…' : one;
  }

  function clockTime(ts) {
    const d = new Date(ts);
    return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }

  function relTime(ts) {
    const diff = Date.now() - ts;
    if (diff < 60000) return 'now';
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return mins + 'm';
    const hours = Math.floor(mins / 60);
    if (hours < 24) return hours + 'h';
    const days = Math.floor(hours / 24);
    if (days < 7) return days + 'd';
    const d = new Date(ts);
    return String(d.getDate()).padStart(2, '0') + '/' + String(d.getMonth() + 1).padStart(2, '0');
  }

  function scrollBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  function autoGrow() {
    el.textarea.style.height = 'auto';
    el.textarea.style.height = Math.min(el.textarea.scrollHeight, 140) + 'px';
  }

  function syncSend() {
    el.send.disabled = waiting || !el.textarea.value.trim();
  }

  function paintStatus() {
    el.statusText.textContent = waiting ? 'Working' : ready ? 'Ready' : 'Loading';
    el.deviceSquare.classList.toggle('ready', !waiting && ready);
    el.deviceSquare.classList.toggle('busy', waiting);
  }

  function setWaiting(value) {
    waiting = value;
    el.form.setAttribute('aria-busy', String(value));
    syncSend();
    paintStatus();
    renderChatList();
  }

  function setDrawer(open) {
    el.sidebar.classList.toggle('open', open);
    el.scrim.classList.toggle('show', open);
    el.drawerToggle.setAttribute('aria-expanded', String(open));
    el.drawerToggle.setAttribute('aria-label', open ? 'Close chats' : 'Open chats');
  }

  function paintTelemetry(t) {
    const filled = Math.round(clamp01(t.prefillPct) * segmentEls.length);
    segmentEls.forEach((seg, i) => seg.classList.toggle('on', i < filled));

    el.tPrefill.textContent = t.promptTotal > 0
      ? t.promptTokens + ' / ' + t.promptTotal + ' tok'
      : t.promptTokens > 0 ? t.promptTokens + ' tok' : '0 / 0 tok';
    el.tGen.textContent = t.tokens + ' tok';
    el.tSpeed.textContent = t.phase === 'idle' ? '— tok/s' : t.tokS.toFixed(1) + ' tok/s';
    el.tElapsed.textContent = (t.elapsedMs / 1000).toFixed(1) + 's';
    el.tState.textContent = {
      idle: 'Idle', prefill: 'Prefill', decode: 'Generating', done: 'Done', error: 'Error',
    }[t.phase] || 'Idle';

    if (t.phase === 'done') {
      el.tFinal.textContent = 'Prefill ' + t.promptTokS.toFixed(1) + ' tok/s · ' + secs(t.promptMs)
        + ' · Decode ' + t.tokS.toFixed(1) + ' tok/s · ' + secs(t.genMs);
      el.tFinal.hidden = false;
    } else {
      el.tFinal.hidden = true;
    }
  }

  function renderChatList() {
    el.chatList.replaceChildren();
    for (const chat of state.chats) {
      const row = document.createElement('div');
      row.className = 'chat-row';
      if (chat.id === state.activeId) row.classList.add('active');
      if (chat.id === confirmingId) row.classList.add('confirming');

      if (chat.id === confirmingId) {
        const question = document.createElement('span');
        question.className = 'confirm-text';
        question.textContent = 'Delete this chat?';
        const actions = document.createElement('div');
        actions.className = 'confirm-actions';

        const yes = document.createElement('button');
        yes.type = 'button';
        yes.className = 'confirm-yes';
        yes.textContent = 'Delete';
        yes.addEventListener('click', () => deleteChat(chat.id));

        const no = document.createElement('button');
        no.type = 'button';
        no.textContent = 'Cancel';
        no.addEventListener('click', () => {
          confirmingId = null;
          renderChatList();
        });

        actions.append(yes, no);
        row.append(question, actions);
      } else {
        const open = document.createElement('button');
        open.type = 'button';
        open.className = 'chat-open';
        open.disabled = waiting;
        if (chat.id === state.activeId) open.setAttribute('aria-current', 'true');

        const title = document.createElement('span');
        title.className = 'chat-title';
        title.textContent = chat.title;
        const time = document.createElement('span');
        time.className = 'chat-time mono';
        time.textContent = relTime(chat.updatedAt);
        open.append(title, time);
        open.addEventListener('click', () => switchChat(chat.id));

        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'chat-delete';
        del.textContent = '×';
        del.title = 'Delete chat';
        del.setAttribute('aria-label', 'Delete chat: ' + chat.title);
        del.disabled = waiting;
        del.addEventListener('click', () => {
          confirmingId = chat.id;
          renderChatList();
        });

        row.append(open, del);
      }
      el.chatList.appendChild(row);
    }
  }

  function renderMessages() {
    messageViews = new Map();
    el.messages.replaceChildren();
    const chat = activeChat();

    if (!chat || !chat.messages.length) {
      el.messages.appendChild(el.empty);
      return;
    }

    for (const msg of chat.messages) {
      const item = document.createElement('article');
      item.className = 'message ' + msg.role;

      const meta = document.createElement('div');
      meta.className = 'meta';
      const who = document.createElement('span');
      who.className = 'who';
      who.textContent = msg.role === 'user' ? 'You' : msg.role === 'error' ? 'Error' : 'Assistant';
      const stamp = document.createElement('span');
      stamp.className = msg.role === 'user' ? 'time mono' : 'run mono';
      stamp.textContent = msg.role === 'user' ? clockTime(msg.createdAt) : '';
      meta.append(who, stamp);

      if (msg.role === 'assistant' || msg.role === 'error') {
        const wrap = document.createElement('div');
        wrap.className = 'reasoning';
        const label = document.createElement('span');
        label.className = 'reasoning-label mono';
        label.textContent = 'Reasoning';
        const reasoningText = document.createElement('div');
        reasoningText.className = 'reasoning-text';
        wrap.append(label, reasoningText);

        const body = document.createElement('div');
        body.className = 'text';
        item.append(meta, wrap, body);

        const parts = splitReply(msg.text);
        reasoningText.textContent = parts.reasoning;
        wrap.hidden = !parts.reasoning;
        body.textContent = parts.answer;
        messageViews.set(msg.id, { run: stamp, reasoning: wrap, reasoningText, body });
      } else {
        const body = document.createElement('div');
        body.className = 'text';
        body.textContent = msg.text;
        item.append(meta, body);
      }

      el.messages.appendChild(item);
    }
    scrollBottom();
  }

  function renderChat() {
    const chat = activeChat();
    el.headTitle.textContent = chat ? chat.title : 'New chat';
    renderChatList();
    renderMessages();

    if (chat && chat.telemetry && chat.telemetry.phase === 'done') {
      const t = chat.telemetry;
      const views = Array.from(messageViews.values()).reverse();
      const last = views.find((v) => v.run);
      if (last) last.run.textContent = t.tokS.toFixed(1) + ' tok/s · ' + t.tokens + ' tok';
    }
    paintTelemetry(chat && chat.telemetry ? chat.telemetry : live);
    syncSend();
  }

  function newChat() {
    if (waiting) return;
    const chat = freshChat();
    state.chats.unshift(chat);
    state.activeId = chat.id;
    confirmingId = null;
    live = blankTelemetry();
    setDrawer(false);
    save();
    renderChat();
    el.textarea.focus();
  }

  function switchChat(id) {
    if (waiting || id === state.activeId) return;
    state.activeId = id;
    confirmingId = null;
    live = blankTelemetry();
    setDrawer(false);
    save();
    renderChat();
    el.textarea.focus();
  }

  function deleteChat(id) {
    if (waiting) return;
    const index = state.chats.findIndex((c) => c.id === id);
    if (index === -1) return;
    state.chats.splice(index, 1);
    if (!state.chats.length) state.chats.push(freshChat());
    if (state.activeId === id) {
      state.activeId = state.chats[Math.min(index, state.chats.length - 1)].id;
    }
    confirmingId = null;
    live = blankTelemetry();
    save();
    renderChat();
  }

  async function sendPrompt(prompt) {
    if (waiting || !prompt) return;
    const chat = activeChat();
    if (!chat) return;

    const now = Date.now();
    const userMsg = { id: uid(), role: 'user', text: prompt, reasoning: '', createdAt: now };
    const replyMsg = { id: uid(), role: 'assistant', text: '', reasoning: '', createdAt: now };
    chat.messages.push(userMsg, replyMsg);
    if (!chat.title || chat.title === 'New chat') chat.title = titleFrom(prompt);
    chat.updatedAt = now;

    el.textarea.value = '';
    autoGrow();
    live = blankTelemetry();
    live.phase = 'prefill';
    setWaiting(true);
    renderChat();
    paintTelemetry(live);
    scheduleSave();

    const start = performance.now();
    let firstTokenAt = 0;
    let frozen = false;

    window.clearInterval(ticker);
    ticker = window.setInterval(() => {
      if (frozen) return;
      live.elapsedMs = performance.now() - (firstTokenAt || start);
      paintTelemetry(live);
    }, 250);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, ...OPTIONS }),
      });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => null);
        throw new Error(data && data.error ? data.error : 'Request failed (' + response.status + ')');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const apply = (event) => {
        if (!event || typeof event !== 'object') return;

        if (event.type === 'prompt') {
          const done = Number(event.done) || 0;
          const total = Number(event.total) || 0;
          live.promptTokens = done;
          live.promptTotal = total;
          live.prefillPct = total > 0 ? clamp01(done / total) : 0;
          live.elapsedMs = performance.now() - start;
          live.tokS = Number(rate(done, live.elapsedMs / 1000));
          const view = messageViews.get(replyMsg.id);
          if (view) view.run.textContent = 'Reading prompt ' + done + '/' + total;
          el.headNote.textContent = 'Reading prompt ' + Math.round(live.prefillPct * 100) + '%';
          paintTelemetry(live);

        } else if (event.type === 'token') {
          if (!firstTokenAt) {
            firstTokenAt = performance.now();
            live.prefillPct = 1;
          }
          live.phase = 'decode';
          live.tokens += 1;
          replyMsg.text += typeof event.text === 'string' ? event.text : '';

          const parts = splitReply(replyMsg.text);
          replyMsg.reasoning = parts.reasoning;
          const view = messageViews.get(replyMsg.id);
          if (view) {
            view.reasoningText.textContent = parts.reasoning;
            view.reasoning.hidden = !parts.reasoning;
            view.body.textContent = parts.answer;
          }

          live.elapsedMs = performance.now() - firstTokenAt;
          live.tokS = Number(rate(live.tokens, live.elapsedMs / 1000));
          if (view) view.run.textContent = live.tokens + ' tok · ' + live.tokS.toFixed(1) + ' tok/s';
          el.headNote.textContent = 'Generating';
          paintTelemetry(live);
          scrollBottom();
          scheduleSave();

        } else if (event.type === 'done') {
          live.phase = 'done';
          live.promptTokens = Number(event.prompt_tokens) || 0;
          live.promptMs = Number(event.prompt_ms) || 0;
          live.promptTokS = Number(event.prompt_tok_s) || 0;
          live.tokens = Number(event.tokens) || 0;
          live.genMs = Number(event.gen_ms) || 0;
          live.tokS = Number(event.tok_s) || 0;
          live.elapsedMs = live.genMs;
          live.prefillPct = 1;
          frozen = true;

          const view = messageViews.get(replyMsg.id);
          if (view) view.run.textContent = live.tokS.toFixed(1) + ' tok/s · ' + live.tokens + ' tok';
          el.headNote.textContent = 'Done in ' + secs(live.genMs);
          paintTelemetry(live);

        } else if (event.type === 'error') {
          throw new Error(typeof event.error === 'string' && event.error ? event.error : 'Generation failed.');
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

      if (!replyMsg.text.trim()) {
        throw new Error('The model returned an empty reply.');
      }
      live.phase = 'done';
      chat.telemetry = { ...live };

    } catch (error) {
      live.phase = 'error';
      replyMsg.role = 'error';
      replyMsg.text = error && error.message ? error.message : 'Something went wrong.';
      replyMsg.reasoning = '';
      chat.telemetry = { ...live };
      el.headNote.textContent = 'Stopped';
      frozen = true;
      renderChat();
    } finally {
      window.clearInterval(ticker);
      ticker = 0;
      chat.updatedAt = Date.now();
      setWaiting(false);
      paintTelemetry(chat.telemetry || live);
      save();
      scrollBottom();
      el.textarea.focus();
      loadStatus();
    }
  }

  async function loadStatus() {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) return;
      const data = await response.json();
      if (!data) return;

      if (typeof data.model === 'string' && data.model) {
        const short = data.model.split('/').pop().replace(/[^.0-9A-Za-z_-]/g, ' ').trim();
        el.model.textContent = short || data.model;
        el.model.title = data.model;
      }
      if (typeof data.gpu === 'string' && data.gpu) {
        el.gpu.textContent = 'GPU · ' + data.gpu;
        el.gpu.hidden = false;
      } else {
        el.gpu.hidden = true;
      }
      ready = data.ready === true;
      paintStatus();
    } catch (_) {
      // Status stays unknown until the backend answers.
    }
  }

  el.form.addEventListener('submit', (event) => {
    event.preventDefault();
    const prompt = el.textarea.value.trim();
    if (prompt) sendPrompt(prompt);
  });

  el.textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      el.form.requestSubmit();
    }
  });

  el.textarea.addEventListener('input', () => {
    autoGrow();
    syncSend();
  });

  el.newChat.addEventListener('click', newChat);
  el.drawerToggle.addEventListener('click', () => setDrawer(!el.sidebar.classList.contains('open')));
  el.scrim.addEventListener('click', () => setDrawer(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && el.sidebar.classList.contains('open')) setDrawer(false);
  });

  autoGrow();
  renderChat();
  loadStatus();
})();
