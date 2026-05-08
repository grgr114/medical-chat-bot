const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const chatList = document.querySelector("#chat-list");
const systemInfo = document.querySelector("#system-info");
const indexButton = document.querySelector("#index-button");
const indexStatus = document.querySelector("#index-status");
const newChatButton = document.querySelector("#new-chat-button");

let sessionId = localStorage.getItem("medical-rag-session-id");
let activeSessionId = null;

marked.setOptions({
  breaks: true,
  gfm: true,
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons({
      attrs: {
        "aria-hidden": "true",
        focusable: "false",
      },
    });
  }
}

function icon(name) {
  return `<i data-lucide="${name}"></i>`;
}

function welcomeMarkup() {
  return `
    <article class="message assistant">
      <div class="assistant-avatar" aria-hidden="true">${icon("stethoscope")}</div>
      <div class="bubble markdown-body welcome-bubble">
        <strong>Готов искать по документации МИС.</strong>
        <span>Задайте вопрос, а ответ будет подкреплен найденными источниками.</span>
      </div>
    </article>
  `;
}

function showWelcome() {
  messages.innerHTML = welcomeMarkup();
  refreshIcons();
}

function renderMarkdown(text) {
  const template = document.createElement("template");
  template.innerHTML = marked.parse(text);
  try {
    renderMathInElement(
      template.content,
      {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
      }
    );
  } catch (e) {
    // KaTeX fallback: keep the markdown HTML if math rendering is unavailable.
  }

  template.content.querySelectorAll("a[href]").forEach((link) => {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });

  return template.innerHTML;
}

function normalizeResponseStats(raw, latencyFallbackMs = null) {
  const fallback = latencyFallbackMs != null && Number.isFinite(Number(latencyFallbackMs)) ? Number(latencyFallbackMs) : 0;
  if (!raw || typeof raw !== "object") {
    return { latency_ms: fallback, input_tokens: 0, output_tokens: 0 };
  }
  return {
    latency_ms: Number(raw.latency_ms ?? fallback),
    input_tokens: Number(raw.input_tokens ?? 0),
    output_tokens: Number(raw.output_tokens ?? 0),
  };
}

function addMessage(role, content, isMarkdown = false, isError = false, stats = null, latencyFallbackMs = null) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "assistant-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.innerHTML = icon(isError ? "triangle-alert" : "stethoscope");
    article.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = `bubble${isError ? " error" : ""}`;

  if (isMarkdown) {
    const markdownEl = document.createElement("div");
    markdownEl.className = "markdown-body";
    markdownEl.innerHTML = renderMarkdown(content);
    bubble.appendChild(markdownEl);
  } else {
    bubble.textContent = content;
  }

  if (role === "assistant" && !isError) {
    bubble.appendChild(renderResponseStats(normalizeResponseStats(stats, latencyFallbackMs)));
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  refreshIcons();
  scrollToBottom();
  return article;
}

function renderResponseStats(stats) {
  const latencyMs = Number(stats.latency_ms ?? 0);
  const inputTokens = Number(stats.input_tokens ?? 0);
  const outputTokens = Number(stats.output_tokens ?? 0);
  const totalTokens = inputTokens + outputTokens;
  const block = document.createElement("div");
  block.className = "response-stats";
  block.setAttribute("role", "status");
  block.innerHTML = `
    <span>${icon("timer")} ${formatLatency(latencyMs)}</span>
    <span>${icon("hash")} ${formatNumber(totalTokens)} токенов</span>
    <span class="response-stats-detail">${icon("arrow-down-to-line")} ${formatNumber(inputTokens)} вход · ${icon("arrow-up-from-line")} ${formatNumber(outputTokens)} выход</span>
  `;
  return block;
}

function formatLatency(latencyMs) {
  if (!Number.isFinite(latencyMs) || latencyMs <= 0) return "0 ms";
  if (latencyMs < 1000) return `${Math.round(latencyMs)} ms`;
  return `${(latencyMs / 1000).toFixed(1)} s`;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "0";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function addThinkingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant thinking";
  const bubble = document.createElement("div");
  bubble.className = "bubble markdown-body";
  bubble.innerHTML = `
    <span class="thinking-dots">
      <span></span><span></span><span></span>
    </span>
    <span>Ищу в документации...</span>
  `;
  const avatar = document.createElement("div");
  avatar.className = "assistant-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.innerHTML = icon("loader-circle");
  article.appendChild(avatar);
  article.appendChild(bubble);
  messages.appendChild(article);
  refreshIcons();
  scrollToBottom();
  return article;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function sourceTitleHtml(source, refNum) {
  const prefix = `[${refNum}] `;
  const label = source.source_ref || "";
  const safeLabel = escapeHtml(label);
  const url = source.doc_url;
  if (url && String(url).trim()) {
    const safeUrl = escapeHtml(String(url).trim());
    return `${escapeHtml(prefix)}<a class="source-title-link" href="${safeUrl}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`;
  }
  return `${escapeHtml(prefix)}${safeLabel}`;
}

function renderSources(items, citedSourceIndices) {
  if (!items.length) return;

  const citedOnly = Array.isArray(citedSourceIndices) && citedSourceIndices.length > 0;
  const order = citedOnly ? citedSourceIndices : items.map((_, i) => i + 1);
  const note = citedOnly
    ? ""
    : `<p class="sources-note">В тексте ответа нет ссылок вида [1], [2]… — ниже показаны все найденные фрагменты.</p>`;

  const panel = document.createElement("div");
  panel.className = "sources-panel";
  panel.innerHTML = `
    <h3>${icon("book-open-check")} Источники</h3>
    ${note}
    <div class="source-list">
      ${order
        .map((refNum) => {
          const source = items[refNum - 1];
          if (!source) return "";
          return `
        <article class="source-card">
          <div class="source-title">${sourceTitleHtml(source, refNum)}</div>
          <div class="source-meta">
            <span>${icon("file-text")} ${escapeHtml(source.source_file || "unknown file")}</span>
            <span>${icon("braces")} chunk ${escapeHtml(source.chunk_id)}</span>
            <span>${icon("gauge")} ${Number(source.rerank_score ?? source.fused_score ?? 0).toFixed(3)}</span>
          </div>
          <div class="source-snippet">${escapeHtml(source.snippet)}</div>
        </article>
      `;
        })
        .join("")}
    </div>
  `;
  messages.appendChild(panel);
  refreshIcons();
  scrollToBottom();
}

async function loadChatHistory() {
  try {
    const response = await fetch("/api/sessions");
    const sessions = await response.json();
    renderChatList(sessions);
  } catch (error) {
    chatList.innerHTML = `<p class="empty-copy">Не удалось загрузить историю.</p>`;
  }
}

function renderChatList(sessions) {
  if (!sessions.length) {
    chatList.innerHTML = `<p class="empty-copy">Диалогов пока нет.</p>`;
    return;
  }

  chatList.innerHTML = sessions
    .map((s) => {
      const isActive = s.id === activeSessionId || s.id === sessionId;
      const preview = s.title || "New conversation";
      const time = formatTime(s.updated_at);
      return `
        <div
          class="chat-list-item${isActive ? " active" : ""}"
          data-session-id="${s.id}"
          title="${escapeHtml(s.title || "")}"
        >
          <button class="chat-open-btn" data-session-id="${s.id}" type="button">
            <span class="item-icon">${icon("message-circle")}</span>
            <span class="item-text">
              <span class="item-title">${escapeHtml(preview)}</span>
              <span class="item-subtitle">Открыть диалог</span>
            </span>
          </button>
          <span class="item-time">${time}</span>
          <button class="delete-btn" data-delete-id="${s.id}" title="Удалить чат" type="button">
            ${icon("trash-2")}
          </button>
        </div>
      `;
    })
    .join("");

  chatList.querySelectorAll(".chat-open-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const id = btn.dataset.sessionId;
      loadSession(id);
    });
  });

  // Delete handlers
  chatList.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.deleteId;
      await deleteSession(id);
    });
  });

  refreshIcons();
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const d = new Date(timestamp);
  const now = new Date();
  const diffMs = now - d;
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMs / 3600000);

  if (diffMin < 1) return "сейчас";
  if (diffMin < 60) return `${diffMin} мин`;
  if (diffHr < 24) return `${diffHr} ч`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

async function loadSession(nextSessionId) {
  activeSessionId = nextSessionId;
  sessionId = nextSessionId;
  localStorage.setItem("medical-rag-session-id", nextSessionId);

  messages.innerHTML = "";

  try {
    const response = await fetch(`/api/sessions/${nextSessionId}/calls`);
    const calls = await response.json();

    if (!calls.length) {
      showWelcome();
      return;
    }

    for (const call of calls) {
      addMessage("user", call.user_message, false);
      addMessage("assistant", call.answer, true, false, call.stats, call.latency_ms);
      if (call.sources && call.sources.length) {
        renderSources(call.sources, call.cited_source_indices);
      }
    }
    scrollToBottom();
  } catch (error) {
    console.error("Failed to load session:", error);
    addMessage("assistant", `Не удалось открыть диалог: ${error.message}`, false, true);
  }

  await loadChatHistory();
}

async function deleteSession(deletedSessionId) {
  try {
    const response = await fetch(`/api/sessions/${deletedSessionId}`, { method: "DELETE" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || "Delete failed");
    }

    if (deletedSessionId === activeSessionId || deletedSessionId === sessionId) {
      activeSessionId = null;
      sessionId = null;
      localStorage.removeItem("medical-rag-session-id");
      showWelcome();
    }
    await loadChatHistory();
  } catch (error) {
    console.error("Failed to delete session:", error);
  }
}

async function createNewChat() {
  sessionId = null;
  activeSessionId = null;
  localStorage.removeItem("medical-rag-session-id");
  showWelcome();
  await loadChatHistory();
  input.focus();
}

newChatButton.addEventListener("click", createNewChat);

async function loadSystemInfo() {
  try {
    const [health, config] = await Promise.all([
      fetch("/api/health").then((r) => r.json()),
      fetch("/api/config").then((r) => r.json()),
    ]);
    systemInfo.innerHTML = `
      <dt>Статус</dt><dd><span class="status-pill"><span class="status-dot"></span>${health.status === "ok" ? "Подключено" : "Ошибка"}</span></dd>
      <dt>Чанки</dt><dd>${health.chunks_loaded}</dd>
      <dt>LLM</dt><dd>${escapeHtml(config.llm_model)}</dd>
      <dt>Эмбеддинги</dt><dd>${escapeHtml(config.embedding_model)}</dd>
      <dt>Коллекция</dt><dd>${escapeHtml(config.collection)}</dd>
    `;
  } catch (error) {
    systemInfo.innerHTML = `<dt>Статус</dt><dd class="error">${escapeHtml(error.message)}</dd>`;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message, false);
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;
  addThinkingMessage();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "RAG call failed");
    }
    sessionId = data.session_id;
    activeSessionId = data.session_id;
    localStorage.setItem("medical-rag-session-id", sessionId);

    const thinkingMsg = messages.lastElementChild;
    if (thinkingMsg) thinkingMsg.remove();

    addMessage("assistant", data.answer, true, false, data.stats, data.latency_ms);
    if (data.sources && data.sources.length) {
      renderSources(data.sources, data.cited_source_indices);
    }

    await loadChatHistory();
  } catch (error) {
    const thinkingMsg = messages.lastElementChild;
    if (thinkingMsg) thinkingMsg.remove();
    addMessage("assistant", error.message, false, true);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});

indexButton.addEventListener("click", async () => {
  indexButton.disabled = true;
  indexStatus.textContent = "Индексация...";
  try {
    const response = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Indexing failed");
    }
    indexStatus.textContent = data.message;
    await loadSystemInfo();
  } catch (error) {
    indexStatus.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  } finally {
    indexButton.disabled = false;
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

refreshIcons();
loadSystemInfo();
loadChatHistory();
