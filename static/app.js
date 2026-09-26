(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const docList = $("#doc-list");
  const docEmpty = $("#doc-empty");
  const uploadQueue = $("#upload-queue");
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const statDocuments = $("#stat-documents");
  const statChunks = $("#stat-chunks");
  const statusQdrant = $("#status-qdrant");
  const statusGemini = $("#status-gemini");

  const chatScroll = $("#chat-scroll");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const sendBtn = $("#send-btn");
  const clearChatBtn = $("#clear-chat");

  let conversationHistory = []; // [{role: 'user'|'model', text}]
  let lastQuestion = null;

  // ---------------------------------------------------------------
  // Tabs (mobile)
  // ---------------------------------------------------------------
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = btn.dataset.panel;
      $$(".panel").forEach((p) => p.classList.remove("active"));
      $(`#panel-${target}`).classList.add("active");
    });
  });
  // default active panel for mobile
  $("#panel-documents").classList.add("active");

  // ---------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 KB";
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(0)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

  function formatDate(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
        " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    } catch { return ""; }
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok || !body || body.success === false) {
      const message = body && body.error ? body.error.message : `Request failed (${res.status})`;
      throw new Error(message);
    }
    return body.data;
  }

  // ---------------------------------------------------------------
  // Health / status
  // ---------------------------------------------------------------
  async function refreshHealth() {
    try {
      const data = await api("/api/health");
      setChip(statusQdrant, !!data.qdrant.connected, "Vector store");
      setChip(statusGemini, !!data.gemini.connected, "Gemini");
      statDocuments.textContent = data.knowledge_base.document_count;
      statChunks.textContent = data.knowledge_base.chunk_count;
    } catch (e) {
      setChip(statusQdrant, false, "Vector store");
      setChip(statusGemini, false, "Gemini");
    }
  }

  function setChip(el, isOnline, label) {
    el.classList.toggle("online", isOnline);
    el.classList.toggle("offline", !isOnline);
    el.innerHTML = `<i class="dot"></i> ${label}`;
  }

  // ---------------------------------------------------------------
  // Documents
  // ---------------------------------------------------------------
  async function refreshDocuments() {
    try {
      const data = await api("/api/documents");
      renderDocuments(data.documents || []);
    } catch (e) {
      // silent - health chip already communicates backend trouble
    }
  }

  function renderDocuments(docs) {
    docList.querySelectorAll(".doc-item").forEach((n) => n.remove());
    docEmpty.style.display = docs.length ? "none" : "block";

    docs.forEach((doc) => {
      const item = document.createElement("div");
      item.className = "doc-item";
      const ext = (doc.file_type || "").toUpperCase();
      const statusClass = doc.status === "completed" ? "completed"
        : doc.status === "failed" ? "failed" : "processing";
      const statusLabel = doc.status === "completed" ? "Indexed"
        : doc.status === "failed" ? "Failed"
        : doc.status.charAt(0).toUpperCase() + doc.status.slice(1);

      item.innerHTML = `
        <div class="doc-icon">${escapeHtml(ext.slice(0, 4))}</div>
        <div class="doc-info">
          <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
          <div class="doc-meta">
            <span>${formatBytes(doc.size_bytes)}</span>
            <span>${formatDate(doc.uploaded_at)}</span>
            ${doc.chunk_count ? `<span>${doc.chunk_count} chunks</span>` : ""}
            <span class="doc-status ${statusClass}">${escapeHtml(statusLabel)}</span>
          </div>
          ${doc.error ? `<div class="doc-error">${escapeHtml(doc.error)}</div>` : ""}
        </div>
        <button class="doc-delete" title="Delete document" data-id="${doc.id}">
          <svg viewBox="0 0 24 24" width="16" height="16">
            <path d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-8 0v13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V7"
              fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      `;
      item.querySelector(".doc-delete").addEventListener("click", () => deleteDocument(doc.id, doc.filename));
      docList.appendChild(item);
    });
  }

  async function deleteDocument(id, filename) {
    if (!confirm(`Delete "${filename}"? This removes it and its indexed chunks.`)) return;
    try {
      await api(`/api/documents/${id}`, { method: "DELETE" });
      await Promise.all([refreshDocuments(), refreshHealth()]);
    } catch (e) {
      alert(`Could not delete document: ${e.message}`);
    }
  }

  // ---------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------
  dropzone.addEventListener("click", (e) => {
    if (e.target === fileInput) return;
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  async function uploadFile(file) {
    const row = document.createElement("div");
    row.className = "upload-item";
    row.innerHTML = `
      <span class="name">${escapeHtml(file.name)}</span>
      <span class="stage">Uploading</span>
      <div class="upload-progress-track"><div class="upload-progress-fill"></div></div>
    `;
    uploadQueue.appendChild(row);
    const stageEl = row.querySelector(".stage");

    const formData = new FormData();
    formData.append("file", file);

    try {
      stageEl.textContent = "Processing";
      const data = await api("/api/documents", { method: "POST", body: formData });
      const doc = data.document;
      if (doc.status === "completed") {
        row.classList.add("completed");
        stageEl.textContent = `Indexed \u00b7 ${doc.chunk_count} chunks`;
      } else {
        row.classList.add("failed");
        stageEl.textContent = doc.error || "Failed";
      }
    } catch (e) {
      row.classList.add("failed");
      stageEl.textContent = e.message;
    } finally {
      row.querySelector(".upload-progress-track").remove();
      setTimeout(() => row.remove(), 6000);
      await Promise.all([refreshDocuments(), refreshHealth()]);
    }
  }

  // ---------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendQuestion(question);
  });

  clearChatBtn.addEventListener("click", () => {
    conversationHistory = [];
    lastQuestion = null;
    chatScroll.innerHTML = `<div class="chat-intro" id="chat-intro"><p>Answers are grounded in your uploaded documents. Ask a question below and I'll cite exactly where the answer came from.</p></div>`;
  });

  function removeIntro() {
    const intro = $("#chat-intro");
    if (intro) intro.remove();
  }

  function appendUserMessage(text) {
    removeIntro();
    const msg = document.createElement("div");
    msg.className = "msg user";
    msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatScroll.appendChild(msg);
    scrollToBottom();
  }

  function appendTyping() {
    const msg = document.createElement("div");
    msg.className = "msg assistant";
    msg.id = "typing-indicator";
    msg.innerHTML = `<div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
    chatScroll.appendChild(msg);
    scrollToBottom();
  }

  function removeTyping() {
    const el = $("#typing-indicator");
    if (el) el.remove();
  }

  function appendAssistantMessage(answer, sources, isError, question) {
    const msg = document.createElement("div");
    msg.className = "msg assistant";

    const sourcesHtml = (sources && sources.length)
      ? `<div class="sources">${sources.map((s) => `
          <div class="source-card">
            <div class="source-top">
              <span class="source-file">${escapeHtml(s.filename)}${s.page_number ? " &middot; p." + s.page_number : ""}</span>
              <span class="source-relevance">${s.relevance}%</span>
            </div>
            <div class="source-snippet">${escapeHtml(s.snippet)}</div>
          </div>`).join("")}</div>`
      : "";

    msg.innerHTML = `
      <div class="bubble${isError ? " error" : ""}">${escapeHtml(answer)}</div>
      ${sourcesHtml}
      <div class="msg-actions">
        <button class="ghost-btn copy-btn" type="button">Copy</button>
        ${question ? `<button class="ghost-btn regen-btn" type="button">Regenerate</button>` : ""}
      </div>
    `;
    chatScroll.appendChild(msg);

    msg.querySelector(".copy-btn").addEventListener("click", () => {
      navigator.clipboard.writeText(answer).catch(() => {});
    });
    const regenBtn = msg.querySelector(".regen-btn");
    if (regenBtn) {
      regenBtn.addEventListener("click", () => {
        msg.remove();
        sendQuestion(question, true);
      });
    }
    scrollToBottom();
  }

  function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  async function sendQuestion(question, isRegenerate = false) {
    if (!isRegenerate) {
      appendUserMessage(question);
    }
    lastQuestion = question;
    sendBtn.disabled = true;
    appendTyping();

    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: conversationHistory }),
      });
      removeTyping();
      appendAssistantMessage(data.answer, data.sources, false, question);
      conversationHistory.push({ role: "user", text: question });
      conversationHistory.push({ role: "model", text: data.answer });
      conversationHistory = conversationHistory.slice(-10);
    } catch (e) {
      removeTyping();
      appendAssistantMessage(e.message, [], true, question);
    } finally {
      sendBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------
  refreshHealth();
  refreshDocuments();
  setInterval(refreshHealth, 20000);
})();
