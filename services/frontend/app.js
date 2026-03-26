(function () {
    "use strict";

    var API_URL = "/api/chat/stream";
    var MAX_HISTORY = 50;

    var messagesEl = document.getElementById("messages");
    var chatContainer = document.getElementById("chat-container");
    var formEl = document.getElementById("chat-form");
    var inputEl = document.getElementById("question-input");
    var sendBtn = document.getElementById("send-btn");
    var typingEl = document.getElementById("typing-indicator");
    var newChatBtn = document.getElementById("new-chat-btn");

    var history = [];

    // ── Helpers ──────────────────────────────────────────

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function renderMarkdownLight(text) {
        var html = escapeHtml(text);
        // Bold links: **[text](url)**
        html = html.replace(/\*\*\[([^\]]+)\]\(([^)]+)\)\*\*/g,
            '<strong><a href="$2" target="_blank" rel="noopener">$1</a></strong>');
        // Links: [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener">$1</a>');
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        // Italic
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        // Inline code
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        // Line breaks
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // ── SVG Icons ───────────────────────────────────────

    var assistantAvatarSVG = '<svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">' +
        '<path d="M14 18L20 24L26 14" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>' +
        '</svg>';

    var userAvatarSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>' +
        '<circle cx="12" cy="7" r="4"/>' +
        '</svg>';

    // ── Welcome Card ────────────────────────────────────

    function showWelcome() {
        var card = document.createElement("div");
        card.className = "welcome-card";
        card.id = "welcome-card";
        card.innerHTML =
            '<h2>Welcome to HokieHelp</h2>' +
            '<p>Your AI assistant for Virginia Tech\'s Computer Science department. Ask me about faculty, courses, research, admissions, and more.</p>' +
            '<div class="welcome-suggestions">' +
                '<button class="suggestion-chip" data-q="Who are the faculty in the CS department?">Faculty</button>' +
                '<button class="suggestion-chip" data-q="What research areas does the CS department focus on?">Research areas</button>' +
                '<button class="suggestion-chip" data-q="Tell me about the CS undergraduate program">Undergrad program</button>' +
                '<button class="suggestion-chip" data-q="What are the admission requirements for CS?">Admissions</button>' +
            '</div>';
        messagesEl.appendChild(card);

        // Suggestion chip click handlers
        var chips = card.querySelectorAll(".suggestion-chip");
        for (var i = 0; i < chips.length; i++) {
            chips[i].addEventListener("click", function () {
                var q = this.getAttribute("data-q");
                if (q) {
                    inputEl.value = q;
                    formEl.dispatchEvent(new Event("submit"));
                }
            });
        }
    }

    // ── Message Creation ────────────────────────────────

    function createMessageRow(role) {
        var row = document.createElement("div");
        row.className = "msg-row " + role;

        var avatar = document.createElement("div");
        avatar.className = "msg-avatar " + (role === "assistant" ? "assistant-avatar" : "user-avatar");
        avatar.innerHTML = role === "assistant" ? assistantAvatarSVG : userAvatarSVG;

        var bubble = document.createElement("div");
        bubble.className = "message " + role;

        row.appendChild(avatar);
        row.appendChild(bubble);
        return { row: row, bubble: bubble };
    }

    function addMessage(role, content, sources) {
        // Remove welcome card on first real interaction
        var welcome = document.getElementById("welcome-card");
        if (welcome && role === "user") {
            welcome.style.animation = "none";
            welcome.style.opacity = "0";
            welcome.style.transform = "translateY(-8px)";
            welcome.style.transition = "all 0.3s ease";
            setTimeout(function () { welcome.remove(); }, 300);
        }

        if (role === "error") {
            var errWrap = document.createElement("div");
            errWrap.className = "msg-error";
            var errDiv = document.createElement("div");
            errDiv.className = "message error";
            errDiv.textContent = content;
            errWrap.appendChild(errDiv);
            messagesEl.appendChild(errWrap);
            scrollToBottom();
            return errDiv;
        }

        var els = createMessageRow(role);
        if (role === "assistant") {
            els.bubble.innerHTML = renderMarkdownLight(content);
            if (sources && sources.length > 0) {
                appendSources(els.bubble, sources);
            }
        } else {
            els.bubble.textContent = content;
        }

        messagesEl.appendChild(els.row);
        scrollToBottom();
        return els.bubble;
    }

    function appendSources(bubble, sources) {
        var srcDiv = document.createElement("div");
        srcDiv.className = "sources";
        var label = document.createElement("div");
        label.className = "sources-label";
        label.textContent = "Sources";
        srcDiv.appendChild(label);

        for (var i = 0; i < sources.length; i++) {
            var s = sources[i];
            var a = document.createElement("a");
            a.className = "source-link";
            a.href = s.url;
            a.target = "_blank";
            a.rel = "noopener";
            a.textContent = s.title || s.url;
            srcDiv.appendChild(a);
        }
        bubble.appendChild(srcDiv);
    }

    // ── Loading State ───────────────────────────────────

    function setLoading(loading) {
        sendBtn.disabled = loading;
        inputEl.disabled = loading;
        typingEl.classList.toggle("hidden", !loading);
        if (loading) scrollToBottom();
    }

    // ── Send Message (SSE Streaming) ────────────────────

    async function sendMessage(question) {
        addMessage("user", question);
        var sendHistory = history.slice(-MAX_HISTORY);
        setLoading(true);

        // Create assistant message row for streaming
        var els = createMessageRow("assistant");
        messagesEl.appendChild(els.row);
        var assistantBubble = els.bubble;

        var fullAnswer = "";

        try {
            var resp = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    history: sendHistory,
                }),
            });

            if (!resp.ok) {
                var errData = await resp.json().catch(function () { return {}; });
                throw new Error(errData.detail || "Server error " + resp.status);
            }

            // Hide typing indicator once stream begins
            typingEl.classList.add("hidden");

            var reader = resp.body.getReader();
            var decoder = new TextDecoder();
            var buffer = "";

            while (true) {
                var result = await reader.read();
                if (result.done) break;

                buffer += decoder.decode(result.value, { stream: true });
                var lines = buffer.split("\n");
                buffer = lines.pop();

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i].trim();
                    if (!line.startsWith("data: ")) continue;
                    var data = line.slice(6);

                    try {
                        var event = JSON.parse(data);
                    } catch (e) {
                        continue;
                    }

                    if (event.type === "token") {
                        fullAnswer += event.content;
                        assistantBubble.innerHTML = renderMarkdownLight(fullAnswer);
                        scrollToBottom();
                    } else if (event.type === "sources") {
                        if (event.sources && event.sources.length > 0) {
                            appendSources(assistantBubble, event.sources);
                        }
                    }
                }
            }

            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: fullAnswer });

        } catch (err) {
            if (!fullAnswer) {
                els.row.remove();
            }
            addMessage("error", "Error: " + err.message);
        } finally {
            setLoading(false);
            inputEl.focus();
        }
    }

    // ── New Chat ────────────────────────────────────────

    function resetChat() {
        history = [];
        messagesEl.innerHTML = "";
        showWelcome();
        inputEl.focus();
    }

    // ── Event Listeners ─────────────────────────────────

    formEl.addEventListener("submit", function (e) {
        e.preventDefault();
        var q = inputEl.value.trim();
        if (!q) return;
        inputEl.value = "";
        sendMessage(q);
    });

    newChatBtn.addEventListener("click", resetChat);

    // Auto-resize input height (future multi-line support)
    inputEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            formEl.dispatchEvent(new Event("submit"));
        }
    });

    // ── Init ────────────────────────────────────────────
    showWelcome();
    inputEl.focus();
})();
