(function () {
    "use strict";

    var API_URL = "/api/chat/stream";
    var MAX_HISTORY = 20;

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

    function sanitizeUrl(url) {
        url = url.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
        if (!/^https?:\/\//i.test(url) && !/^mailto:/i.test(url)) return "#";
        return url;
    }

    function renderMarkdownLight(text) {
        var html = escapeHtml(text);
        // Headings — must run before inline transforms and newline conversion
        html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
        html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
        html = html.replace(/\*\*\[([^\]]+)\]\(([^)]+)\)\*\*/g, function (_, label, url) {
            return '<strong><a href="' + sanitizeUrl(url) + '" target="_blank" rel="noopener">' + label + '</a></strong>';
        });
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, label, url) {
            return '<a href="' + sanitizeUrl(url) + '" target="_blank" rel="noopener">' + label + '</a>';
        });
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // ── Welcome Card ────────────────────────────────────

    function showWelcome() {
        var card = document.createElement("div");
        card.className = "welcome-card";
        card.id = "welcome-card";
        card.innerHTML =
            '<h2>Hi Hokies!</h2>' +
            '<p>I am your AI assistant for Virginia Tech. Ask me about programs, courses, faculty, admissions, and more. I answer using available university sources.</p>';
        messagesEl.appendChild(card);

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
        if (role === "assistant") {
            var img = document.createElement("img");
            img.src = "assets/images/vt-mark-orange.svg";
            img.alt = "VT";
            avatar.appendChild(img);
        } else {
            avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>';
        }

        var bubble = document.createElement("div");
        bubble.className = "message " + role;

        row.appendChild(avatar);
        row.appendChild(bubble);
        return { row: row, bubble: bubble };
    }

    function addMessage(role, content, sources) {
        var welcome = document.getElementById("welcome-card");
        if (welcome && role === "user") {
            welcome.style.animation = "none";
            welcome.style.opacity = "0";
            welcome.style.transform = "translateY(-8px)";
            welcome.style.transition = "all 0.25s ease";
            setTimeout(function () { welcome.remove(); }, 250);
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

    function linkCitations(bubble, sources) {
        var byId = {};
        sources.forEach(function (s) { byId[s.id] = s; });
        var walker = document.createTreeWalker(bubble, NodeFilter.SHOW_TEXT);
        var nodes = [];
        while (walker.nextNode()) {
            var node = walker.currentNode;
            if (!node.parentElement.closest('a, code, pre, .sources')) nodes.push(node);
        }
        nodes.forEach(function (node) {
            var text = node.textContent;
            var pattern = /\[(\d+)\]/g;
            var match, last = 0;
            var fragment = document.createDocumentFragment();
            while ((match = pattern.exec(text))) {
                var source = byId[Number(match[1])];
                if (!source || !/^https?:\/\//i.test(source.url)) continue;
                fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
                var link = document.createElement('a');
                link.className = 'citation';
                link.href = source.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.title = source.title || source.url;
                link.textContent = match[0];
                fragment.appendChild(link);
                last = pattern.lastIndex;
            }
            if (last) {
                fragment.appendChild(document.createTextNode(text.slice(last)));
                node.replaceWith(fragment);
            }
        });
    }

    function appendSources(bubble, sources) {
        var cited = new Set((bubble.textContent.match(/\[\d+\]/g) || []).map(function (value) { return Number(value.slice(1, -1)); }));
        sources = sources.filter(function (s) { return cited.has(s.id) && /^https?:\/\//i.test(s.url || ''); });
        if (!sources.length) return;
        linkCitations(bubble, sources);
        var srcDiv = document.createElement("div");
        srcDiv.className = "sources";

        var toggle = document.createElement("button");
        toggle.className = "sources-toggle";
        toggle.type = "button";
        toggle.setAttribute("aria-expanded", "false");
        toggle.innerHTML = '<span class="sources-arrow"></span> Sources (' + sources.length + ')';
        srcDiv.appendChild(toggle);

        var list = document.createElement("div");
        list.className = "sources-list";

        for (var i = 0; i < sources.length; i++) {
            var s = sources[i];
            var a = document.createElement("a");
            a.className = "source-link";
            a.href = s.url;
            a.target = "_blank";
            a.rel = "noopener";
            a.textContent = '[' + s.id + '] ' + (s.title || s.url);
            list.appendChild(a);
        }
        srcDiv.appendChild(list);

        toggle.addEventListener("click", function () {
            var expanded = toggle.classList.toggle("open");
            toggle.setAttribute("aria-expanded", String(expanded));
            list.classList.toggle("visible");
        });

        bubble.appendChild(srcDiv);
    }

    // ── Loading State ───────────────────────────────────

    function setLoading(loading) {
        sendBtn.disabled = loading;
        inputEl.disabled = loading;
        newChatBtn.disabled = loading;
        typingEl.classList.toggle("hidden", !loading);
        if (loading) scrollToBottom();
    }

    // ── Send Message (SSE Streaming) ────────────────────

    async function sendMessage(question) {
        addMessage("user", question);
        var sendHistory = history.slice(-MAX_HISTORY);
        setLoading(true);

        var els = createMessageRow("assistant");
        var assistantBubble = els.bubble;
        var assistantRowAdded = false;

        var fullAnswer = "";
        var receivedDone = false;
        var receivedFinal = false;
        var reader;

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
                var detail = errData.detail || "";
                if (resp.status === 429) {
                    throw new Error("You've reached the limit of 100 messages per hour. Please try again later.");
                } else if (resp.status === 400 && detail) {
                    throw new Error(detail);
                } else {
                    throw new Error(detail || "Server error " + resp.status);
                }
            }

            reader = resp.body.getReader();
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
                        if (!assistantRowAdded) {
                            assistantRowAdded = true;
                            typingEl.classList.add("hidden");
                            messagesEl.appendChild(els.row);
                        }
                        fullAnswer += event.content;
                        assistantBubble.innerHTML = renderMarkdownLight(fullAnswer); // existing pattern — content is escaped by renderMarkdownLight
                        scrollToBottom();
                    } else if (event.type === "answer") {
                        fullAnswer = event.content || '';
                        receivedFinal = true;
                        if (!assistantRowAdded) {
                            assistantRowAdded = true;
                            messagesEl.appendChild(els.row);
                        }
                        assistantBubble.innerHTML = renderMarkdownLight(fullAnswer);
                    } else if (event.type === "sources") {
                        if (event.sources && event.sources.length > 0 && receivedFinal) {
                            appendSources(assistantBubble, event.sources);
                        }
                    } else if (event.type === "error") {
                        throw new Error(event.content || 'The response was interrupted.');
                    } else if (event.type === "done") {
                        receivedDone = true;
                    }
                }
            }

            if (!receivedDone || !receivedFinal || !fullAnswer.trim()) {
                throw new Error('The response was incomplete. Please try again.');
            }
            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: fullAnswer });
            // Match the backend's aggregate input bound while retaining recent turns.
            while (history.length > MAX_HISTORY || history.reduce(function (n, m) { return n + m.content.length; }, 0) > 60000) {
                history.splice(0, 2);
            }

        } catch (err) {
            if (!fullAnswer && assistantRowAdded) {
                els.row.remove();
            }
            addMessage("error", "Error: " + err.message);
        } finally {
            if (reader) {
                try { await reader.cancel(); } catch (_) { /* already closed */ }
                reader.releaseLock();
            }
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
