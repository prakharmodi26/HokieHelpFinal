(function () {
    "use strict";

    var API_URL = "/api/chat";
    var MAX_HISTORY = 50;

    var messagesEl = document.getElementById("messages");
    var formEl = document.getElementById("chat-form");
    var inputEl = document.getElementById("question-input");
    var sendBtn = document.getElementById("send-btn");
    var typingEl = document.getElementById("typing-indicator");

    var history = [];

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function renderMarkdownLight(text) {
        var html = escapeHtml(text);
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    function addMessage(role, content, sources) {
        var div = document.createElement("div");
        div.classList.add("message", role);

        if (role === "assistant") {
            div.innerHTML = renderMarkdownLight(content);
            if (sources && sources.length > 0) {
                var srcDiv = document.createElement("div");
                srcDiv.classList.add("sources");
                srcDiv.innerHTML = "<strong>Sources:</strong><br>" +
                    sources.map(function (s) {
                        return '<a href="' + escapeHtml(s.url) + '" target="_blank" rel="noopener">' +
                            escapeHtml(s.title || s.url) + "</a>";
                    }).join("<br>");
                div.appendChild(srcDiv);
            }
        } else if (role === "error") {
            div.textContent = content;
        } else {
            div.textContent = content;
        }

        messagesEl.appendChild(div);
        messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
    }

    function setLoading(loading) {
        sendBtn.disabled = loading;
        inputEl.disabled = loading;
        typingEl.classList.toggle("hidden", !loading);
        if (loading) {
            messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
        }
    }

    async function sendMessage(question) {
        addMessage("user", question);

        var sendHistory = history.slice(-MAX_HISTORY);

        setLoading(true);

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

            var data = await resp.json();

            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: data.answer });

            addMessage("assistant", data.answer, data.sources);
        } catch (err) {
            addMessage("error", "Error: " + err.message);
        } finally {
            setLoading(false);
            inputEl.focus();
        }
    }

    formEl.addEventListener("submit", function (e) {
        e.preventDefault();
        var q = inputEl.value.trim();
        if (!q) return;
        inputEl.value = "";
        sendMessage(q);
    });

    addMessage("assistant", "Welcome to HokieHelp! Ask me anything about Virginia Tech's Computer Science department.");
})();
