
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const sendButton = document.getElementById("send-button");

// FastAPI endpoint
const API_URL = "https://challenge-portfolio-gloves-permissions.trycloudflare.com/chat";

// Conversation history
let history = [];

function addMessage(role, content) {
    const message = document.createElement("div");

    message.classList.add("message");

    if (role === "user") {
        message.classList.add("user-message");
    } else {
        message.classList.add("bot-message");
    }

    const label = document.createElement("div");
    label.classList.add("message-label");
    label.textContent = role === "user" ? "You" : "AI Assistant";

    const text = document.createElement("div");
    text.textContent = content;

    message.appendChild(label);
    message.appendChild(text);

    messages.appendChild(message);

    message.scrollIntoView({
        behavior: "smooth",
        block: "end"
    });

    return message;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) return;

    welcome.style.display = "none";

    addMessage("user", question);

    questionInput.value = "";
    sendButton.disabled = true;

    const loadingMessage = addMessage(
        "assistant",
        "Thinking... Retrieving relevant lecture material."
    );

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                history: history
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        loadingMessage.remove();

        addMessage("assistant", data.answer);

        history.push(
            { role: "user", content: question },
            { role: "assistant", content: data.answer }
        );

    } catch (error) {
    loadingMessage.remove();

    addMessage("assistant", `Error: ${error.message}`);

    console.error("Backend error:", error);
}
 finally {
        sendButton.disabled = false;
        questionInput.focus();
    }
});
