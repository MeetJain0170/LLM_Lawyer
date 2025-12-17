/******************************************************************************
 * Legal LLM Web Chat – Frontend Controller
 * ----------------------------------------------------------------------------
 * This frontend communicates with a local backend API (api_server.py),
 * which internally wraps the Legal LLM SFT Inference CLI.
 *
 * Backend Model Reference:
 * - Model: legal_llm_sft_final.pt (Supervised Fine-Tuned Legal GPT)
 * - Tokenizer: legal_tokenizer.json
 * - Prompt Discipline: Indian Legal Consultant, section-cited answers only
 *
 * Inference Core (Server-side):
 * - Uses PyTorch GPT model
 * - Autoregressive decoding with temperature + top-k
 * - Hard stop tokens (<EOS>, instruction, response)
 *
 * This JS file is UI-only. All legal reasoning happens in Python.
 ******************************************************************************/

// -----------------------------------------------------------------------------
// API Configuration
// -----------------------------------------------------------------------------
const API_BASE_URL = 'http://localhost:5000';

// -----------------------------------------------------------------------------
// DOM Elements
// -----------------------------------------------------------------------------
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatHistory = document.getElementById('chatHistory');

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
let isGenerating = false;
let currentChatId = Date.now();

// -----------------------------------------------------------------------------
// Initialize
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    setupEventListeners();
    loadChatHistory();
});

// -----------------------------------------------------------------------------
// Helper: Format Text (Markdown → HTML, Legal Citations Highlighting)
// -----------------------------------------------------------------------------
function formatText(text) {
    if (!text) return '';

    // 1. Escape HTML (basic XSS safety)
    let formatted = text.replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;");

    // 2. Bold: **text**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 3. Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    // 4. Highlight legal citations
    formatted = formatted.replace(
        /(Section\s+\d+|Article\s+\d+|IPC\s+\d+|CrPC\s+\d+)/gi,
        '<span class="legal-citation">$1</span>'
    );

    return formatted;
}

// -----------------------------------------------------------------------------
// Event Listeners
// -----------------------------------------------------------------------------
function setupEventListeners() {
    sendBtn.addEventListener('click', handleSend);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
        sendBtn.disabled = !messageInput.value.trim() || isGenerating;
    });

    newChatBtn.addEventListener('click', startNewChat);

    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            messageInput.value = question;
            messageInput.style.height = 'auto';
            messageInput.style.height = messageInput.scrollHeight + 'px';
            sendBtn.disabled = false;
            handleSend();
        });
    });
}

// -----------------------------------------------------------------------------
// API Health Check
// -----------------------------------------------------------------------------
async function checkApiHealth() {
    try {
        await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'ping' })
        });
    } catch (error) {
        console.warn("⚠️ Backend not reachable yet. Start api_server.py.");
    }
}

// -----------------------------------------------------------------------------
// Send Message → Backend (Legal LLM SFT)
// -----------------------------------------------------------------------------
async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isGenerating) return;

    if (welcomeScreen) welcomeScreen.style.display = 'none';

    addMessage('user', message);

    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    isGenerating = true;

    const loadingId = addMessage('assistant', '', true);

    try {
        /**
         * Backend Call:
         * - POST /chat
         * - Backend injects the Legal SFT prompt template
         * - Calls generate() from Legal LLM CLI logic
         */
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        updateMessage(loadingId, data.response);
        saveToChatHistory(message, data.response);

    } catch (error) {
        updateMessage(
            loadingId,
            "❌ Could not connect to Legal AI. Ensure api_server.py is running.",
            true
        );
        console.error(error);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// -----------------------------------------------------------------------------
// Message Rendering
// -----------------------------------------------------------------------------
function addMessage(role, text, isLoading = false) {
    const messageId = `msg-${Date.now()}-${Math.random()}`;
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message ${role} ${isLoading ? 'loading' : ''}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚖️';

    const content = document.createElement('div');
    content.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';

    textDiv.innerHTML = isLoading
        ? '<div class="typing-indicator"><div></div><div></div><div></div></div>'
        : formatText(text);

    content.appendChild(textDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
    return messageId;
}

function updateMessage(messageId, text, isError = false) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;

    messageDiv.classList.remove('loading');
    const textDiv = messageDiv.querySelector('.message-text');

    textDiv.innerHTML = isError
        ? `<div class="error-message">${text}</div>`
        : formatText(text);

    scrollToBottom();
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// -----------------------------------------------------------------------------
// Chat History (LocalStorage)
// -----------------------------------------------------------------------------
function saveToChatHistory(userMessage, assistantMessage) {
    const history = getChatHistory();
    const chat = history.find(c => c.id === currentChatId) || {
        id: currentChatId,
        title: userMessage.slice(0, 40) + "...",
        messages: []
    };

    chat.messages.push({ user: userMessage, assistant: assistantMessage });

    if (chat.messages.length === 1) {
        chat.title = userMessage.slice(0, 40) + "...";
    }

    const index = history.findIndex(c => c.id === currentChatId);
    index >= 0 ? history[index] = chat : history.unshift(chat);

    if (history.length > 20) history.pop();

    localStorage.setItem('chatHistory', JSON.stringify(history));
    loadChatHistory();
}

function getChatHistory() {
    return JSON.parse(localStorage.getItem('chatHistory') || '[]');
}

function loadChatHistory() {
    if (!chatHistory) return;
    chatHistory.innerHTML = '';

    getChatHistory().forEach(chat => {
        const item = document.createElement('div');
        item.className = 'chat-history-item';
        item.textContent = chat.title;
        item.onclick = () => loadChat(chat);
        chatHistory.appendChild(item);
    });
}

function loadChat(chat) {
    chatContainer.innerHTML = '';
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    currentChatId = chat.id;

    chat.messages.forEach(m => {
        addMessage('user', m.user);
        addMessage('assistant', m.assistant);
    });
}

function startNewChat() {
    currentChatId = Date.now();
    chatContainer.innerHTML = '';
    if (welcomeScreen) welcomeScreen.style.display = 'flex';
    messageInput.value = '';
    sendBtn.disabled = true;
}
