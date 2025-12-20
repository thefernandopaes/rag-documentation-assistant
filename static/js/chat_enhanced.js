/**
 * Enhanced Chat Interface with Professional Features
 *
 * Features:
 * - Markdown rendering with DOMPurify sanitization
 * - Code examples with tabs and copy buttons
 * - Loading states with skeleton
 * - API metadata visualization
 * - Responsive design
 * - Keyboard shortcuts
 * - Toast notifications
 * - Error handling
 */

// Global state
let lastQuery = '';
let currentSessionId = null;

/**
 * Initialize chat interface
 */
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupKeyboardShortcuts();

    // Configure marked.js
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: true,
            mangle: false
        });
    }

    console.log('✓ Chat interface initialized');
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const clearBtn = document.getElementById('clear-chat');

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    if (userInput) {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', clearChat);
    }
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+L to clear chat
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            if (confirm('Clear chat history?')) {
                clearChat();
            }
        }

        // Ctrl+/ to focus input
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            const input = document.getElementById('user-input');
            if (input) input.focus();
        }
    });
}

/**
 * Send message to API
 */
async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const query = userInput.value.trim();

    if (!query) {
        showToast('Please enter a message', 'warning');
        return;
    }

    lastQuery = query;

    // Add user message
    addMessage(query, 'user');
    userInput.value = '';

    // Show loading state
    showLoadingState();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        hideLoadingState();

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.details || errorData.error || `HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResponse(data);

    } catch (error) {
        hideLoadingState();
        console.error('Error:', error);
        displayError(error.message || 'Failed to get response. Please try again.');
    }
}

/**
 * Add user message to chat
 */
function addMessage(content, type) {
    const messagesDiv = document.getElementById('chat-messages');
    const isUser = type === 'user';

    const messageHtml = `
        <div class="message ${isUser ? 'user-message' : 'assistant-message'} mb-3">
            <div class="d-flex align-items-start ${isUser ? 'justify-content-end' : ''}">
                ${!isUser ? `
                <div class="avatar bg-primary rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-robot text-white"></i>
                </div>
                ` : ''}
                <div class="message-content ${isUser ? 'text-end' : 'flex-grow-1'}">
                    <div class="message-bubble ${isUser ? 'bg-primary' : 'bg-dark border'} p-3 rounded">
                        ${escapeHtml(content)}
                    </div>
                </div>
                ${isUser ? `
                <div class="avatar bg-secondary rounded-circle d-flex align-items-center justify-content-center ms-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-user text-white"></i>
                </div>
                ` : ''}
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', messageHtml);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Show loading state
 */
function showLoadingState() {
    const messagesDiv = document.getElementById('chat-messages');

    const loadingHtml = `
        <div class="message assistant-message mb-3" id="loading-state">
            <div class="d-flex align-items-start">
                <div class="avatar bg-primary rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-robot text-white"></i>
                </div>
                <div class="message-content flex-grow-1">
                    <div class="message-bubble bg-dark border p-3 rounded">
                        <div class="loading-skeleton">
                            <div class="skeleton-line"></div>
                            <div class="skeleton-line"></div>
                            <div class="skeleton-line short"></div>
                        </div>
                        <div class="loading-text mt-2">
                            <i class="fas fa-circle-notch fa-spin me-2"></i>
                            <span id="loading-stage">Searching documentation...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', loadingHtml);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Simulate progress stages
    setTimeout(() => updateLoadingStage('Generating response...'), 2000);
    setTimeout(() => updateLoadingStage('Formatting result...'), 4000);
}

/**
 * Update loading stage text
 */
function updateLoadingStage(text) {
    const stage = document.getElementById('loading-stage');
    if (stage) stage.textContent = text;
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const loadingEl = document.getElementById('loading-state');
    if (loadingEl) loadingEl.remove();
}

/**
 * Display complete response
 */
function displayResponse(data) {
    const messagesDiv = document.getElementById('chat-messages');

    // Render main response with markdown
    const responseHtml = renderMarkdown(data.response || 'No response available');

    // Render code examples with tabs
    const examplesHtml = renderCodeExamples(data.examples || data.code_examples || []);

    // Render API metadata
    const metadataHtml = renderApiMetadata(data);

    // Render sources
    const sourcesHtml = renderSources(data.sources || []);

    // Render related questions
    const relatedHtml = renderRelatedQuestions(data.related_questions || data.related_concepts || []);

    // Performance info (if debug mode)
    const perfHtml = renderPerformanceInfo(data);

    const messageHtml = `
        <div class="message assistant-message mb-3">
            <div class="d-flex align-items-start">
                <div class="avatar bg-primary rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-robot text-white"></i>
                </div>
                <div class="message-content flex-grow-1">
                    <div class="message-bubble bg-dark border p-3 rounded">
                        <div class="response-text">${responseHtml}</div>
                        ${examplesHtml}
                        ${metadataHtml}
                        ${sourcesHtml}
                        ${relatedHtml}
                        ${perfHtml}
                    </div>
                </div>
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', messageHtml);

    // Apply syntax highlighting
    messagesDiv.querySelectorAll('pre code').forEach((block) => {
        if (typeof Prism !== 'undefined') {
            Prism.highlightElement(block);
        }
    });

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Render markdown with DOMPurify sanitization
 */
function renderMarkdown(content) {
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
        return escapeHtml(content);
    }

    const rawHtml = marked.parse(content);
    return DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'a', 'blockquote', 'hr'],
        ALLOWED_ATTR: ['href', 'class', 'id', 'target']
    });
}

/**
 * Render code examples with tabs
 */
function renderCodeExamples(examples) {
    if (!examples || examples.length === 0) return '';

    const tabsId = `tabs-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    let html = `
        <div class="code-examples-section mt-3">
            <h6 class="mb-2"><i class="fas fa-code me-2"></i>Code Examples</h6>
            <div class="code-tabs">
                <ul class="nav nav-tabs mb-2" role="tablist">
    `;

    // Tab headers
    examples.forEach((example, index) => {
        const lang = (example.language || example.lang || 'text').toUpperCase();
        const isActive = index === 0 ? 'active' : '';
        html += `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${isActive}" data-bs-toggle="tab" data-bs-target="#${tabsId}-${index}" type="button">
                    ${lang}
                </button>
            </li>
        `;
    });

    html += `
                </ul>
                <div class="tab-content">
    `;

    // Tab content
    examples.forEach((example, index) => {
        const lang = (example.language || example.lang || 'text').toLowerCase();
        const code = example.code || '';
        const isActive = index === 0 ? 'active show' : '';

        html += `
            <div class="tab-pane fade ${isActive}" id="${tabsId}-${index}" role="tabpanel">
                <div class="position-relative">
                    <button class="btn btn-sm btn-outline-secondary position-absolute top-0 end-0 m-2 copy-btn"
                            onclick="copyCode(this)" title="Copy code">
                        <i class="fas fa-copy"></i>
                    </button>
                    <pre class="line-numbers"><code class="language-${lang}">${escapeHtml(code)}</code></pre>
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>
        </div>
    `;

    return html;
}

/**
 * Render API metadata (endpoints, auth, params, errors)
 */
function renderApiMetadata(data) {
    let html = '';

    // Endpoints
    if (data.endpoints && data.endpoints.length > 0) {
        html += `
            <div class="api-section mt-3">
                <h6 class="mb-2"><i class="fas fa-link me-2"></i>Endpoints</h6>
                <div class="list-group">
        `;

        data.endpoints.forEach(endpoint => {
            const method = endpoint.method || 'GET';
            const path = endpoint.path || '/';
            const desc = endpoint.description || '';

            html += `
                <div class="list-group-item bg-secondary">
                    <div class="d-flex align-items-center">
                        <span class="badge bg-${getMethodColor(method)} me-2">${method}</span>
                        <code class="flex-grow-1">${path}</code>
                    </div>
                    ${desc ? `<small class="text-muted d-block mt-1">${desc}</small>` : ''}
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    // Authentication
    if (data.authentication) {
        const auth = data.authentication;
        html += `
            <div class="api-section mt-3">
                <h6 class="mb-2"><i class="fas fa-lock me-2"></i>Authentication</h6>
                <div class="alert alert-info mb-0">
                    <strong>Type:</strong> ${auth.type || 'N/A'}
                    ${auth.header ? `<br><code>${auth.header}</code>` : ''}
                </div>
            </div>
        `;
    }

    // Parameters
    if (data.parameters && data.parameters.length > 0) {
        html += `
            <div class="api-section mt-3">
                <h6 class="mb-2"><i class="fas fa-sliders-h me-2"></i>Parameters</h6>
                <div class="table-responsive">
                    <table class="table table-sm table-dark">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Required</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        data.parameters.forEach(param => {
            const required = param.required ?
                '<span class="badge bg-danger">Required</span>' :
                '<span class="badge bg-secondary">Optional</span>';

            html += `
                <tr>
                    <td><code>${param.name}</code></td>
                    <td>${param.type || param.in || 'string'}</td>
                    <td>${required}</td>
                </tr>
            `;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // Error Codes
    if (data.error_codes && data.error_codes.length > 0) {
        html += `
            <div class="api-section mt-3">
                <h6 class="mb-2"><i class="fas fa-exclamation-triangle me-2"></i>Error Codes</h6>
                <div class="list-group">
        `;

        data.error_codes.forEach(error => {
            const code = error.code || error.status_code;
            const message = error.message;

            html += `
                <div class="list-group-item bg-secondary">
                    <span class="badge bg-danger me-2">${code}</span>
                    ${message}
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    return html;
}

/**
 * Render sources
 */
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    let html = `
        <div class="sources-section mt-3">
            <h6 class="mb-2"><i class="fas fa-book me-2"></i>Sources</h6>
            <div class="list-group list-group-flush">
    `;

    sources.forEach(source => {
        const title = source.title || 'Untitled';
        const url = source.url || '#';
        const relevance = source.relevance ? Math.round(source.relevance * 100) : 0;

        html += `
            <a href="${url}" target="_blank" class="list-group-item list-group-item-action bg-secondary d-flex justify-content-between align-items-center">
                <span>
                    <i class="fas fa-external-link-alt me-2"></i>
                    ${title}
                </span>
                <span class="badge bg-primary">${relevance}%</span>
            </a>
        `;
    });

    html += `
            </div>
        </div>
    `;

    return html;
}

/**
 * Render related questions
 */
function renderRelatedQuestions(questions) {
    if (!questions || questions.length === 0) return '';

    let html = `
        <div class="related-section mt-3">
            <h6 class="mb-2"><i class="fas fa-lightbulb me-2"></i>Related Topics</h6>
            <div class="d-flex flex-wrap gap-2">
    `;

    questions.forEach(q => {
        html += `
            <button class="btn btn-sm btn-outline-primary" onclick="askRelated('${escapeQuotes(q)}')">
                ${q}
            </button>
        `;
    });

    html += `
            </div>
        </div>
    `;

    return html;
}

/**
 * Render performance info (debug mode)
 */
function renderPerformanceInfo(data) {
    if (!data.perf_metrics) return '';

    const m = data.perf_metrics;
    const responseTime = data.response_time || m.total || 0;
    const cached = data.cached ? '⚡ Cached' : '';

    return `
        <div class="perf-info mt-3">
            <details>
                <summary class="text-muted small">
                    <i class="fas fa-chart-line me-1"></i>
                    Performance: ${responseTime.toFixed(2)}s ${cached}
                </summary>
                <ul class="small text-muted mt-2 mb-0">
                    <li>Embedding: ${(m.embedding_generation || 0).toFixed(3)}s</li>
                    <li>ChromaDB: ${(m.chromadb_query || 0).toFixed(3)}s</li>
                    <li>LLM: ${(m.llm_generation || 0).toFixed(3)}s</li>
                </ul>
            </details>
        </div>
    `;
}

/**
 * Display error message
 */
function displayError(message) {
    const messagesDiv = document.getElementById('chat-messages');

    const errorHtml = `
        <div class="message assistant-message mb-3">
            <div class="d-flex align-items-start">
                <div class="avatar bg-danger rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-exclamation-triangle text-white"></i>
                </div>
                <div class="message-content flex-grow-1">
                    <div class="message-bubble bg-dark border border-danger p-3 rounded">
                        <div class="alert alert-danger mb-0">
                            <h6 class="alert-heading"><i class="fas fa-exclamation-circle me-2"></i>Error</h6>
                            <p class="mb-2">${escapeHtml(message)}</p>
                            <button class="btn btn-sm btn-danger" onclick="retryLastMessage()">
                                <i class="fas fa-redo me-1"></i>Try Again
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', errorHtml);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    showToast('An error occurred', 'error');
}

/**
 * Retry last message
 */
function retryLastMessage() {
    if (lastQuery) {
        document.getElementById('user-input').value = lastQuery;
        sendMessage();
    }
}

/**
 * Clear chat
 */
function clearChat() {
    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.innerHTML = `
        <div class="message assistant-message mb-3">
            <div class="d-flex align-items-start">
                <div class="avatar bg-primary rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                    <i class="fas fa-robot text-white"></i>
                </div>
                <div class="message-content flex-grow-1">
                    <div class="message-bubble bg-dark border p-3 rounded">
                        <p class="mb-2">👋 Chat cleared! How can I help you?</p>
                    </div>
                </div>
            </div>
        </div>
    `;

    showToast('Chat cleared', 'success');
}

/**
 * Ask related question
 */
function askRelated(question) {
    document.getElementById('user-input').value = question;
    sendMessage();
}

/**
 * Copy code to clipboard
 */
async function copyCode(button) {
    const pre = button.closest('.position-relative').querySelector('code');
    const code = pre.textContent;

    try {
        await navigator.clipboard.writeText(code);

        const icon = button.querySelector('i');
        const originalClass = icon.className;
        icon.className = 'fas fa-check';
        button.classList.add('btn-success');
        button.classList.remove('btn-outline-secondary');

        showToast('Code copied!', 'success');

        setTimeout(() => {
            icon.className = originalClass;
            button.classList.remove('btn-success');
            button.classList.add('btn-outline-secondary');
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
        showToast('Failed to copy code', 'error');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.innerHTML = `
        <i class="fas ${getToastIcon(type)} me-2"></i>
        ${message}
    `;

    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Helper functions
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeQuotes(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function getMethodColor(method) {
    const colors = {
        'GET': 'info',
        'POST': 'success',
        'PUT': 'warning',
        'DELETE': 'danger',
        'PATCH': 'secondary'
    };
    return colors[method.toUpperCase()] || 'secondary';
}

function getToastIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };
    return icons[type] || 'fa-info-circle';
}
