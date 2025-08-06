// Chat interface JavaScript
class ChatInterface {
    constructor() {
        this.chatForm = document.getElementById('chat-form');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.chatMessages = document.getElementById('chat-messages');
        this.clearBtn = document.getElementById('clear-chat');
        this.historyBtn = document.getElementById('history-btn');
        this.charCount = document.getElementById('char-count');
        this.rateLimitWarning = document.getElementById('rate-limit-warning');
        
        this.isLoading = false;
        this.conversationHistory = [];
        
        this.initializeEventListeners();
        this.checkSystemStatus();
    }
    
    initializeEventListeners() {
        // Form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Input character count
        this.chatInput.addEventListener('input', () => {
            const length = this.chatInput.value.length;
            this.charCount.textContent = length;
            
            if (length > 450) {
                this.charCount.parentElement.classList.add('text-warning');
            } else {
                this.charCount.parentElement.classList.remove('text-warning');
            }
        });
        
        // Clear chat
        this.clearBtn.addEventListener('click', () => {
            this.clearChat();
        });
        
        // History modal
        this.historyBtn.addEventListener('click', () => {
            this.loadHistory();
        });
        
        // Enter key handling
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (data.documents.document_count === 0) {
                this.showInitializeModal();
            }
        } catch (error) {
            console.error('Error checking system status:', error);
        }
    }
    
    async showInitializeModal() {
        const modal = new bootstrap.Modal(document.getElementById('initializeModal'));
        modal.show();
        
        try {
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            const initContent = document.getElementById('init-content');
            
            if (response.ok) {
                initContent.innerHTML = `
                    <div class="text-success">
                        <i class="fas fa-check-circle fa-2x mb-3"></i>
                        <p>System initialized successfully!</p>
                        <small class="text-muted">Processed ${data.document_count} documents</small>
                    </div>
                `;
                
                setTimeout(() => {
                    modal.hide();
                }, 2000);
            } else {
                initContent.innerHTML = `
                    <div class="text-danger">
                        <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                        <p>Initialization failed</p>
                        <small class="text-muted">${data.message}</small>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error initializing system:', error);
            document.getElementById('init-content').innerHTML = `
                <div class="text-danger">
                    <i class="fas fa-times-circle fa-2x mb-3"></i>
                    <p>Network error</p>
                    <small class="text-muted">Please check your connection and try again</small>
                </div>
            `;
        }
    }
    
    async sendMessage() {
        const query = this.chatInput.value.trim();
        
        if (!query || this.isLoading) {
            return;
        }
        
        // Disable input
        this.setLoading(true);
        
        // Add user message to chat
        this.addMessage('user', query);
        
        // Clear input
        this.chatInput.value = '';
        this.charCount.textContent = '0';
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            
            if (response.status === 429) {
                const errorData = await response.json();
                this.showRateLimitWarning(errorData);
                this.setLoading(false);
                return;
            }
            
            const data = await response.json();
            
            if (response.ok) {
                // Add assistant response
                this.addMessage('assistant', data.response, {
                    codeExamples: data.code_examples,
                    sources: data.sources,
                    relatedQuestions: data.related_questions,
                    responseTime: data.response_time,
                    cached: data.cached
                });
                
                // Update conversation history
                this.conversationHistory.push({
                    user: query,
                    assistant: data.response
                });
                
                // Keep only last 5 exchanges
                if (this.conversationHistory.length > 5) {
                    this.conversationHistory = this.conversationHistory.slice(-5);
                }
            } else {
                this.addMessage('assistant', `Sorry, I encountered an error: ${data.message || 'Unknown error'}`, {
                    isError: true
                });
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('assistant', 'Sorry, I\'m having trouble connecting to the server. Please try again later.', {
                isError: true
            });
        }
        
        this.setLoading(false);
    }
    
    addMessage(sender, content, options = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message mb-3 message-enter`;
        
        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="d-flex justify-content-end">
                    <div class="message-bubble bg-primary text-white p-3 rounded" style="max-width: 80%;">
                        ${this.escapeHtml(content)}
                    </div>
                    <div class="avatar bg-secondary rounded-circle d-flex align-items-center justify-content-center ms-3" style="width: 40px; height: 40px;">
                        <i class="fas fa-user text-white"></i>
                    </div>
                </div>
            `;
        } else {
            const isError = options.isError || false;
            const cached = options.cached || false;
            
            let messageContent = `
                <div class="d-flex align-items-start">
                    <div class="avatar bg-primary rounded-circle d-flex align-items-center justify-content-center me-3" style="width: 40px; height: 40px;">
                        <i class="fas fa-robot text-white"></i>
                    </div>
                    <div class="message-content flex-grow-1">
                        <div class="message-bubble bg-dark border p-3 rounded">
                            ${isError ? `<div class="text-danger">${this.escapeHtml(content)}</div>` : this.formatMessage(content)}
                            
                            ${options.codeExamples ? this.renderCodeExamples(options.codeExamples) : ''}
                            ${options.sources ? this.renderSources(options.sources) : ''}
                            ${options.relatedQuestions ? this.renderRelatedQuestions(options.relatedQuestions) : ''}
                            
                            <div class="response-metadata">
                                <div>
                                    ${cached ? '<span class="cached-indicator"><i class="fas fa-bolt me-1"></i>Cached</span>' : ''}
                                    ${options.responseTime ? `<span class="text-muted"><i class="fas fa-clock me-1"></i>${options.responseTime.toFixed(2)}s</span>` : ''}
                                </div>
                                ${!isError ? this.renderFeedbackButtons() : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            messageDiv.innerHTML = messageContent;
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Initialize syntax highlighting for code blocks
        if (sender === 'assistant' && !options.isError) {
            Prism.highlightAllUnder(messageDiv);
        }
    }
    
    formatMessage(content) {
        // Convert markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code class="language-text">$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }
    
    renderCodeExamples(codeExamples) {
        if (!codeExamples || codeExamples.length === 0) {
            return '';
        }
        
        return codeExamples.map(example => {
            const language = example.language || 'text';
            const code = this.escapeHtml(example.code);
            const explanation = example.explanation || '';
            
            return `
                <div class="code-block mt-3">
                    <div class="code-header">
                        <span><i class="fas fa-code me-2"></i>${language.toUpperCase()}</span>
                        <button class="copy-btn" onclick="copyToClipboard(this)" data-code="${this.escapeHtml(example.code)}">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                    <pre><code class="language-${language}">${code}</code></pre>
                    ${explanation ? `<div class="mt-2 text-muted small">${this.escapeHtml(explanation)}</div>` : ''}
                </div>
            `;
        }).join('');
    }
    
    renderSources(sources) {
        if (!sources || sources.length === 0) {
            return '';
        }
        
        const sourceItems = sources.map(source => {
            const icon = this.getSourceIcon(source.type);
            return `
                <div class="source-item">
                    <i class="source-icon ${icon}"></i>
                    <div class="flex-grow-1">
                        <a href="${source.url}" target="_blank" class="text-decoration-none">
                            ${this.escapeHtml(source.title)}
                        </a>
                        <small class="text-muted d-block">${source.type} • Relevance: ${(source.relevance * 100).toFixed(0)}%</small>
                    </div>
                </div>
            `;
        }).join('');
        
        return `
            <div class="sources">
                <h6><i class="fas fa-link me-2"></i>Sources</h6>
                ${sourceItems}
            </div>
        `;
    }
    
    renderRelatedQuestions(questions) {
        if (!questions || questions.length === 0) {
            return '';
        }
        
        const questionChips = questions.map(question => 
            `<span class="question-chip" onclick="askRelatedQuestion('${this.escapeHtml(question)}')">${this.escapeHtml(question)}</span>`
        ).join('');
        
        return `
            <div class="related-questions">
                <h6><i class="fas fa-lightbulb me-2"></i>Related Questions</h6>
                ${questionChips}
            </div>
        `;
    }
    
    renderFeedbackButtons() {
        return `
            <div class="feedback-buttons">
                <button class="feedback-btn" onclick="giveFeedback(this, 1)">
                    <i class="fas fa-thumbs-up me-1"></i>Helpful
                </button>
                <button class="feedback-btn" onclick="giveFeedback(this, -1)">
                    <i class="fas fa-thumbs-down me-1"></i>Not helpful
                </button>
            </div>
        `;
    }
    
    getSourceIcon(type) {
        switch (type) {
            case 'react': return 'fab fa-react text-info';
            case 'python': return 'fab fa-python text-warning';
            case 'fastapi': return 'fas fa-lightning-bolt text-success';
            default: return 'fas fa-file-alt text-secondary';
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        this.sendBtn.disabled = loading;
        this.chatInput.disabled = loading;
        
        if (loading) {
            this.sendBtn.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
        } else {
            this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    showRateLimitWarning(errorData) {
        this.rateLimitWarning.classList.remove('d-none');
        document.getElementById('rate-limit-text').textContent = 
            `${errorData.message}. Try again in ${errorData.retry_after} seconds.`;
        
        setTimeout(() => {
            this.rateLimitWarning.classList.add('d-none');
        }, errorData.retry_after * 1000);
    }
    
    clearChat() {
        // Keep the welcome message
        const welcomeMessage = this.chatMessages.querySelector('.assistant-message');
        this.chatMessages.innerHTML = '';
        if (welcomeMessage) {
            this.chatMessages.appendChild(welcomeMessage);
        }
        this.conversationHistory = [];
    }
    
    async loadHistory() {
        const modal = new bootstrap.Modal(document.getElementById('historyModal'));
        modal.show();
        
        try {
            const response = await fetch('/api/history');
            const history = await response.json();
            
            const historyContent = document.getElementById('history-content');
            
            if (history.length === 0) {
                historyContent.innerHTML = `
                    <div class="text-center text-muted">
                        <i class="fas fa-history fa-3x mb-3"></i>
                        <p>No conversation history available</p>
                    </div>
                `;
            } else {
                const historyItems = history.reverse().map(item => `
                    <div class="border-bottom border-secondary pb-3 mb-3">
                        <div class="mb-2">
                            <strong class="text-primary">Q:</strong> ${this.escapeHtml(item.query)}
                        </div>
                        <div class="mb-2">
                            <strong class="text-success">A:</strong> ${this.escapeHtml(item.response.substring(0, 200))}${item.response.length > 200 ? '...' : ''}
                        </div>
                        <small class="text-muted">
                            ${new Date(item.created_at).toLocaleString()} • 
                            ${item.response_time ? `${item.response_time.toFixed(2)}s` : ''} •
                            ${item.feedback === 1 ? '👍' : item.feedback === -1 ? '👎' : '❓'}
                        </small>
                    </div>
                `).join('');
                
                historyContent.innerHTML = historyItems;
            }
        } catch (error) {
            console.error('Error loading history:', error);
            document.getElementById('history-content').innerHTML = `
                <div class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                    <p>Error loading conversation history</p>
                </div>
            `;
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Global functions for button interactions
function copyToClipboard(button) {
    const code = button.getAttribute('data-code');
    navigator.clipboard.writeText(code).then(() => {
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check text-success"></i>';
        button.disabled = true;
        
        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.disabled = false;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy code:', err);
    });
}

function askRelatedQuestion(question) {
    const chatInterface = window.chatInterface;
    if (chatInterface) {
        chatInterface.chatInput.value = question;
        chatInterface.sendMessage();
    }
}

function giveFeedback(button, feedback) {
    // Find conversation ID (would need to be added to message rendering)
    const message = button.closest('.message');
    // For now, just provide visual feedback
    
    const buttons = button.parentElement.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active-positive', 'active-negative');
    });
    
    if (feedback === 1) {
        button.classList.add('active-positive');
    } else {
        button.classList.add('active-negative');
    }
    
    // Here you would normally send feedback to the server
    console.log('Feedback:', feedback);
}

// Initialize chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.chatInterface = new ChatInterface();
});
