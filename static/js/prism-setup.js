/**
 * Prism.js configuration and setup for DocRag
 * Handles syntax highlighting for code blocks
 */

// Configure Prism.js
if (typeof Prism !== 'undefined') {
    // Disable automatic highlighting on page load
    Prism.manual = true;
    
    // Configure language aliases
    Prism.languages.js = Prism.languages.javascript;
    Prism.languages.py = Prism.languages.python;
    Prism.languages.sh = Prism.languages.bash;
    Prism.languages.shell = Prism.languages.bash;
    
    // Custom language definitions if needed
    if (!Prism.languages.fastapi) {
        Prism.languages.fastapi = Prism.languages.python;
    }
    
    // Add line numbers plugin configuration
    if (Prism.plugins && Prism.plugins.lineNumbers) {
        Prism.plugins.lineNumbers.config = {
            startFrom: 1
        };
    }
    
    // Custom highlighting function
    window.highlightCode = function(element) {
        if (!element) return;
        
        const codeBlocks = element.querySelectorAll('code[class*="language-"]');
        codeBlocks.forEach(block => {
            if (!block.classList.contains('highlighted')) {
                Prism.highlightElement(block);
                block.classList.add('highlighted');
            }
        });
    };
    
    // Auto-detect language for code blocks without explicit language
    window.detectLanguage = function(code) {
        // Simple heuristics for language detection
        if (code.includes('def ') || code.includes('import ') || code.includes('from ')) {
            return 'python';
        } else if (code.includes('function') || code.includes('const ') || code.includes('let ')) {
            return 'javascript';
        } else if (code.includes('@app.') || code.includes('FastAPI')) {
            return 'python'; // FastAPI is Python-based
        } else if (code.includes('<') && code.includes('>')) {
            return 'html';
        } else if (code.includes('{') && code.includes('}')) {
            return 'json';
        } else if (code.includes('$') || code.includes('curl')) {
            return 'bash';
        }
        return 'text';
    };
    
    // Initialize highlighting for existing content
    document.addEventListener('DOMContentLoaded', function() {
        // Highlight all existing code blocks
        Prism.highlightAll();
        
        // Set up observer for dynamic content
        if (window.MutationObserver) {
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList') {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                window.highlightCode(node);
                            }
                        });
                    }
                });
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    });
}

// Copy to clipboard functionality
window.copyToClipboard = function(button) {
    const code = button.getAttribute('data-code');
    
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(function() {
            showCopySuccess(button);
        }).catch(function(err) {
            console.error('Failed to copy code:', err);
            fallbackCopyToClipboard(code, button);
        });
    } else {
        fallbackCopyToClipboard(code, button);
    }
};

// Fallback copy method for older browsers
function fallbackCopyToClipboard(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showCopySuccess(button);
        } else {
            showCopyError(button);
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showCopyError(button);
    }
    
    document.body.removeChild(textArea);
}

// Show copy success feedback
function showCopySuccess(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check text-success"></i>';
    button.disabled = true;
    button.classList.add('success');
    
    setTimeout(function() {
        button.innerHTML = originalContent;
        button.disabled = false;
        button.classList.remove('success');
    }, 2000);
}

// Show copy error feedback
function showCopyError(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-times text-danger"></i>';
    button.classList.add('error');
    
    setTimeout(function() {
        button.innerHTML = originalContent;
        button.classList.remove('error');
    }, 2000);
}

// Format code blocks with proper structure
window.formatCodeBlock = function(code, language = 'text', explanation = '') {
    const languageLabel = language.toUpperCase();
    const escapedCode = escapeHtml(code);
    const copyButton = `<button class="copy-btn" onclick="copyToClipboard(this)" data-code="${escapeHtml(code)}">
        <i class="fas fa-copy"></i>
    </button>`;
    
    let html = `
        <div class="code-block">
            <div class="code-header">
                <span><i class="fas fa-code me-2"></i>${languageLabel}</span>
                ${copyButton}
            </div>
            <pre><code class="language-${language}">${escapedCode}</code></pre>
    `;
    
    if (explanation) {
        html += `<div class="code-explanation text-muted small mt-2">${escapeHtml(explanation)}</div>`;
    }
    
    html += '</div>';
    return html;
};

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add styles for copy button states
if (document.head) {
    const style = document.createElement('style');
    style.textContent = `
        .copy-btn.success {
            background-color: rgba(25, 135, 84, 0.1) !important;
            color: #198754 !important;
        }
        
        .copy-btn.error {
            background-color: rgba(220, 53, 69, 0.1) !important;
            color: #dc3545 !important;
        }
        
        .code-block {
            margin: 1rem 0;
        }
        
        .code-header {
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem 0.375rem 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.875rem;
        }
        
        .code-explanation {
            padding: 0.5rem 1rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 0 0 0.375rem 0.375rem;
            margin-top: 0 !important;
        }
    `;
    document.head.appendChild(style);
}
