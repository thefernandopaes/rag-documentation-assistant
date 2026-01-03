"""
Text processing utilities for DocRag system
Handles text cleaning, keyword extraction, and content analysis
"""

import re
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for text processing operations"""
    
    def __init__(self):
        # Common stop words for filtering
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'but',
            'they', 'have', 'had', 'what', 'when', 'where', 'who', 'which',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now'
        }
        
        # Technical terms that should not be filtered
        self.technical_terms = {
            'api', 'rest', 'http', 'json', 'xml', 'css', 'html', 'js',
            'python', 'react', 'fastapi', 'django', 'flask', 'node',
            'component', 'function', 'class', 'method', 'variable',
            'database', 'query', 'model', 'view', 'controller'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Remove URLs (keep domain for context)
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        
        # Clean up special characters but keep code-related ones
        text = re.sub(r'[^\w\s\-_\.\(\)\[\]\{\}:;,\'"<>/\\@#$%^&*+=|`~]', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Clean and tokenize
        cleaned_text = self.clean_text(text.lower())
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', cleaned_text)
        
        # Filter words
        filtered_words = []
        for word in words:
            if (len(word) >= 3 and 
                (word not in self.stop_words or word in self.technical_terms)):
                filtered_words.append(word)
        
        # Count frequency and return top keywords
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from text"""
        code_blocks = []
        
        # Pattern for fenced code blocks
        pattern = r'```(\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            language = match[0].strip() if match[0] else 'text'
            code = match[1].strip()
            
            if code:  # Only add non-empty blocks
                code_blocks.append({
                    'language': language,
                    'code': code,
                    'lines': len(code.split('\n'))
                })
        
        # Pattern for inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, text)
        
        for match in inline_matches:
            if len(match) > 5:  # Only longer inline code
                code_blocks.append({
                    'language': 'text',
                    'code': match,
                    'lines': 1,
                    'type': 'inline'
                })
        
        return code_blocks
    
    def analyze_content_type(self, text: str) -> Dict[str, Any]:
        """Analyze the type and characteristics of content"""
        analysis = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_code': False,
            'languages': [],
            'complexity': 'low',
            'topics': []
        }
        
        # Check for code blocks
        code_blocks = self.extract_code_blocks(text)
        if code_blocks:
            analysis['has_code'] = True
            analysis['languages'] = list(set(block['language'] for block in code_blocks))
        
        # Analyze complexity based on technical terms
        keywords = self.extract_keywords(text, 20)
        tech_keywords = [kw for kw in keywords if kw in self.technical_terms]
        
        if len(tech_keywords) > 5:
            analysis['complexity'] = 'high'
        elif len(tech_keywords) > 2:
            analysis['complexity'] = 'medium'
        
        # Identify topics
        topics = []
        text_lower = text.lower()
        
        topic_patterns = {
            'react': r'\b(react|jsx|component|props|state|hook)\b',
            'python': r'\b(python|def|import|class|django|flask)\b',
            'fastapi': r'\b(fastapi|pydantic|uvicorn|endpoint)\b',
            'database': r'\b(database|sql|query|table|model)\b',
            'api': r'\b(api|rest|endpoint|request|response)\b',
            'authentication': r'\b(auth|login|token|jwt|session)\b',
            'frontend': r'\b(frontend|ui|css|html|javascript)\b',
            'backend': r'\b(backend|server|api|database)\b'
        }
        
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, text_lower):
                topics.append(topic)
        
        analysis['topics'] = topics
        
        return analysis
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Create a simple extractive summary"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences based on keyword frequency
        keywords = self.extract_keywords(text)
        keyword_set = set(keywords)
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            
            # Score based on keyword presence
            for word in words:
                if word in keyword_set:
                    score += 1
            
            # Prefer sentences not at the very beginning or end
            if 0 < i < len(sentences) - 1:
                score += 0.5
            
            sentence_scores.append((score, sentence))
        
        # Get top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for score, sent in sentence_scores[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text or chunk_size <= 0:
            return []
        
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            return [{
                'text': text,
                'start_idx': 0,
                'end_idx': len(text),
                'word_count': len(words),
                'chunk_id': 0
            }]
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'word_count': len(chunk_words),
                'chunk_id': chunk_id,
                'keywords': self.extract_keywords(chunk_text, 5)
            })
            
            # Move start position, accounting for overlap
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - overlap
            if start_idx < 0:
                start_idx = 0
            
            chunk_id += 1
        
        return chunks

# Convenience functions for direct use
def clean_text(text: str) -> str:
    """Clean text using default processor"""
    processor = TextProcessor()
    return processor.clean_text(text)

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords using default processor"""
    processor = TextProcessor()
    return processor.extract_keywords(text, max_keywords)

def analyze_content(text: str) -> Dict[str, Any]:
    """Analyze content using default processor"""
    processor = TextProcessor()
    return processor.analyze_content_type(text)
