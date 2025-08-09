import json
import logging
import time
import uuid
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session, current_app
from config import Config
from models import Conversation, DocumentChunk
from hashlib import sha256
from app import db
from rate_limiter import rate_limit_decorator
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@main_bp.route('/chat')
def chat():
    """Chat interface page"""
    # Initialize session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('chat.html', session_id=session['session_id'])

@main_bp.route('/api/chat', methods=['POST'])
@rate_limit_decorator
def api_chat():
    """API endpoint for chat queries"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Get conversation history
        conversation_history = get_conversation_history(session_id)
        
        # Generate response using RAG engine
        start_time = time.time()
        
        try:
            rag_engine = current_app.rag_engine
            response_data = rag_engine.generate_response(query, conversation_history)
        except Exception as e:
            logger.error(f"RAG engine error: {e}")
            return jsonify({
                'error': 'Failed to generate response',
                'message': 'Our AI system is temporarily unavailable. Please try again later.'
            }), 500
        
        # Save conversation to database
        try:
            conversation = Conversation(
                session_id=session_id,
                user_query=query,
                ai_response=response_data['response'],
                sources=json.dumps(response_data.get('sources', [])),
                response_time=response_data.get('response_time', time.time() - start_time)
            )
            db.session.add(conversation)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            # Continue even if saving fails
        
        return jsonify({
            'response': response_data['response'],
            'code_examples': response_data.get('code_examples', []),
            'sources': response_data.get('sources', []),
            'related_questions': response_data.get('related_questions', []),
            'response_time': response_data.get('response_time', 0),
            'cached': response_data.get('cached', False)
        })
        
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500

@main_bp.route('/api/feedback', methods=['POST'])
@rate_limit_decorator
def api_feedback():
    """API endpoint for user feedback"""
    try:
        data = request.get_json()
        
        if not data or 'conversation_id' not in data or 'feedback' not in data:
            return jsonify({'error': 'Conversation ID and feedback are required'}), 400
        
        conversation_id = data['conversation_id']
        feedback = data['feedback']  # 1 for thumbs up, -1 for thumbs down
        
        if feedback not in [1, -1]:
            return jsonify({'error': 'Feedback must be 1 (thumbs up) or -1 (thumbs down)'}), 400
        
        # Update conversation with feedback
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        conversation.feedback = feedback
        db.session.commit()
        
        logger.info(f"Feedback received for conversation {conversation_id}: {feedback}")
        
        return jsonify({'message': 'Feedback recorded successfully'})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to record feedback'
        }), 500

@main_bp.route('/api/history')
def api_history():
    """Get conversation history for current session"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify([])
        
        conversations = Conversation.query.filter_by(session_id=session_id)\
                                         .order_by(Conversation.created_at.desc())\
                                         .limit(20).all()
        
        history = []
        for conv in conversations:
            sources = []
            try:
                if conv.sources:
                    sources = json.loads(conv.sources)
            except json.JSONDecodeError:
                pass
            
            history.append({
                'id': conv.id,
                'query': conv.user_query,
                'response': conv.ai_response,
                'sources': sources,
                'created_at': conv.created_at.isoformat(),
                'response_time': conv.response_time,
                'feedback': conv.feedback
            })
        
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify([])

@main_bp.route('/api/stats')
def api_stats():
    """Get system statistics"""
    try:
        # Get RAG engine stats
        rag_engine = current_app.rag_engine
        collection_stats = rag_engine.get_collection_stats()
        
        # Get conversation stats
        total_conversations = Conversation.query.count()
        avg_response_time = db.session.query(db.func.avg(Conversation.response_time)).scalar() or 0
        
        # Get feedback stats
        positive_feedback = Conversation.query.filter_by(feedback=1).count()
        negative_feedback = Conversation.query.filter_by(feedback=-1).count()
        
        # Get cache stats
        cache_stats = rag_engine.cache.get_stats()
        
        return jsonify({
            'documents': collection_stats,
            'conversations': {
                'total': total_conversations,
                'avg_response_time': round(avg_response_time, 2),
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback
            },
            'cache': cache_stats,
            'system': {
                'is_production': Config._is_production()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'documents': {'document_count': 0},
            'conversations': {'total': 0, 'avg_response_time': 0},
            'cache': {'total_entries': 0}
        })

@main_bp.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Initialize the RAG system with documents"""
    try:
        # Protect in production with ADMIN_API_KEY
        if Config._is_production():
            provided = request.headers.get('X-Admin-Api-Key') or request.args.get('admin_key')
            if not provided or provided != Config.ADMIN_API_KEY:
                return jsonify({'error': 'Unauthorized'}), 401
        # Check if documents already exist
        existing_docs = DocumentChunk.query.count()
        if existing_docs > 0:
            return jsonify({
                'message': f'System already initialized with {existing_docs} document chunks',
                'status': 'already_initialized'
            })
        
        # Process and add documents
        processor = DocumentProcessor()
        documents = processor.process_documentation_sources()
        
        if not documents:
            return jsonify({
                'error': 'No documents processed',
                'message': 'Failed to process any documentation sources'
            }), 500
        
        # Add documents to RAG engine (idempotent IDs inside RAGEngine)
        rag_engine = current_app.rag_engine
        rag_engine.add_documents(documents)
        
        # Save document metadata to database with idempotency (content hash)
        for doc in documents:
            content_hash = sha256(doc['content'].encode('utf-8')).hexdigest()
            exists = DocumentChunk.query.filter_by(source_url=doc['source_url'], content_hash=content_hash).first()
            if exists:
                continue
            doc_chunk = DocumentChunk(
                source_url=doc['source_url'],
                title=doc['title'],
                content=doc['content'][:1000],  # Store first 1000 chars for reference
                chunk_index=0,
                doc_type=doc['doc_type'],
                version=doc['version'],
                content_hash=content_hash
            )
            db.session.add(doc_chunk)
        
        db.session.commit()
        
        logger.info(f"Initialized RAG system with {len(documents)} documents")
        
        return jsonify({
            'message': f'System initialized successfully with {len(documents)} documents',
            'status': 'initialized',
            'document_count': len(documents)
        })
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return jsonify({
            'error': 'Initialization failed',
            'message': str(e)
        }), 500

def get_conversation_history(session_id: str, limit: int = 5) -> list:
    """Get recent conversation history for context"""
    try:
        conversations = Conversation.query.filter_by(session_id=session_id)\
                                         .order_by(Conversation.created_at.desc())\
                                         .limit(limit).all()
        
        history = []
        for conv in reversed(conversations):  # Reverse to get chronological order
            history.append({
                'user': conv.user_query,
                'assistant': conv.ai_response
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []

@main_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@main_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500
