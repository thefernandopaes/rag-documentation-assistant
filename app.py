import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

def create_app():
    # Create the app
    app = Flask(__name__)
    # Prefer centralized config for session secret
    app.secret_key = Config.SESSION_SECRET or os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
    # Trust proxy headers for correct scheme/host/port/IP when behind TLS-terminating proxy
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    # Secure cookies and URL scheme when in production behind TLS
    if Config._is_production():
        app.config.update(
            PREFERRED_URL_SCHEME='https',
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
        )
    
    # Configure the database (supports Postgres via DATABASE_URL)
    app.config["SQLALCHEMY_DATABASE_URI"] = Config.get_database_uri()
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = Config.get_sqlalchemy_engine_options()
    
    # Initialize extensions
    db.init_app(app)
    
    with app.app_context():
        # Import models to create tables
        import models  # noqa: F401
        db.create_all()
        
        # Register routes
        from routes import main_bp
        app.register_blueprint(main_bp)
        
        # Initialize RAG system
        from rag_engine import RAGEngine
        app.rag_engine = RAGEngine()
        logger.info("RAG Engine initialized")
    
    return app

# Create app instance
app = create_app()
