"""
Script to automatically update imports after reorganization
"""
import os
import re

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Core
    'from rag_engine_async import': 'from app.core.rag_engine import',
    'import rag_engine_async': 'import app.core.rag_engine',
    'from cache_manager_inmemory import': 'from app.core.cache import',
    'from code_generator import': 'from app.core.code_generator import',

    # API
    'from routes_async import': 'from app.api.routes import',
    'from schemas import': 'from app.api.schemas import',
    'from dependencies import': 'from app.api.dependencies import',

    # Database
    'from database_async import': 'from app.db.database import',
    'from models_async import': 'from app.db.models import',

    # Services
    'from document_processor_async import': 'from app.services.document_processor import',
    'from api_discovery import': 'from app.services.api_discovery import',

    # Middleware
    'from rate_limiter import': 'from app.middleware.rate_limiter import',
    'from security import': 'from app.middleware.security import',
    'from monitoring import': 'from app.middleware.monitoring import',

    # Config
    'from uvicorn_config import': 'from config.uvicorn import',
    'from gunicorn.conf import': 'from config.gunicorn import',
}

def update_imports_in_file(filepath):
    """Update imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = []

        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes_made.append(f"{old_import} -> {new_import}")

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made

        return False, []

    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False, []

def main():
    """Main function to update all files"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    files_updated = 0
    total_changes = 0

    # Files to update
    files_to_check = [
        'fastapi_app.py',
        'run_dev.py',
        'run_prod.py',
        'app/api/routes.py',
        'app/api/dependencies.py',
        'app/core/rag_engine.py',
        'app/core/cache.py',
        'app/core/code_generator.py',
        'app/db/database.py',
        'app/services/document_processor.py',
        'app/services/api_discovery.py',
        'app/middleware/rate_limiter.py',
        'app/middleware/security.py',
        'config/uvicorn.py',
        'config/gunicorn.py',
    ]

    # Also check all test files
    for root, dirs, files in os.walk(os.path.join(root_dir, 'tests')):
        for file in files:
            if file.endswith('.py'):
                relative_path = os.path.relpath(os.path.join(root, file), root_dir)
                files_to_check.append(relative_path)

    print("=" * 60)
    print("UPDATING IMPORTS")
    print("=" * 60)

    for filepath in files_to_check:
        full_path = os.path.join(root_dir, filepath)
        if os.path.exists(full_path):
            updated, changes = update_imports_in_file(full_path)
            if updated:
                files_updated += 1
                total_changes += len(changes)
                print(f"\n[OK] {filepath}")
                for change in changes:
                    print(f"  - {change}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {files_updated} files updated, {total_changes} imports changed")
    print("=" * 60)

if __name__ == '__main__':
    main()
