import os
import sys
import time
import argparse
from hashlib import sha256

from app import create_app, db
from document_processor import DocumentProcessor
from models import DocumentChunk


def main():
    parser = argparse.ArgumentParser(description='Offline ingestion job for DocRag.')
    parser.add_argument('--use-sample', action='store_true', help='Force using sample docs regardless of config')
    parser.add_argument('--max-pages', type=int, help='Override max pages per source')
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        processor = DocumentProcessor()
        if args.max_pages is not None:
            os.environ['DOC_MAX_PAGES_PER_SOURCE'] = str(args.max_pages)
        if args.use_sample:
            os.environ['DOC_USE_SAMPLE'] = 'true'

        documents = processor.process_documentation_sources()
        if not documents:
            print('No documents processed.', file=sys.stderr)
            sys.exit(1)

        app.rag_engine.upsert_documents(documents)

        upserts = 0
        for doc in documents:
            content_hash = sha256(doc['content'].encode('utf-8')).hexdigest()
            exists = DocumentChunk.query.filter_by(source_url=doc['source_url'], content_hash=content_hash).first()
            if exists:
                continue
            db.session.add(DocumentChunk(
                source_url=doc['source_url'],
                title=doc['title'],
                content=doc['content'][:1000],
                chunk_index=0,
                doc_type=doc['doc_type'],
                version=doc['version'],
                content_hash=content_hash,
            ))
            upserts += 1
        db.session.commit()
        print(f'Ingestion complete. Upserted metadata rows: {upserts}.')


if __name__ == '__main__':
    main()

