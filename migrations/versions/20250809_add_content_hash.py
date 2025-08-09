from alembic import op
import sqlalchemy as sa

revision = '20250809_add_content_hash'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('document_chunk') as batch_op:
        batch_op.add_column(sa.Column('content_hash', sa.String(length=64), nullable=True))
    op.create_index(
        'ix_document_chunk_source_url_content_hash',
        'document_chunk',
        ['source_url', 'content_hash'],
        unique=True
    )


def downgrade():
    op.drop_index('ix_document_chunk_source_url_content_hash', table_name='document_chunk')
    with op.batch_alter_table('document_chunk') as batch_op:
        batch_op.drop_column('content_hash')

