"""Add topic model meta table and foreign keys

Revision ID: 34
Revises: 33
Create Date: 2025-01-27

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '34'
down_revision = '33'
branch_labels = None
depends_on = None

def upgrade():
    """Create topic_model_meta table and add foreign key columns"""
    # Create topic_model_meta table
    op.create_table(
        'topic_model_meta',
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_method', sa.Text(), nullable=False),
        sa.Column('num_topics', sa.Integer(), nullable=False),
        sa.Column('num_words_per_topic', sa.Integer(), nullable=False),
        sa.Column('training_parameters', postgresql.JSONB(), nullable=False),
        sa.Column('model_file_paths', postgresql.JSONB(), nullable=False),
        sa.Column('coherence_score', sa.Float(), nullable=True),
        sa.Column('perplexity_score', sa.Float(), nullable=True),
        sa.Column('training_start_time', sa.TIMESTAMP(), nullable=False),
        sa.Column('training_end_time', sa.TIMESTAMP(), nullable=False),
        sa.Column('tool_source', sa.Text(), nullable=False),
        sa.Column('tool_version', sa.Text(), nullable=False),
        sa.Column('data_source', sa.Text(), nullable=False),
        sa.Column('data_collection_date', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('model_id'),
        schema='augur_data'
    )
    # Add model_id column to repo_topic table (nullable to handle existing data)
    op.add_column('repo_topic', 
                  sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
                  schema='augur_data')
    # Add model_id column to topic_words table (nullable to handle existing data)
    op.add_column('topic_words', 
                  sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
                  schema='augur_data')
    # Create foreign key constraints
    op.create_foreign_key(
        'fk_repo_topic_model_id',
        'repo_topic', 'topic_model_meta',
        ['model_id'], ['model_id'],
        source_schema='augur_data', referent_schema='augur_data',
        ondelete='SET NULL', onupdate='CASCADE'
    )
    op.create_foreign_key(
        'fk_topic_words_model_id',
        'topic_words', 'topic_model_meta',
        ['model_id'], ['model_id'],
        source_schema='augur_data', referent_schema='augur_data',
        ondelete='SET NULL', onupdate='CASCADE'
    )
    # Create indexes for better query performance
    op.create_index('idx_repo_topic_model_id', 'repo_topic', ['model_id'], schema='augur_data')
    op.create_index('idx_topic_words_model_id', 'topic_words', ['model_id'], schema='augur_data')

def downgrade():
    """Remove topic_model_meta table and foreign key columns"""
    # Drop indexes
    op.drop_index('idx_topic_words_model_id', table_name='topic_words', schema='augur_data')
    op.drop_index('idx_repo_topic_model_id', table_name='repo_topic', schema='augur_data')
    # Drop foreign key constraints
    op.drop_constraint('fk_topic_words_model_id', 'topic_words', schema='augur_data', type_='foreignkey')
    op.drop_constraint('fk_repo_topic_model_id', 'repo_topic', schema='augur_data', type_='foreignkey')
    # Drop model_id columns
    op.drop_column('topic_words', 'model_id', schema='augur_data')
    op.drop_column('repo_topic', 'model_id', schema='augur_data')
    # Drop topic_model_meta table
    op.drop_table('topic_model_meta', schema='augur_data') 