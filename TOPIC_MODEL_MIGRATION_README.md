# Topic Model Meta Migration

This migration adds support for topic modeling metadata and versioning to the Augur database.

## Overview

The migration creates a new `topic_model_meta` table and adds foreign key relationships to existing `repo_topic` and `topic_words` tables to track which topic model was used to generate the data.

## Changes Made

### 1. New Table: `topic_model_meta`

Creates a comprehensive table to store topic model metadata:

- `model_id` (UUID, Primary Key): Unique identifier for each topic model
- `model_method` (Text, Required): The algorithm used (e.g., "LDA", "HDP", "NMF")
- `num_topics` (Integer, Required): Number of topics in the model
- `num_words_per_topic` (Integer, Required): Number of words per topic
- `training_parameters` (JSONB, Required): Model-specific training parameters
- `model_file_paths` (JSONB, Required): Paths to saved model artifacts
- `coherence_score` (Float, Optional): Topic coherence score
- `perplexity_score` (Float, Optional): Model perplexity score
- `training_start_time` (Timestamp, Required): When training started
- `training_end_time` (Timestamp, Required): When training completed
- `tool_source` (Text, Required): Source of the tool
- `tool_version` (Text, Required): Version of the tool
- `data_source` (Text, Required): Source of the training data
- `data_collection_date` (Timestamp, Auto): When the record was created

### 2. Foreign Key Columns

Adds `model_id` (UUID, nullable) columns to:
- `repo_topic` table
- `topic_words` table

These columns allow linking topic data to the specific model that generated it.

### 3. Indexes

Creates indexes on the new foreign key columns for better query performance:
- `idx_repo_topic_model_id`
- `idx_topic_words_model_id`

## Backward Compatibility

- All new columns are nullable to avoid issues with existing data
- Foreign key constraints use `ON DELETE SET NULL` to handle model deletion gracefully
- Existing data will have `model_id` set to NULL until explicitly linked

## Usage

### Creating a Topic Model Record

```python
from augur.application.db.models import TopicModelMeta
import uuid
from datetime import datetime

model = TopicModelMeta(
    model_id=uuid.uuid4(),
    model_method="LDA",
    num_topics=10,
    num_words_per_topic=20,
    training_parameters={"alpha": 0.1, "beta": 0.01},
    model_file_paths={"model": "/path/to/model.pkl", "vocab": "/path/to/vocab.pkl"},
    coherence_score=0.85,
    perplexity_score=120.5,
    training_start_time=datetime.now(),
    training_end_time=datetime.now(),
    tool_source="topic_modeling_worker",
    tool_version="1.0",
    data_source="github_issues"
)
session.add(model)
session.commit()
```

### Linking Topic Data to Models

```python
# When creating topic words
topic_word = TopicWord(
    topic_id=1,
    word="python",
    word_prob=0.8,
    model_id=model.model_id,  # Link to specific model
    tool_source="test",
    tool_version="1.0",
    data_source="test"
)

# When creating repo topics
repo_topic = RepoTopic(
    repo_id=1,
    topic_id=1,
    topic_prob=0.7,
    model_id=model.model_id,  # Link to specific model
    tool_source="test",
    tool_version="1.0",
    data_source="test"
)
```

### Querying with Model Information

```python
# Get all topic words for a specific model
words = session.query(TopicWord).filter_by(model_id=model_id).all()

# Get model information for topic words
result = session.query(TopicWord, TopicModelMeta).join(
    TopicModelMeta, TopicWord.model_id == TopicModelMeta.model_id
).filter_by(model_id=model_id).all()
```

## Migration Commands

### Apply Migration
```bash
cd augur
alembic upgrade head
```

### Rollback Migration
```bash
cd augur
alembic downgrade 31
```

### Test Migration
```bash
python test_migration.py
```

## Benefits

1. **Model Versioning**: Track different versions of topic models
2. **Reproducibility**: Link results to specific model configurations
3. **Performance Tracking**: Compare coherence and perplexity scores across models
4. **Data Lineage**: Understand which model generated which topic data
5. **Flexibility**: Support multiple topic modeling algorithms and parameters

## Notes

- The migration is designed to be safe for production use
- Existing data will continue to work without modification
- New topic modeling workflows should include model metadata
- Consider running the test script before applying to production 