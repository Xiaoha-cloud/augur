-- Create topic_model_meta table in augur_data schema
CREATE TABLE IF NOT EXISTS augur_data.topic_model_meta (
    model_id UUID PRIMARY KEY,
    model_method TEXT NOT NULL,
    num_topics INT NOT NULL,
    num_words_per_topic INT NOT NULL,
    training_parameters JSONB NOT NULL,
    model_file_paths JSONB NOT NULL,
    coherence_score FLOAT,
    perplexity_score FLOAT,
    training_start_time TIMESTAMP NOT NULL,
    training_end_time TIMESTAMP NOT NULL,
    tool_source TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    data_source TEXT NOT NULL,
    data_collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add model_id to repo_topic table
ALTER TABLE augur_data.repo_topic 
ADD COLUMN IF NOT EXISTS model_id UUID REFERENCES augur_data.topic_model_meta(model_id);

-- Add model_id to topic_words table
ALTER TABLE augur_data.topic_words
ADD COLUMN IF NOT EXISTS model_id UUID REFERENCES augur_data.topic_model_meta(model_id);

-- Create index on model_id for better query performance
CREATE INDEX IF NOT EXISTS idx_repo_topic_model_id ON augur_data.repo_topic(model_id);
CREATE INDEX IF NOT EXISTS idx_topic_words_model_id ON augur_data.topic_words(model_id); 