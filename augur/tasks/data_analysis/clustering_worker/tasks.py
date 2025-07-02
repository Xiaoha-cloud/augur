import logging
import os
import time
import traceback
import re
import pickle
import uuid
import datetime
from pathlib import Path
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sqlalchemy as s
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation  as LDA
from collections import OrderedDict
from textblob import TextBlob
from collections import Counter

from augur.tasks.init.celery_app import celery_app as celery
from augur.application.db.lib import get_value, get_session, get_repo_by_repo_git
from augur.application.db.models import RepoClusterMessage, RepoTopic, TopicWord, TopicModelMeta
from augur.tasks.init.celery_app import AugurMlRepoCollectionTask


MODEL_FILE_NAME = "kmeans_repo_messages"
stemmer = nltk.stem.snowball.SnowballStemmer("english")


@celery.task(base=AugurMlRepoCollectionTask, bind=True)
def clustering_task(self, repo_git):

    logger = logging.getLogger(clustering_model.__name__)
    engine = self.app.engine

    clustering_model(repo_git, logger, engine)

def clustering_model(repo_git: str,logger,engine) -> None:

    logger.info(f"Starting clustering analysis for {repo_git}")

    ngram_range = (1, 4)
    clustering_by_content = True
    clustering_by_mechanism = False

    # define topic modeling specific parameters
    num_topics = 8
    num_words_per_topic = 12

    tool_source = 'Clustering Worker'
    tool_version = '0.2.0'
    data_source = 'Augur Collected Messages'

    repo_id = get_repo_by_repo_git(repo_git).repo_id

    num_clusters = get_value("Clustering_Task", 'num_clusters')
    max_df = get_value("Clustering_Task", 'max_df')
    max_features = get_value("Clustering_Task", 'max_features')
    min_df = get_value("Clustering_Task", 'min_df')

    logger.info(f"Min df: {min_df}. Max df: {max_df}")

    logger.info("If you did not install NLTK libraries when you installed Augur, this will fail. ")
    #nltk.download('all')

    logger.info(f"Getting repo messages for repo_id: {repo_id}")
    get_messages_for_repo_sql = s.sql.text(
        """
            SELECT
                r.repo_group_id,
                r.repo_id,
                r.repo_git,
                r.repo_name,
                i.issue_id thread_id,
                M.msg_text,
                i.issue_title thread_title,
                M.msg_id 
            FROM
                augur_data.repo r,
                augur_data.issues i,
                augur_data.message M,
                augur_data.issue_message_ref imr 
            WHERE
                r.repo_id = i.repo_id 
                AND imr.issue_id = i.issue_id 
                AND imr.msg_id = M.msg_id 
                AND r.repo_id = :repo_id 
            UNION
            SELECT
                r.repo_group_id,
                r.repo_id,
                        r.repo_git,
                r.repo_name,
                pr.pull_request_id thread_id,
                M.msg_text,
                pr.pr_src_title thread_title,
                M.msg_id 
            FROM
                augur_data.repo r,
                augur_data.pull_requests pr,
                augur_data.message M,
                augur_data.pull_request_message_ref prmr 
            WHERE
                r.repo_id = pr.repo_id 
                AND prmr.pull_request_id = pr.pull_request_id 
                AND prmr.msg_id = M.msg_id 
                AND r.repo_id = :repo_id
            """
    )
    # result = db.execute(delete_points_SQL, repo_id=repo_id, min_date=min_date)

    with engine.connect() as conn:
        msg_df_cur_repo = pd.read_sql(get_messages_for_repo_sql, conn, params={"repo_id": repo_id})
    logger.info(msg_df_cur_repo.head())
    logger.debug(f"Repo message df size: {len(msg_df_cur_repo.index)}")

    # check if dumped pickle file exists, if exists no need to train the model
    if not os.path.exists(MODEL_FILE_NAME):
        logger.info("clustering model not trained. Training the model.........")
        train_model(logger, engine, max_df, min_df, max_features, ngram_range, num_clusters, num_topics, num_words_per_topic, tool_source, tool_version, data_source)
    else:
        model_stats = os.stat(MODEL_FILE_NAME)
        model_age = (time.time() - model_stats.st_mtime)
        # if the model is more than month old, retrain it.
        logger.debug(f'model age is: {model_age}')
        if model_age > 2000000:
            logger.info("clustering model to old. Retraining the model.........")
            train_model(logger, engine, max_df, min_df, max_features, ngram_range, num_clusters, num_topics, num_words_per_topic, tool_source, tool_version, data_source)
        else:
            logger.info("using pre-trained clustering model....")

    with open("kmeans_repo_messages", 'rb') as model_file:
        kmeans_model = pickle.load(model_file)

    msg_df = msg_df_cur_repo.groupby('repo_id')['msg_text'].apply(','.join).reset_index()

    logger.debug(f'messages being clustered: {msg_df}')

    if msg_df.empty:
        logger.info("not enough data for prediction")
        # self.register_task_completion(task, repo_id, 'clustering')
        return

    vocabulary = pickle.load(open("vocabulary", "rb"))

    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features,
                                       min_df=min_df, stop_words='english',
                                       use_idf=True, tokenizer=preprocess_and_tokenize,
                                       ngram_range=ngram_range, vocabulary=vocabulary)
    tfidf_transformer = tfidf_vectorizer.fit(
        msg_df['msg_text'])  # might be fitting twice, might have been used in training

    # save new vocabulary ??
    feature_matrix_cur_repo = tfidf_transformer.transform(msg_df['msg_text'])

    prediction = kmeans_model.predict(feature_matrix_cur_repo)
    logger.info("prediction: " + str(prediction[0]))

    with get_session() as session:

        # inserting data
        record = {
            'repo_id': int(repo_id),
            'cluster_content': int(prediction[0]),
            'cluster_mechanism': -1,
            'tool_source': tool_source,
            'tool_version': tool_version,
            'data_source': data_source
        }
        repo_cluster_messages_obj = RepoClusterMessage(**record)
        session.add(repo_cluster_messages_obj)
        session.commit()

    # result = db.execute(repo_cluster_messages_table.insert().values(record))
    logging.info(
        "Primary key inserted into the repo_cluster_messages table: {}".format(repo_cluster_messages_obj.msg_cluster_id))
    try:
        logger.debug('pickling')
        lda_model = pickle.load(open("lda_model", "rb"))
        logger.debug('loading vocab')
        vocabulary = pickle.load(open("vocabulary_count", "rb"))
        logger.debug('count vectorizing vocab')
        count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_features, min_df=min_df,
                                           stop_words="english", tokenizer=preprocess_and_tokenize,
                                           vocabulary=vocabulary)
        logger.debug('count transforming vocab')
        count_transformer = count_vectorizer.fit(
            msg_df['msg_text'])  # might be fitting twice, might have been used in training

        # save new vocabulary ??
        logger.debug('count matric cur repo vocab')
        count_matrix_cur_repo = count_transformer.transform(msg_df['msg_text'])
        logger.debug('prediction vocab')
        prediction = lda_model.transform(count_matrix_cur_repo)

        with get_session() as session:
            # Get the latest model_id from topic_model_meta
            latest_model = session.query(TopicModelMeta).order_by(TopicModelMeta.training_end_time.desc()).first()
            if not latest_model:
                logger.error("No topic model found in database")
                return

            model_id = latest_model.model_id
            logger.debug('for loop for vocab')
            for i, prob_vector in enumerate(prediction):
                for i, prob in enumerate(prob_vector):
                    record = {
                        'repo_id': int(repo_id),
                        'topic_id': i + 1,
                        'topic_prob': prob,
                        'model_id': model_id,  # Add model_id
                        'tool_source': tool_source,
                        'tool_version': tool_version,
                        'data_source': data_source
                    }

                    repo_topic_object = RepoTopic(**record)
                    session.add(repo_topic_object)
                    session.commit()

    except Exception as e:
        logger.debug(f'error is: {e}.')
        stacker = traceback.format_exc()
        logger.debug(f"\n\n{stacker}\n\n")
        pass

    # self.register_task_completion(task, repo_id, 'clustering')


def get_tf_idf_matrix(text_list, max_df, max_features, min_df, ngram_range, logger):

    logger.debug("Getting the tf idf matrix from function")
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features,
                                       min_df=min_df, stop_words='english',
                                       use_idf=True, tokenizer=preprocess_and_tokenize,
                                       ngram_range=ngram_range)
    tfidf_transformer = tfidf_vectorizer.fit(text_list)
    tfidf_matrix = tfidf_transformer.transform(text_list)
    pickle.dump(tfidf_transformer.vocabulary_, open("vocabulary", 'wb'))
    return tfidf_matrix, tfidf_vectorizer.get_feature_names()

def cluster_and_label(feature_matrix, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(feature_matrix)
    pickle.dump(kmeans_model, open("kmeans_repo_messages", 'wb'))
    return kmeans_model.labels_.tolist()

def count_func(msg):
    blobed = TextBlob(msg)
    counts = Counter(tag for word, tag in blobed.tags if
                     tag not in ['NNPS', 'RBS', 'SYM', 'WP$', 'LS', 'POS', 'RP', 'RBR', 'JJS', 'UH', 'FW', 'PDT'])
    total = sum(counts.values())
    normalized_count = {key: value / total for key, value in counts.items()}
    return normalized_count

def preprocess_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[@]\w+', '', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)

    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if len(token) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def train_model(logger, engine, max_df, min_df, max_features, ngram_range, num_clusters, num_topics, num_words_per_topic, tool_source, tool_version, data_source):
    # Generate unique model ID and record start time
    model_id = uuid.uuid4()
    training_start_time = datetime.datetime.now()
    
    # Create model and artifacts directories
    model_dir = Path("artifacts") / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Store model file paths
    model_files = {
        "lda_model": str(model_dir / "lda_model.pkl"),
        "vocabulary": str(model_dir / "vocabulary.pkl"),
        "vocabulary_count": str(model_dir / "vocabulary_count.pkl"),
        "kmeans_model": str(model_dir / "kmeans_repo_messages.pkl"),
        "html": str(model_dir / f"topic_vis_{model_id}.html"),
        "wordclouds": []
    }

    def visualize_labels_PCA(features, labels, annotations, num_components, title):
        labels_color_map = {-1: "red"}
        for label in labels:
            labels_color_map[label] = [list([x / 255.0 for x in list(np.random.choice(range(256), size=3))])]
        low_dim_data = PCA(n_components=num_components).fit_transform(features)

        fig, ax = plt.subplots(figsize=(20, 10))

        for i, data in enumerate(low_dim_data):
            pca_comp_1, pca_comp_2 = data
            color = labels_color_map[labels[i]]
            ax.scatter(pca_comp_1, pca_comp_2, c=color, label=labels[i])
        # ax.annotate(annotations[i],(pca_comp_1, pca_comp_2))

        handles, labels = ax.get_legend_handles_labels()
        handles_label_dict = OrderedDict(zip(labels, handles))
        ax.legend(handles_label_dict.values(), handles_label_dict.keys())

        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        # plt.show()
        filename = str(model_dir / f"{labels}_PCA.png")
        plt.savefig(filename)
        return filename

    get_messages_sql = s.sql.text(
        """
        SELECT r.repo_group_id, r.repo_id, r.repo_git, r.repo_name, i.issue_id thread_id,m.msg_text,i.issue_title thread_title,m.msg_id
        FROM augur_data.repo r, augur_data.issues i,
        augur_data.message m, augur_data.issue_message_ref imr
        WHERE r.repo_id=i.repo_id
        AND imr.issue_id=i.issue_id
        AND imr.msg_id=m.msg_id
        UNION
        SELECT r.repo_group_id, r.repo_id, r.repo_git, r.repo_name, pr.pull_request_id thread_id,m.msg_text,pr.pr_src_title thread_title,m.msg_id
        FROM augur_data.repo r, augur_data.pull_requests pr,
        augur_data.message m, augur_data.pull_request_message_ref prmr
        WHERE r.repo_id=pr.repo_id
        AND prmr.pull_request_id=pr.pull_request_id
        AND prmr.msg_id=m.msg_id
        """
    )

    with engine.connect() as conn:
        msg_df_all = pd.read_sql(get_messages_sql, conn, params={})

    # select only highly active repos
    logger.debug("Selecting highly active repos")
    msg_df_all = msg_df_all.groupby("repo_id").filter(lambda x: len(x) > 500)

    # combining all the messages in a repository to form a single doc
    logger.debug("Combining messages in repo to form single doc")
    msg_df = msg_df_all.groupby('repo_id')['msg_text'].apply(','.join)
    msg_df = msg_df.reset_index()

    # dataframe summarizing total message count in a repositoryde
    logger.debug("Summarizing total message count in a repo")
    message_desc_df = msg_df_all[["repo_id", "repo_git", "repo_name", "msg_id"]].groupby(
        ["repo_id", "repo_git", "repo_name"]).agg('count').reset_index()
    message_desc_df.columns = ["repo_id", "repo_git", "repo_name", "message_count"]
    logger.info(msg_df.head())

    tfidf_matrix, features = get_tf_idf_matrix(msg_df['msg_text'], max_df, max_features, min_df,
                                                    ngram_range, logger)
    msg_df['cluster'] = cluster_and_label(tfidf_matrix, num_clusters)

    # LDA - Topic Modeling
    logger.debug("Calling CountVectorizer in train model function")
    count_vectorizer = CountVectorizer(max_df=max_df, max_features=max_features, min_df=min_df,
                                       stop_words="english", tokenizer=preprocess_and_tokenize)

    # count_matrix = count_vectorizer.fit_transform(msg_df['msg_text'])
    count_transformer = count_vectorizer.fit(msg_df['msg_text'])
    count_matrix = count_transformer.transform(msg_df['msg_text'])
    pickle.dump(count_transformer.vocabulary_, open(model_files["vocabulary_count"], 'wb'))
    feature_names = count_vectorizer.get_feature_names()

    logger.debug("Calling LDA")
    lda_model = LDA(n_components=num_topics)
    lda_model.fit(count_matrix)
    
    # Generate and save pyLDAvis visualization
    logger.info("Generating pyLDAvis visualization...")
    vocab = count_vectorizer.get_feature_names_out()
    term_frequency = np.array(count_matrix.sum(axis=0)).flatten()
    vis_data = pyLDAvis.prepare(
        topic_model=lda_model,
        dtm=count_matrix,
        vectorizer=count_vectorizer,
        vocab=vocab,
        term_frequency=term_frequency
    )
    pyLDAvis.save_html(vis_data, model_files["html"])
    
    # Generate and save wordclouds for each topic
    logger.info("Generating wordclouds...")
    for topic_id in range(num_topics):
        # Get top words and their weights for this topic
        top_words_idx = lda_model.components_[topic_id].argsort()[:-num_words_per_topic-1:-1]
        word_weights = {feature_names[i]: lda_model.components_[topic_id][i] 
                       for i in top_words_idx}
        
        # Generate wordcloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=num_words_per_topic)
        wordcloud.generate_from_frequencies(word_weights)
        
        # Save wordcloud
        wordcloud_path = model_dir / f"wordcloud_{model_id}_{topic_id}.png"
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path)
        plt.close()
        
        model_files["wordclouds"].append(str(wordcloud_path))
    
    # Calculate model metrics
    coherence_score = calculate_coherence_score(lda_model, count_matrix, feature_names)
    perplexity_score = lda_model.perplexity(count_matrix)
    
    # Record training end time
    training_end_time = datetime.datetime.now()
    
    # Save models to versioned directory
    pickle.dump(lda_model, open(model_files["lda_model"], 'wb'))
    pickle.dump(count_transformer.vocabulary_, open(model_files["vocabulary_count"], 'wb'))
    
    # Store model metadata
    model_meta = {
        'model_id': model_id,
        'model_method': 'LDA',
        'num_topics': num_topics,
        'num_words_per_topic': num_words_per_topic,
        'training_parameters': {
            'max_df': max_df,
            'min_df': min_df,
            'max_features': max_features,
            'ngram_range': ngram_range,
            'num_clusters': num_clusters
        },
        'model_file_paths': model_files,
        'coherence_score': coherence_score,
        'perplexity_score': perplexity_score,
        'training_start_time': training_start_time,
        'training_end_time': training_end_time,
        'tool_source': tool_source,
        'tool_version': tool_version,
        'data_source': data_source
    }
    
    # Save metadata to database
    with get_session() as session:
        topic_model_meta = TopicModelMeta(**model_meta)
        session.add(topic_model_meta)
        session.commit()
        logger.info(f"Saved model metadata with ID: {model_id}")

    ## Advance Sequence SQL

    # key_sequence_words_sql = s.sql.text(
    #                           """
    #       SELECT nextval('augur_data.topic_words_topic_words_id_seq'::text)
    #       """
    #                               )

    # twid = self.db.execute(key_sequence_words_sql)
    # logger.info("twid variable is: {}".format(twid))
    # insert topic list into database

    with get_session() as session:
        topic_id = 1
        for topic in lda_model.components_:
            for i in topic.argsort()[:-num_words_per_topic - 1:-1]:
                record = {
                    'topic_id': int(topic_id),
                    'word': feature_names[i],
                    'model_id': model_id,
                    'tool_source': tool_source,
                    'tool_version': tool_version,
                    'data_source': data_source
                }

                topic_word_obj = TopicWord(**record)
                session.add(topic_word_obj)
                session.commit()
                logger.info(
                    "Primary key inserted into the topic_words table: {}".format(topic_word_obj.topic_words_id))
            topic_id += 1

    # insert topic list into database

    # save the model and predict on each repo separately

    logger.debug(f'entering prediction in model training, count matric is {count_matrix}')
    prediction = lda_model.transform(count_matrix)

    topic_model_dict_list = []
    logger.debug('entering for loop in model training. ')
    for i, prob_vector in enumerate(prediction):
        topic_model_dict = {}
        topic_model_dict['repo_id'] = msg_df.loc[i]['repo_id']
        for i, prob in enumerate(prob_vector):
            topic_model_dict["topic" + str(i + 1)] = prob
        topic_model_dict_list.append(topic_model_dict)
    logger.debug('creating topic model data frame.')
    topic_model_df = pd.DataFrame(topic_model_dict_list)

    result_content_df = topic_model_df.set_index('repo_id').join(message_desc_df.set_index('repo_id')).join(
        msg_df.set_index('repo_id'))
    result_content_df = result_content_df.reset_index()
    logger.info(result_content_df)
    try:
        POS_count_dict = msg_df.apply(lambda row: count_func(row['msg_text']), axis=1)
        logger.debug('POS_count_dict has no exceptions.')
    except Exception as e:
        logger.debug(f'POS_count_dict error is: {e}.')
        stacker = traceback.format_exc()
        logger.debug(f"\n\n{stacker}\n\n")
        pass
    try:
        msg_df_aug = pd.concat([msg_df, pd.DataFrame.from_records(POS_count_dict)], axis=1)
        logger.info(f'msg_df_aug worked: {msg_df_aug}')
    except Exception as e:
        logger.debug(f'msg_df_aug error is: {e}.')
        stacker = traceback.format_exc()
        logger.debug(f"\n\n{stacker}\n\n")
        pass

    visualize_labels_PCA(tfidf_matrix.todense(), msg_df['cluster'], msg_df['repo_id'], 2, "tex!")

# visualize_labels_PCA(tfidf_matrix.todense(), msg_df['cluster'], msg_df['repo_id'], 2, "MIN_DF={} and MAX_DF={} and NGRAM_RANGE={}".format(MIN_DF, MAX_DF, NGRAM_RANGE))

def calculate_coherence_score(model, count_matrix, feature_names):
    """Calculate topic coherence score using normalized pointwise mutual information (NPMI)"""
    # This is a simplified implementation - you may want to use a more sophisticated metric
    coherence = 0
    n_topics = len(model.components_)
    
    for topic_idx in range(n_topics):
        # Get top words for this topic
        top_words_idx = model.components_[topic_idx].argsort()[:-10-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        # Calculate average NPMI between top words
        topic_coherence = 0
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                # Simplified NPMI calculation
                word1, word2 = top_words[i], top_words[j]
                p1 = np.mean(count_matrix[:, feature_names.index(word1)].toarray())
                p2 = np.mean(count_matrix[:, feature_names.index(word2)].toarray())
                p12 = np.mean((count_matrix[:, feature_names.index(word1)].toarray() > 0) & 
                            (count_matrix[:, feature_names.index(word2)].toarray() > 0))
                
                if p12 > 0:
                    npmi = np.log(p12 / (p1 * p2)) / -np.log(p12)
                    topic_coherence += npmi
                    
        coherence += topic_coherence / (len(top_words) * (len(top_words)-1) / 2)
    
    return coherence / n_topics

# Add helper function for inserting topic data
def insert_topic_data(session, model_id, repo_id, topic_id, topic_prob, tool_source, tool_version, data_source):
    """
    Helper function to insert topic data into repo_topic table
    
    Args:
        session: SQLAlchemy session
        model_id: UUID of the topic model
        repo_id: Repository ID
        topic_id: Topic ID
        topic_prob: Topic probability
        tool_source: Source of the tool
        tool_version: Version of the tool
        data_source: Source of the data
    """
    record = {
        'repo_id': int(repo_id),
        'topic_id': topic_id,
        'topic_prob': topic_prob,
        'model_id': model_id,
        'tool_source': tool_source,
        'tool_version': tool_version,
        'data_source': data_source
    }
    
    repo_topic_object = RepoTopic(**record)
    session.add(repo_topic_object)
    session.commit()
    return repo_topic_object

def insert_topic_word(session, model_id, topic_id, word, tool_source, tool_version, data_source):
    """
    Helper function to insert topic word data into topic_words table
    
    Args:
        session: SQLAlchemy session
        model_id: UUID of the topic model
        topic_id: Topic ID
        word: Word associated with the topic
        tool_source: Source of the tool
        tool_version: Version of the tool
        data_source: Source of the data
    """
    record = {
        'topic_id': topic_id,
        'word': word,
        'model_id': model_id,
        'tool_source': tool_source,
        'tool_version': tool_version,
        'data_source': data_source
    }
    
    topic_word_obj = TopicWord(**record)
    session.add(topic_word_obj)
    session.commit()
    return topic_word_obj

# Add helper functions for model comparison and visualization
def get_latest_model_metadata():
    """Get metadata for the most recent model"""
    with get_session() as session:
        latest_model = session.query(TopicModelMeta).order_by(
            TopicModelMeta.training_end_time.desc()
        ).first()
        return latest_model

def get_model_topic_distributions(model_id):
    """Get topic distributions for a specific model"""
    with get_session() as session:
        distributions = session.query(RepoTopic).filter(
            RepoTopic.model_id == model_id
        ).all()
        return distributions

def compare_models(model_id1, model_id2):
    """Compare two models based on keyword overlap and coherence scores"""
    with get_session() as session:
        # Get model metadata
        model1 = session.query(TopicModelMeta).filter(
            TopicModelMeta.model_id == model_id1
        ).first()
        model2 = session.query(TopicModelMeta).filter(
            TopicModelMeta.model_id == model_id2
        ).first()
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
            
        # Get topic words for both models
        words1 = session.query(TopicWord).filter(
            TopicWord.model_id == model_id1
        ).all()
        words2 = session.query(TopicWord).filter(
            TopicWord.model_id == model_id2
        ).all()
        
        # Calculate keyword overlap
        words1_set = {w.word for w in words1}
        words2_set = {w.word for w in words2}
        overlap = len(words1_set.intersection(words2_set)) / len(words1_set.union(words2_set))
        
        return {
            'model1_coherence': model1.coherence_score,
            'model2_coherence': model2.coherence_score,
            'coherence_diff': model1.coherence_score - model2.coherence_score,
            'keyword_overlap': overlap,
            'model1_topics': model1.num_topics,
            'model2_topics': model2.num_topics
        }

def display_model_visualization(model_id):
    """Display pyLDAvis visualization for a specific model"""
    with get_session() as session:
        model = session.query(TopicModelMeta).filter(
            TopicModelMeta.model_id == model_id
        ).first()
        
        if not model:
            raise ValueError("Model not found")
            
        html_path = model.model_file_paths['html']
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"Visualization file not found: {html_path}")
            
        # For Jupyter notebook
        from IPython.display import IFrame
        return IFrame(html_path, width=1000, height=600)




