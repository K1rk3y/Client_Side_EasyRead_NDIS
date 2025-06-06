from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from hdbscan import HDBSCAN
from math import log
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# https://github.com/MaartenGr/BERTopic/issues/286
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Resources:
# Topic modelling in general: https://medium.com/@m.nath/topic-modeling-algorithms-b7f97cec6005
# BERTopic author's words: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# Documentation: https://maartengr.github.io/BERTopic/index.html
# We have also considered LDA and normal DBSCAN. See older commits for that. Feel free to replace the current model with those!

def cluster_sentences(sentences, config={}):
    """
    Parameters:
    -----------
    sentences : list of str
        List of sentences to be clustered.
    config : dict
        Dictionary of hyperparameters for the clustering model.
        Default values are:
        - embedding_model: 'all-MiniLM-L6-v2'
        - min_cluster_size: 8
        - min_samples: 8
        - cluster_selection_epsilon: 0.0
        - cluster_selection_method: 'leaf'
        - ngram_range: (1, 2)
        - max_df: 1.0
        - min_df: 1
        - nr_topics: 20

    Returns:
    --------
    result : list of list of str
        A list of groups, where each group is a list of sentences that belong to the same topic.
    coherence : float
        The coherence score of the topics.
    """

    sentence_model = SentenceTransformer(config.get('embedding_model', 'all-MiniLM-L6-v2'))
    embeddings = normalize(sentence_model.encode(sentences))

    umap_model = UMAP(random_state=42)

    # https://www.reddit.com/r/datascience/comments/5sfj0y/hdbscan_cluster_still_unclear_to_me_how_to_chose/
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.get('min_cluster_size', 10),
        min_samples=config.get('min_samples', int(log(len(sentences)))),
        cluster_selection_epsilon=config.get('cluster_selection_epsilon', 0.0),
        cluster_selection_method=config.get('cluster_selection_method', "leaf"),
        prediction_data=True,
    )

    stop_words = stopwords.words('english')
    domain_specific_stopwords = [] # e.g. 'disability', 'service', 'intellectual'. Make sure you know what you're doing!
    stop_words.extend(domain_specific_stopwords)

    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        tokenizer=LemmaTokenizer(),
        ngram_range=config.get('ngram_range', (1, 3)),
        max_df=config.get('max_df', 1.0),
        min_df=config.get('min_df', 1),
    )

    representation_model = KeyBERTInspired()
    
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=config.get('nr_topics', 10),
    )

    topics, probs = topic_model.fit_transform(sentences, embeddings)

    # removing stopwords from topic representation
    topic_model.update_topics(sentences, vectorizer_model=vectorizer_model)

    print(topic_model.get_topic_info())

    groups = {t: [] for t in range(-1, max(topics)+1)}
    for i, sentence in enumerate(sentences):
        groups[topics[i]] = groups[topics[i]] + [sentence]
    
    # If there is only one group, it's the whole document
    if len(groups) == 1:
        result = list(groups.values())
        # coherence = calculate_coherence_score(sentences, groups, topic_model)
        return result
    
    # Remove outliers
    groups = {k: v for k, v in groups.items() if k != -1}
    
    result = list(groups.values())
    # coherence = calculate_coherence_score(sentences, groups, topic_model)
    
    return result

def calculate_coherence_score(sentences, groups, topic_model):
    """
    Coherence score tells you how well the words in a topic are related to each other. 
    0 means no relation (e.g. random words), 1 means perfect relation (e.g. synonyms).
    https://www.reddit.com/r/LanguageTechnology/comments/ap36l1/what_is_a_good_coherence_score_for_an_lda_model/
    https://stackoverflow.com/questions/54762690/evaluation-of-topic-modeling-how-to-understand-a-coherence-value-c-v-of-0-4
    """

    # Get top words for each topic
    top_words = []
    for group in groups:
        topic = topic_model.get_topic(group)
        words = [word for word, prob in topic]
        top_words.append(words)
    
    # https://stackoverflow.com/questions/70548316/gensim-coherencemodel-gives-valueerror-unable-to-interpret-topic-as-either-a-l
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    texts = [analyzer(doc) for doc in sentences]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_model = CoherenceModel(
        topics=top_words,
        corpus=corpus,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )

    return coherence_model.get_coherence()

