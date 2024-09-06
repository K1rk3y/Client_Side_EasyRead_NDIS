from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Args:
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def group_similar_paragraphs_lda_dynamic(paragraphs, max_topics=11, min_topics=5, step=3):
    """
    Groups similar paragraphs using Latent Dirichlet Allocation (LDA) with dynamic topic selection.
    
    Args:
    paragraphs (list): List of paragraph strings to be grouped.
    max_topics (int): Maximum number of topics to consider.
    min_topics (int): Minimum number of topics to consider.
    step (int): Step size for topic number range.
    random_state (int): Random state for reproducibility.
    
    Returns:
    list: List of grouped paragraphs, where each group is a list of paragraphs assigned to the same topic.
    """
    # Tokenize the paragraphs
    tokenized_paragraphs = [paragraph.split() for paragraph in paragraphs]
    
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(tokenized_paragraphs)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_paragraphs]
    
    # Compute coherence values
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, 
                                                            texts=tokenized_paragraphs, 
                                                            start=min_topics, limit=max_topics, step=step)
    
    # Get the model with the highest coherence score
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    optimal_num_topics = optimal_model.num_topics
    
    print(f"Optimal number of topics: {optimal_num_topics}")
    
    # Use the optimal model to get topic assignments
    topic_assignments = [optimal_model.get_document_topics(doc) for doc in corpus]
    
    # Assign each paragraph to the most probable topic
    paragraph_topics = [max(topics, key=lambda x: x[1])[0] for topics in topic_assignments]
    
    # Group paragraphs based on assigned topics
    grouped_paragraphs = [[] for _ in range(optimal_num_topics)]
    for idx, topic in enumerate(paragraph_topics):
        grouped_paragraphs[topic].append(paragraphs[idx])
    
    # Remove empty groups and sort by size (largest first)
    grouped_paragraphs = [group for group in grouped_paragraphs if group]
    grouped_paragraphs.sort(key=len, reverse=True)
    
    return grouped_paragraphs


def group_similar_paragraphs_dbscan(paragraphs, eps=0.5, min_samples=5, use_transformer=False):
    """
    Groups similar paragraphs using DBSCAN clustering based on cosine similarity of their text embeddings.
    
    Args:
    paragraphs (list): List of paragraph strings to be grouped.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    use_transformer (bool): If True, use SentenceTransformer for embeddings. If False, use TF-IDF.
    
    Returns:
    list: List of grouped paragraphs, where each group is a list of similar paragraphs.
    """
    if use_transformer:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(paragraphs)
    else:
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(paragraphs).toarray()
    
    # Compute distance matrix
    distances = pairwise_distances(embeddings, metric='cosine')
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed') # Need experimenting
    cluster_labels = dbscan.fit_predict(distances)
    
    # Group paragraphs based on cluster labels
    groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(paragraphs[i])
    
    # Sort groups by size (largest first) and remove noise points (label -1)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    return [group for label, group in sorted_groups if label != -1]
