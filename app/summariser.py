from core.context_generation import extract, preprocess
from core.context_clustering import cluster_sentences
from core.translation import translate
import os

# Essential Function
def split_into_sentences(text):
    sentences = [sentence.strip() + '.' for sentence in text[1:].split('.') if sentence.strip()][:-1]
    return sentences

# Essential Function
def summarise(pdf_path):
    extracted_text = extract(pdf_path)
    preprocessed_text = preprocess(extracted_text)
    grouped_paragraphs = cluster_sentences(preprocessed_text)
    array_2d = [split_into_sentences(translate(" ".join(group))) for group in grouped_paragraphs]
    ai_summary = [item for sublist in array_2d for item in sublist]
    
    # Save preprocessed text to app/text/output.txt
    output_dir = 'app/text'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'output.txt'), 'w') as f:
        for line in preprocessed_text:
            f.write(line + '\n')

    return ai_summary