import pymupdf4llm
import re
import nltk

# Download the Punkt tokenizer for sentence tokenization if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Resources:
# https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
# https://ayselaydin.medium.com/1-text-preprocessing-techniques-for-nlp-37544483c007

def extract(pdf_path):
    text = pymupdf4llm.to_markdown(pdf_path)
    return text

def preprocess(text):
    nltk_tokenized_sentences = nltk.sent_tokenize(text)
    sentences = []
    for sent in nltk_tokenized_sentences:
        # two or more consecutive newline characters indicate a new paragraph, which should be a new sentence
        sent = re.split(r'\n\n+', sent)
        sentences.extend(sent)

    sentences = [subsent.strip() for sent in sentences for subsent in re.split(r'\n\n+\-', sent)]
    cleaned_sentences = [clean(sent) for sent in sentences]
    filtered_sentences = [sent for sent in cleaned_sentences if sent.strip() != ""]

    # remove duplicates
    visited = set()
    unique_sentences = []
    for sent in filtered_sentences:
        if sent not in visited:
            unique_sentences.append(sent)
            visited.add(sent)

    return unique_sentences

def clean(text):
    text = text.lower()
    # remove hashtags, urls, html tags, and lines with only dashes
    text = re.sub(r'#+\s.*|https?://\S+|www\.\S+|<.*?>|^\s*-+\s*$', '', text, flags=re.MULTILINE)
    # remove apostrophes or hyphens that are not part of a word (i.e. save contracted and hyphenated words)
    text = re.sub(r"(?<!\w)['’‑-]|['’‑-](?!\w)", "", text)
    # remove non-alphabetic characters except for newlines, spaces, apostrophes, and hyphens
    text = re.sub(r"[^\r\na-z\s'’\-]", ' ', text)
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text