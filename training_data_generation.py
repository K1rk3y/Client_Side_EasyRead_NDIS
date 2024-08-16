import fitz
import json
import re
import os
from collections import defaultdict, Counter


def pair_pdfs(directory):
    # Create a dictionary to store the pairs based on the index
    pairs = defaultdict(list)
    
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            # Split the filename to extract the index and order
            parts = filename.split('_')
            if len(parts) == 3 and parts[0] == "ER" and parts[2].split('.')[0] in {"0", "1"}:
                index = parts[1]
                order = int(parts[2].split('.')[0])
                pairs[index].append((order, filename))
    
    # Sort the files by the order (0 first, 1 second) and return the 2D list
    paired_list = [sorted(pairs[index]) for index in sorted(pairs)]
    paired_list = [[file1[1], file2[1]] for file1, file2 in paired_list]

    return paired_list


def remove_newlines(serie):
    serie = serie.replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ')
    return serie


def clean_text(text):
    """
    Cleans the text by removing special characters and extra whitespace.
    """
    # Remove special characters except basic punctuation (.,!?)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = remove_newlines(cleaned_text)
    return cleaned_text


def is_unwanted_text(line):
    """
    Determines if a line is unwanted by checking for common metadata patterns.
    """
    # Patterns that typically indicate metadata, headers, or footers
    unwanted_patterns = [
        r"owner gm", r"authoriser gm", r"uncontrolled when printed", r"page \d+ of \d+",
        r"version \d+", r"issue date", r"review date"
    ]
    for pattern in unwanted_patterns:
        if re.search(pattern, line.lower()):
            return True
    return False


def is_capitalized_paragraph(paragraph):
    words = paragraph.split()
    return all(word[0].isupper() for word in words)


def most_common_font_size(doc):
    font_size_counter = Counter()
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size_counter[span["size"]] += 1

    return font_size_counter.most_common(1)[0][0] if font_size_counter else None


def extract_text_from_pdf(pdf_path, ignore_small_font=False):
    """
    Extracts text from a PDF file, cleans it, and returns it as a list of paragraphs.
    This function eliminates headers, footers, and formatting.
    If ignore_small_font is True, it ignores text below font size 16.
    """
    doc = fitz.open(pdf_path)
    paragraphs = []
    unique_paragraphs = set()

    fsize = most_common_font_size(doc)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        page_paragraphs = []
        current_paragraph = []

        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                for line in block["lines"]:
                    for span in line["spans"]:
                        if ignore_small_font and span["size"] != fsize:
                            continue
                        cleaned_text = clean_text(span["text"].strip())
                        if cleaned_text and not is_unwanted_text(cleaned_text):
                            current_paragraph.append(cleaned_text)

            if current_paragraph:
                paragraph = " ".join(current_paragraph)
                if not is_capitalized_paragraph(paragraph):
                    page_paragraphs.append(paragraph)
                current_paragraph = []

        for paragraph in page_paragraphs:
            if paragraph in unique_paragraphs:
                # Remove all occurrences if a duplicate is found
                unique_paragraphs.discard(paragraph)
            else:
                unique_paragraphs.add(paragraph)

    # Join remaining paragraphs
    return " ".join(unique_paragraphs)


def create_jsonl(doc1, doc2, output_file):
    """
    Creates a JSONL file with the specified structure and writes it to output_file.
    """
    data = {"messages": [{"role": "system", "content": "You are a translator, your role is to translate the input text into easy read format based on the user input."},{"role": "user", "content": doc1},{"role": "assistant", "content": doc2}]}
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    # Example usage
    directory_path = 'dir'  # Replace with your directory path
    pdf_pairs = pair_pdfs(directory_path)

    for path in pdf_pairs:
        # Replace these with the paths to your PDF files
        pdf1_path = "dir/" + path[0]
        pdf2_path = "dir/" + path[1]
        
        # Extract paragraphs from both PDFs
        doc1_text = extract_text_from_pdf(pdf1_path, ignore_small_font=False)
        doc2_text = extract_text_from_pdf(pdf2_path, ignore_small_font=True)
        
        # Create the JSONL file
        output_file = "training_data.jsonl"
        create_jsonl(doc1_text, doc2_text, output_file)
        
        print(f"JSONL file created: {output_file}")


if __name__ == "__main__":
    main()
