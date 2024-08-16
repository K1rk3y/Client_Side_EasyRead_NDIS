import fitz
import csv
import re
import os
from collections import defaultdict


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


def clean_text(text):
    """
    Cleans the text by removing special characters and extra whitespace.
    """
    # Remove special characters except basic punctuation (.,!?)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
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


def extract_text_from_pdf(pdf_path, ignore_small_font=False):
    """
    Extracts text from a PDF file, cleans it, and returns it as a list of paragraphs.
    This function eliminates headers, footers, and formatting.
    If ignore_small_font is True, it ignores text below font size 16.
    """
    doc = fitz.open(pdf_path)
    paragraphs = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        page_paragraphs = []
        current_paragraph = []

        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                for line in block["lines"]:
                    for span in line["spans"]:
                        if ignore_small_font and span["size"] < 9:
                            continue
                        cleaned_text = clean_text(span["text"].strip())
                        if cleaned_text and not is_unwanted_text(cleaned_text):
                            current_paragraph.append(cleaned_text)

            if current_paragraph:
                page_paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

        paragraphs.extend(page_paragraphs)

    return "\n\n".join(paragraphs)  # Join all paragraphs with double newline for separation


def write_text_to_csv(text, csv_path, num_columns=4):
    lines = text.splitlines()
    
    # Ensure each row in the CSV has a uniform number of columns
    rows = [lines[i:i + num_columns] for i in range(0, len(lines), num_columns)]
    
    # Write to CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            # Pad the row to ensure uniform column count
            while len(row) < num_columns:
                row.append('')
            writer.writerow(row)


if __name__ == "__main__":
    pdf_path = 'test.pdf'  # Replace with your PDF file path
    csv_path = 'output.csv'  # Replace with your desired CSV output file path
    
    text = extract_text_from_pdf(pdf_path, ignore_small_font=True)
    write_text_to_csv(text, csv_path)
