import fitz
import re
from collections import defaultdict, Counter


def non_body_filter(block, page_rect):
    is_not_header = block['bbox'][1] > page_rect.height * 0.1  # Not in top 10% of page
    is_not_footer = block['bbox'][3] < page_rect.height * 0.9  # Not in bottom 10% of page
    is_not_side_margin = (block['bbox'][0] > page_rect.width * 0.1 and 
                          block['bbox'][2] < page_rect.width * 0.9)  # Not in left/right 10% of page
    return all([is_not_header, is_not_footer, is_not_side_margin])


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


def most_common_font_size(doc, ignore_small_font):
    font_size_counter = Counter()
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size_counter[span["size"]] += 1

    most_common_font_size = None
    for font_size, _ in font_size_counter.most_common():
        if ignore_small_font and font_size >= 12:
            most_common_font_size = font_size
            break

        if not ignore_small_font and font_size >= 10:
            most_common_font_size = font_size
            break

    # If no font size >= 12 was found, fall back to the most common font size
    if most_common_font_size is None and font_size_counter:
        most_common_font_size = font_size_counter.most_common(1)[0][0]

    return most_common_font_size


def extract_text_from_pdf(pdf_path, ignore_small_font=False):
    """
    Extracts text from a PDF file, cleans it, and returns it as a list of paragraphs.
    This function eliminates headers, footers, and formatting.
    If ignore_small_font is True, it ignores text below font size 16.
    """
    doc = fitz.open(pdf_path)
    unique_paragraphs = set()

    fsize = most_common_font_size(doc, ignore_small_font)
    print("FSIZE: ", fsize)

    body_text_locations = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        page_paragraphs = []
        current_paragraph = []

        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                if non_body_filter(block, page.rect):
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] != fsize:
                                continue
                            cleaned_text = clean_text(span["text"].strip())
                            if cleaned_text and not is_unwanted_text(cleaned_text):
                                current_paragraph.append(cleaned_text)

            if current_paragraph:
                paragraph = " ".join(current_paragraph)
                if not is_capitalized_paragraph(paragraph):
                    page_paragraphs.append(paragraph)
                    body_text_locations[paragraph] = fitz.Rect(block["bbox"])
                current_paragraph = []

        for paragraph in page_paragraphs:
            if paragraph in unique_paragraphs:
                # Remove all occurrences if a duplicate is found
                unique_paragraphs.discard(paragraph)
                body_text_locations.pop(paragraph)

            else:
                unique_paragraphs.add(paragraph)

    return unique_paragraphs, list(body_text_locations.values())


def write_string_to_file(content, file_name):
    try:
        with open(file_name, 'w') as file:
            file.write(content)
        print(f"Content successfully written to {file_name}.")
    except IOError as e:
        print(f"An error occurred: {e}")
