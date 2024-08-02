import PyPDF2
import csv

def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfFileReader(pdf_path)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
    return text


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
    
    text = extract_text_from_pdf(pdf_path)
    write_text_to_csv(text, csv_path)
