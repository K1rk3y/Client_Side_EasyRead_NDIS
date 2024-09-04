import fitz

def replace_pdf_text(input_pdf_path, output_pdf_path, body_text_locations, new_texts):
    doc = fitz.open(input_pdf_path)
    
    text_index = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Remove existing body text
        for rect in body_text_locations:
            if isinstance(rect, tuple) and len(rect) == 4:
                rect = fitz.Rect(*rect)
            page.add_redact_annot(rect)
        page.apply_redactions()

        # Add new text
        for rect in body_text_locations:
            if text_index < len(new_texts):
                if isinstance(rect, tuple) and len(rect) == 4:
                    rect = fitz.Rect(*rect)
                page.insert_textbox(rect, new_texts[text_index], fontsize=11, fontname="helv",
                                    align=fitz.TEXT_ALIGN_LEFT)
                text_index += 1
            else:
                break  # No more new texts to add

        if text_index >= len(new_texts):
            break  # No more new texts to add
    
    # Save the modified PDF
    doc.save(output_pdf_path)
    doc.close()
