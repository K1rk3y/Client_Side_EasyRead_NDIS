import customtkinter as ctk
from tkinter import filedialog, messagebox
from context_generation import extract_text_from_pdf, write_string_to_file
from translation import Wrapper
from pdf_generation import replace_pdf_text
from context_clustering import group_similar_paragraphs_lda_dynamic, group_similar_paragraphs_dbscan
import threading


ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class MainApplication(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Easy Read Converter Interface")
        self.geometry("600x350")
        
        self.filename = None
        self.create_widgets()

    def create_widgets(self):
        # Text banner
        self.banner = ctk.CTkLabel(self, text="Easy Read Converter", font=("Arial", 24, "bold"))
        self.banner.pack(pady=20)

        self.sub_text = ctk.CTkLabel(self, text="By Cheng Li @2024", font=("Arial", 10))
        self.sub_text.pack(pady=(10, 0))

        # Frame for buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)

        # File selection button
        self.file_button = ctk.CTkButton(button_frame, text="Select File", command=self.select_file, width=200, height=40)
        self.file_button.pack(side="left", padx=10)

        # Submit button
        self.submit_button = ctk.CTkButton(button_frame, text="Submit", command=self.submit, width=200, height=40)
        self.submit_button.pack(side="left", padx=10)

        # Dropdown menu
        self.option_label = ctk.CTkLabel(self, text="Difficulty Options:")
        self.option_label.pack(pady=(20, 5))
        self.options = ["Default", "Easy", "Advanced"]
        self.option_menu = ctk.CTkOptionMenu(self, values=self.options)
        self.option_menu.pack()

    def select_file(self):
        self.filename = filedialog.askopenfilename(title="Select a file")
        if self.filename:
            messagebox.showinfo("File Selected", f"You selected: {self.filename}")

    def submit(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please select a file before submitting.")
            return
        
        pdf_path = self.filename
        txt_path = 'text/output.txt'
        self.text, self.locations = extract_text_from_pdf(pdf_path, ignore_small_font=False)
        write_string_to_file(" ".join(self.text), txt_path)

        print("SET SIZE: ", len(self.text))
        print("LOCATION SIZE: ", len(self.locations))
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_input, daemon=True).start()

        # Show loading window
        self.show_loading_window()

    def process_input(self):
        # grouped_para = group_similar_paragraphs_lda_dynamic(list(self.text), max_topics=9, min_topics=3, step=3)  # Need experimenting
        grouped_para = group_similar_paragraphs_dbscan(list(self.text), eps=0.2, min_samples=3, use_transformer=True)   # Need experimenting
        print("GROUP NUM: ", len(grouped_para))
        print(grouped_para)

        results = []
        for group in grouped_para:
            results.append(Wrapper(" ".join(group), "", 'ft:gpt-3.5-turbo-0125:intelife-group::A3EN89gL'))

        # replace_pdf_text(self.filename, "output_pdf.pdf", self.locations, results)

        # Update the GUI in the main thread
        self.after(0, self.show_result, "\n\n".join(results))

    def show_loading_window(self):
        self.loading_window = ctk.CTkToplevel(self)
        self.loading_window.title("Processing")
        self.loading_window.geometry("300x150")

        label = ctk.CTkLabel(self.loading_window, text="Processing, please wait...")
        label.pack(pady=20)

        progress_bar = ctk.CTkProgressBar(self.loading_window, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()

    def show_result(self, result):
        # Close the loading window
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()

        # Open the text window with the result
        TextWindow(self, result)

class TextWindow(ctk.CTkToplevel):
    def __init__(self, parent, content):
        super().__init__(parent)
        self.title("Result")
        self.geometry("600x500")

        self.text_widget = ctk.CTkTextbox(self, wrap="word")
        self.text_widget.pack(expand=True, fill="both", padx=10, pady=10)

        self.text_widget.insert("end", content)
        self.text_widget.configure(state="disabled")  # Make the text read-only

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
