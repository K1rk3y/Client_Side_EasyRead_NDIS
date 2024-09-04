import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from context_generation import extract_text_from_pdf, write_string_to_file
from translation import Wrapper
from pdf_generation import replace_pdf_text
import threading


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Easy Read Converter Interface")
        self.geometry("600x450")
        
        self.filename = None
        self.create_widgets()

    def create_widgets(self):
        # File selection button
        self.file_button = tk.Button(self, text="Select File", command=self.select_file)
        self.file_button.pack(pady=10)

        # Text box for additional instructions
        self.instruction_label = tk.Label(self, text="Input Original Text")
        self.instruction_label.pack()
        self.instruction_text = tk.Text(self, height=10, width=35)
        self.instruction_text.pack()

        # Dropdown menu
        self.option_label = tk.Label(self, text="Select Option:")
        self.option_label.pack()
        self.options = ["Blank", "Option 1", "Option 2", "Option 3"]
        self.selected_option = tk.StringVar(self)
        self.selected_option.set(self.options[0])
        self.option_menu = ttk.Combobox(self, textvariable=self.selected_option, values=self.options)
        self.option_menu.pack()

        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=20)

    def select_file(self):
        self.filename = filedialog.askopenfilename(title="Select a file")
        if self.filename:
            messagebox.showinfo("File Selected", f"You selected: {self.filename}")

    def submit(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please select a file before submitting.")
            return
        
        # Get input
        self.user_input = self.instruction_text.get("1.0", tk.END).strip()

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
        results = []
        for item in list(self.text):
            results.append(Wrapper(item, "", 'ft:gpt-3.5-turbo-0125:intelife-group::A3EN89gL'))

        replace_pdf_text(self.filename, "output_pdf.pdf", self.locations, results)

        # Update the GUI in the main thread
        self.after(0, self.show_result, "\n\n".join(results))

    def show_loading_window(self):
        self.loading_window = tk.Toplevel(self)
        self.loading_window.title("Processing")
        self.loading_window.geometry("300x100")

        label = tk.Label(self.loading_window, text="Processing, please wait...")
        label.pack(pady=20)

        progress_bar = ttk.Progressbar(self.loading_window, mode='indeterminate')
        progress_bar.pack(pady=10)
        progress_bar.start()

    def show_result(self, result):
        # Close the loading window
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()

        # Open the text window with the result
        TextWindow(self, result)

class TextWindow(tk.Toplevel):
    def __init__(self, parent, content):
        super().__init__(parent)
        self.title("Result")
        self.geometry("600x500")

        self.text_widget = tk.Text(self, wrap=tk.WORD, padx=10, pady=10)
        self.text_widget.pack(expand=True, fill=tk.BOTH)

        self.text_widget.insert(tk.END, content)
        self.text_widget.config(state=tk.DISABLED)  # Make the text read-only

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
