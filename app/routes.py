from flask import Blueprint, render_template, request, redirect, send_file, jsonify, url_for, flash
from app.forms import PDFUploadForm, LoginForm, SignupForm
from app.summariser import summarise
from generate_images import generate_images_from_prompts  # Import your function
from generate_images import generate_images_from_prompts
from pdf_generation import compile_info_for_pdf, update_text, caller
from word_generation import create_docx
from word2pdf import convert_to_pdf
from flask_login import login_user, logout_user, current_user, login_required
from app.models import User
from app import db

import os
import time
import threading

index_bp = Blueprint('index', __name__)

UPLOAD_PROGRESS = {"progress": 0}


@index_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index.process'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index.process'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('index.login'))  # Add redirect on failure
    
    # Add this to show form errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{field}: {error}", 'danger')
    
    return render_template('login.html', form=form)

@index_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index.process'))
    
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered', 'danger')
            return redirect(url_for('index.signup'))
        
        user = User(name=form.name.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please login', 'success')
        return redirect(url_for('index.login'))
    
    return render_template('signup.html', form=form)

@index_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index.login'))

def process_pdf(pdf_file_path):
    global UPLOAD_PROGRESS, results, generated_images, docx_results

    # Simulate progress updates
    UPLOAD_PROGRESS['progress'] = 10
    time.sleep(2.5)  # Simulate processing time

    UPLOAD_PROGRESS['progress'] = 20
    time.sleep(1)  # Simulate processing time
    
    # Process the PDF and get results
    results = summarise(pdf_file_path)

    UPLOAD_PROGRESS['progress'] = 40
    time.sleep(1)  # Simulate processing time
    
    total_images = len(results)
    
    def progress_callback(current_image, total_images):
        # Update the progress based on the current image
        progress_start = 40
        progress_end = 80
        progress_range = progress_end - progress_start
        progress_increment = progress_range / total_images
        UPLOAD_PROGRESS['progress'] = int(round(progress_start + (current_image * progress_increment)))

    # Generate images using the results as prompts
    generated_images, docx_results = generate_images_from_prompts(results, progress_callback)

    UPLOAD_PROGRESS['progress'] = 80
    time.sleep(1)  # Simulate processing time

    # Remove the uploaded PDF file after processing
    os.remove(pdf_file_path)

    UPLOAD_PROGRESS['progress'] = 100

@index_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf_file']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file and file_extension in {'.pdf', '.docx'}:
        files_dir = os.path.join(os.path.dirname(__file__), 'static/uploads')
        if not os.path.exists(files_dir):
            os.makedirs(files_dir)

        # Remove uploaded files stored on server side.
        temp_filenames = ["upload.pdf", "upload.docx"]
        for filename in temp_filenames:
            file_path = os.path.join(files_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Define a static file name and save the uploaded file to that path
        static_file_name = 'upload.pdf' if file_extension == '.pdf' else 'upload.docx'
        file_path = os.path.join(files_dir, static_file_name)
        file.save(file_path)

        # Convert DOCX to PDF if it's a DOCX file
        if file_extension == '.docx':
            # Convert the DOCX to PDF
            convert_to_pdf(file_path)

        # Return the static file path (relative or absolute as needed)
        pdf_url = "static/uploads/upload.pdf"
        return jsonify({'file_url': pdf_url}), 200

    return jsonify({'error': f"Unsupported file type: {file_extension}"}), 400


@index_bp.route('/', methods=['GET', 'POST'])
@login_required
def process():
    form = PDFUploadForm()

    if request.method == 'POST' and form.validate_on_submit():
        UPLOAD_PROGRESS['progress'] = 0  # Reset progress

        file_path = "app/static/uploads/upload.pdf"

        # Start the background thread for processing
        thread = threading.Thread(target=process_pdf, args=(file_path,))
        thread.start()

        return render_template('processing.html')  # Render a template that shows the progress bar

    return render_template('upload.html', form=form)

@index_bp.route('/progress', methods=['GET'])
@login_required
def get_progress():
    return jsonify(UPLOAD_PROGRESS)

@index_bp.route('/upload-template', methods=['GET', 'POST'])
@login_required
def upload_template():
    if request.method == 'POST':
        # Get the selected conversion option
        conversion_option = request.form.get('conversion_option')
        print(f"DEBUG: Conversion option received: {conversion_option}")  # Debug statement
        
        global filetype, temp_file_path
        temp_file_path = None
        file_path = None
        
        # Handle file upload if a file was uploaded
        uploaded_file = request.files.get('docx_file')
        if uploaded_file:
            # Determine the file type
            file_extension = uploaded_file.filename.split('.')[-1].lower()

            if file_extension == 'docx':
                conversion_option = 'DOCX'
            elif file_extension == 'pdf':
                conversion_option = 'PDF'
            else:
                return "Unsupported file type. Only DOCX and PDF are allowed.", 400

            # Save the uploaded file
            file_dir = 'app/static/uploads/'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = f'app/static/uploads/{uploaded_file.filename}'
            uploaded_file.save(file_path)
        
        # Handle the case where no file is uploaded but an option is selected 
        if conversion_option == 'DOCX':
            filetype = 'DOCX'
            temp_file_path = file_path
            
        else:
            filetype = 'PDF'
            temp_file_path = file_path
            
        return redirect('/choose-template')
    
    return render_template('upload_template.html')

@index_bp.route('/choose-template', methods=['GET', 'POST'])
@login_required
def choose_template():
    if request.method == 'POST':
        # Get the selected option from the form
        selected_option = request.form.get('option', '3')  # Default to 3 if no selection
        
        if filetype == 'DOCX':
            global docx_boxes
            docx_boxes = int(selected_option)
            return redirect('/docx')
        
        return redirect(f'/display?template={selected_option}')

    return render_template('templates.html')

@index_bp.route('/docx')
@login_required
def docx():
    print("RESULTS:", docx_results)
    print("BOXES:", docx_boxes)
    output_file = f'app/static/docx/output.docx'
    create_docx(output_file, docx_results, docx_boxes, temp_file_path)
    
    return render_template('docx.html')
    

TOTAL_PAGES = None
page_text_boxes = None
template = None

@index_bp.route('/display')
@login_required
def display():
    global TOTAL_PAGES, page_text_boxes, all_groups, mapping, template

    # Get the selected template variable from the query parameters
    temp = int(request.args.get('template', 999)) 
    if template is None or (temp != template and temp != 999):
        template = temp
    
    # Check if TOTAL_PAGES and page_text_boxes are already set
    if TOTAL_PAGES is None or page_text_boxes is None:
        TOTAL_PAGES, page_text_boxes, all_groups, mapping = compile_info_for_pdf(results, generated_images, template, temp_file_path)
    else:
        TOTAL_PAGES, page_text_boxes = caller(all_groups, template, temp_file_path)

    # Get the current page from the query parameters (default to 1)
    page = int(request.args.get('page', 1))
    if page < 1 or page > TOTAL_PAGES:
        page = 1

    # Render the requested page and its text boxes
    return render_template(
        'display.html',
        total_pages=TOTAL_PAGES,
        text_boxes=page_text_boxes.get(page, {}),
        current_page=page,
        selected_template=int(template)
    )

@index_bp.route('/submit', methods=['POST'])
@login_required
def submit():
    # Get the current page from the form (ensure it's valid)
    page = int(request.form.get('page', 1))
    if page < 1 or page > TOTAL_PAGES:
        page = 1

    # Ensure the page exists in the dictionary
    if page not in page_text_boxes:
        page_text_boxes[page] = {}

    # Update the text box content for the current page
    for key, value in request.form.items():
        if key.startswith('box'):  # Only update keys that start with 'box'
            update_text(all_groups, page_text_boxes, mapping, page, key, value)

    # Debugging: Print the updated page content to the console
    print(f"Updated Text Boxes for Page {page}:", page_text_boxes[page])
    print("All Text Boxes:", page_text_boxes)

    # Redirect back to the same page
    return redirect(f"/display?page={page}")

@index_bp.route('/download-pdf')
@login_required
def download_pdf():
    pdf_path = "static/pdf/output.pdf"  # Update this to match your PDF file's location
    return send_file(pdf_path, as_attachment=True, download_name="output.pdf")
