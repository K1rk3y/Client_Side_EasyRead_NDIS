from django.shortcuts import render, redirect
from .forms import LoginForm, RegisterForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from django.http import FileResponse, Http404, JsonResponse
import os
import threading
import time
from core.word_generation import create_docx
from myapp.summariser import summarise
from core.generate_images import generate_images_from_prompts

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, 'Login successful!')
                return redirect('process')
            else:
                messages.error(request, 'Invalid email or password.')
        else:
            messages.error(request, 'Invalid login credentials.')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def signup_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            if User.objects.filter(username=email).exists():
                messages.error(request, 'A user with that email already exists.')
            else:
                user = User.objects.create_user(
                    username=email,
                    email=email,
                    password=form.cleaned_data['password'],
                    first_name=form.cleaned_data['name']
                )
                user.save()
                messages.success(request, 'Registration successful! Please log in.')
                return redirect('login')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RegisterForm()
    return render(request, 'signup.html', {'form': form})

def process_pdf_task(session, pdf_file_path):
    """
    后台线程任务：处理PDF，更新进度，生成图片，保存结果到session。
    """
    session['progress'] = 10
    session.save()
    time.sleep(2.5)

    session['progress'] = 20
    session.save()
    time.sleep(1)

    # 1. 调用 summarise 获取结果
    results = summarise(pdf_file_path)
    session['progress'] = 40
    session.save()
    time.sleep(1)

    # 2. 生成图片
    total_images = len(results)
    images = []
    def progress_callback(current_image, total_images):
        progress_start = 40
        progress_end = 80
        progress_range = progress_end - progress_start
        progress_increment = progress_range / total_images if total_images else 1
        session['progress'] = int(round(progress_start + (current_image * progress_increment)))
        session.save()
    if total_images > 0:
        generated_images, docx_results = generate_images_from_prompts(results, progress_callback)
        images = generated_images
    else:
        images = []
        docx_results = []
    session['progress'] = 80
    session['results'] = results
    session['images'] = images
    session['docx_results'] = docx_results if 'docx_results' in locals() else []
    session.save()
    time.sleep(1)

    # 删除上传文件
    if os.path.exists(pdf_file_path):
        os.remove(pdf_file_path)
    session['progress'] = 100
    session.save()
'''
def process_pdf_task(session, pdf_file_path):
    # Simulate progress updates and processing
    session['progress'] = 10
    session.save()
    time.sleep(2.5)
    session['progress'] = 20
    session.save()
    time.sleep(1)
    # Simulate PDF processing (replace with real logic)
    results = [f"Summary for {os.path.basename(pdf_file_path)}"]
    session['progress'] = 40
    session.save()
    time.sleep(1)
    # Simulate image generation
    total_images = len(results)
    images = []
    for i, result in enumerate(results):
        session['progress'] = 40 + int(40 * (i+1)/total_images)
        session.save()
        images.append((f"Prompt {i+1}", f"sample{i+1}.png"))
        time.sleep(1)
    session['progress'] = 80
    session['results'] = results
    session['images'] = images
    session.save()
    time.sleep(1)
    # Remove uploaded file (simulate)
    # os.remove(pdf_file_path)  # Uncomment in production
    session['progress'] = 100
    session.save()
'''
@login_required
def process_view(request):
    from .forms import PDFUploadForm
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save uploaded file
            pdf_file = request.FILES['pdf_file']
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            pdf_path = os.path.join(upload_dir, 'upload.pdf')
            with open(pdf_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
            # Start background thread for processing
            request.session['progress'] = 0
            request.session.save()
            thread = threading.Thread(target=process_pdf_task, args=(request.session, pdf_path))
            thread.start()
            return render(request, 'processing.html')
    else:
        form = PDFUploadForm()
    return render(request, 'upload.html', {'form': form})

@login_required
def progress_view(request):
    progress = request.session.get('progress', 0)
    return JsonResponse({'progress': progress})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')

@login_required
@csrf_exempt
def upload_template_view(request):
    if request.method == 'POST':
        docx_file = request.FILES.get('docx_file')
        conversion_option = request.POST.get('conversion_option')
        file_path = None
        filetype = None
        if docx_file:
            file_extension = docx_file.name.split('.')[-1].lower()
            upload_dir = os.path.join('static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, docx_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in docx_file.chunks():
                    destination.write(chunk)
            if file_extension == 'docx':
                filetype = 'DOCX'
                conversion_option = 'DOCX'
            elif file_extension == 'pdf':
                filetype = 'PDF'
                conversion_option = 'PDF'
            else:
                return render(request, 'upload_template.html', {'error': 'Unsupported file type. Only DOCX and PDF are allowed.'})
        # Store filetype and file_path in session for later steps
        request.session['filetype'] = filetype or conversion_option
        request.session['temp_file_path'] = file_path
        request.session['conversion_option'] = conversion_option
        return redirect('choose_template')
    return render(request, 'upload_template.html')

@login_required
def choose_template_view(request):
    if request.method == 'POST':
        option = request.POST.get('option', '3')
        request.session['selected_option'] = option
        filetype = request.session.get('filetype', 'PDF')
        if filetype == 'DOCX':
            request.session['docx_boxes'] = int(option)
            return redirect('/docx')
        else:
            return redirect(f'/display?template={option}')
    return render(request, 'templates.html')

@login_required
def display_view(request):
    # Get template selection from query or session
    selected_template = int(request.GET.get('template', request.session.get('selected_option', 3)))
    request.session['selected_template'] = selected_template
    # Get results and images from session (set by processing)
    results = request.session.get('results', [])
    images = request.session.get('images', [])
    # Simulate page/text box structure based on results (replace with real logic)
    total_pages = len(results) if results else 1
    current_page = int(request.GET.get('page', 1))
    if current_page < 1 or current_page > total_pages:
        current_page = 1
    # Use session to persist text box edits per page
    if 'page_text_boxes' not in request.session:
        request.session['page_text_boxes'] = {}
    page_text_boxes = request.session['page_text_boxes']
    # Initialize text boxes for this page if not present
    if str(current_page) not in page_text_boxes:
        page_text_boxes[str(current_page)] = {f'box{i}': f'Text for box {i}' for i in range(1, 4)}
        request.session['page_text_boxes'] = page_text_boxes
    text_boxes = page_text_boxes[str(current_page)]
    context = {
        'current_page': current_page,
        'total_pages': total_pages,
        'text_boxes': text_boxes,
        'selected_template': selected_template,
        'results': results,
        'images': images,
    }
    return render(request, 'display.html', context)

@login_required
def submit_view(request):
    if request.method == 'POST':
        page = int(request.POST.get('page', 1))
        # Update text boxes in session
        page_text_boxes = request.session.get('page_text_boxes', {})
        if str(page) not in page_text_boxes:
            page_text_boxes[str(page)] = {}
        for key, value in request.POST.items():
            if key.startswith('box'):
                page_text_boxes[str(page)][key] = value
        request.session['page_text_boxes'] = page_text_boxes
        return redirect(f'/display?page={page}')
    return redirect('display')

@login_required
def download_pdf_view(request):
    pdf_path = os.path.join('static', 'pdf', 'output.pdf')
    if os.path.exists(pdf_path):
        return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename='output.pdf')
    else:
        raise Http404('PDF not found')

@csrf_exempt
@login_required
def upload_file_view(request):
    if request.method == 'POST':
        file = request.FILES.get('pdf_file')
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension not in ['.pdf', '.docx']:
            return JsonResponse({'error': f'Unsupported file type: {file_extension}'}, status=400)
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        temp_filenames = ['upload.pdf', 'upload.docx']
        for filename in temp_filenames:
            file_path = os.path.join(upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        static_file_name = 'upload.pdf' if file_extension == '.pdf' else 'upload.docx'
        file_path = os.path.join(upload_dir, static_file_name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        # TODO: Add DOCX to PDF conversion if needed
        if file_extension == '.docx':
            # Call your conversion function here
            pass
        pdf_url = '/static/uploads/upload.pdf'
        return JsonResponse({'file_url': pdf_url}, status=200)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def docx_view(request):
    # Get docx_results, docx_boxes, and temp_file_path from session
    docx_results = request.session.get('results', [])
    docx_boxes = request.session.get('docx_boxes', 3)
    temp_file_path = request.session.get('temp_file_path', None)
    output_file = os.path.join('static', 'docx', 'output.docx')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Call real DOCX creation logic
    create_docx(output_file, docx_results, docx_boxes, temp_file_path)
    request.session['docx_output_file'] = output_file
    context = {
        'docx_results': docx_results,
        'docx_boxes': docx_boxes,
        'output_file': output_file,
    }
    return render(request, 'docx.html', context)
