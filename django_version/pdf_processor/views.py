from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django import forms
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import get_user_model

import sys
import os
from django.conf import settings

# 添加父目录到路径以访问 core 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

User = get_user_model()

def index(request):
    ### Home page view
    return render(request, 'pdf_processor/base.html')

### PDF upload form
class UploadForm(forms.Form):
    pdf_file = forms.FileField()

def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['pdf_file']
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            file_url = settings.MEDIA_URL + uploaded_file.name
            # 判断是 Ajax 请求还是普通表单
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'success', 'file_url': file_url})
            else:
                messages.success(request, 'File uploaded successfully!')
                return redirect('pdf_processor:upload_file')
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    else:
        form = UploadForm()
    return render(request, 'pdf_processor/upload.html', {'form': form})


def process_pdf(request):
    ### PDF processing view
    return render(request, 'pdf_processor/processing.html')

class LoginForm(forms.Form):
    username = forms.CharField(label='Username', max_length=150)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

def login_view(request):
    ### login view
    form = LoginForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                return redirect('pdf_processor:index')
            else:
                messages.error(request, 'Invalid credentials')
    return render(request, 'pdf_processor/login.html', {'form': form})

class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    class Meta:
        model = User
        fields = ['username', 'password']

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, 'Account created successfully! Please log in.')
            return redirect('pdf_processor:login_view')
    else:
        form = SignupForm()
    return render(request, 'pdf_processor/signup.html', {'form': form})

def logout_view(request):
    ### Logout view
    logout(request)
    return redirect('pdf_processor:login')

def get_progress(request):
    ### Progress view for PDF processing
    return JsonResponse({'progress': 50, 'message': 'Processing in progress'})

def download_file(request):
    ### File download view
    return render(request, 'pdf_processor/results.html')