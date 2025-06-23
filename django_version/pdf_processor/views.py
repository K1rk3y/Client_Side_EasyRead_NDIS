from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import sys
import os

# 添加父目录到路径以访问 core 模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def index(request):
    """主页视图"""
    return HttpResponse("<h1>Welcome to PDF Processor!</h1><p>Django version is running successfully.</p>")

def upload_file(request):
    """文件上传视图"""
    if request.method == 'POST':
        return JsonResponse({'status': 'success', 'message': 'File uploaded'})
    return HttpResponse("<h2>Upload File</h2><p>Upload your PDF or DOCX files here.</p>")

def process_pdf(request):
    """PDF 处理视图"""
    return JsonResponse({'status': 'processing', 'message': 'PDF is being processed'})

def login_view(request):
    """登录视图"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('pdf_processor:index')
        else:
            messages.error(request, 'Invalid credentials')
    return HttpResponse("<h2>Login</h2><p>Please login to continue.</p>")

def signup_view(request):
    """注册视图"""
    return HttpResponse("<h2>Sign Up</h2><p>Create your account here.</p>")

def logout_view(request):
    """登出视图"""
    logout(request)
    return redirect('pdf_processor:login')

def get_progress(request):
    """获取处理进度"""
    return JsonResponse({'progress': 50, 'message': 'Processing in progress'})

def download_file(request):
    """文件下载"""
    return HttpResponse("<h2>Download</h2><p>Your processed file is ready for download.</p>")