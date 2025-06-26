from django.shortcuts import render, redirect
from .forms import LoginForm, RegisterForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout, get_user_model
import sys
import os
from django.conf import settings

# Add parent directory to path to access core module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Get the user model
User = get_user_model()

# Home page view
def home(request):
    return render(request, 'base.html')

# login view
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, 'Login successfully!')
                return redirect('myapp:process')
            else:
                messages.error(request, 'Invalid email or password.')
        else:
            messages.error(request, 'Invalid login credentials.')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

# Signup view
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

@login_required
def process_view(request):
    from .forms import PDFUploadForm
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # TODO: Add PDF processing logic here
            return render(request, 'processing.html')
    else:
        form = PDFUploadForm()
    return render(request, 'upload.html', {'form': form})

# download view
@login_required
def download_file(request):
    return render(request, 'results.html')

# Logout view
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')

# User guide view
def user_guide(request):
    return render(request, 'user_guide.html')
