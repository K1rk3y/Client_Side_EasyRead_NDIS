from django.urls import path
from . import views
from .views import (
    signup_view, process_view, logout_view, upload_template_view, choose_template_view, display_view, submit_view, download_pdf_view,
    about_view, contact_view
)

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('process/', process_view, name='process'),
    path('logout/', logout_view, name='logout'),
    path('upload-template/', upload_template_view, name='upload_template'),
    path('choose-template/', choose_template_view, name='choose_template'),
    path('display/', display_view, name='display'),
    path('submit/', submit_view, name='submit'),
    path('download-pdf/', download_pdf_view, name='download_pdf'),
    path('progress/', views.progress_view, name='progress'),
    path('upload/', views.upload_file_view, name='upload_file'),
    path('docx/', views.docx_view, name='docx'),
    path('userguide/', views.user_guide, name='user_guide'),
    path('account/', views.account_view, name='account'),
    path('about/', about_view, name='about'),
    path('contact/', contact_view, name='contact'),
    path('easyread-intro/', views.easyread_intro, name='easyread_intro'),
    path('cancel-processing/', views.cancel_processing_view, name='cancel_processing'),
]
