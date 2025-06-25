from django.urls import path
from . import views

app_name = 'pdf_processor'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload_file'),
    path('process/', views.process_pdf, name='process_pdf'),
    path('login/', views.login_view, name='login_view'),
    path('signup/', views.signup_view, name='signup_view'),
    path('logout/', views.logout_view, name='logout_view'),
    path('progress/', views.get_progress, name='get_progress'),
    path('download/', views.download_file, name='download_file'),
]