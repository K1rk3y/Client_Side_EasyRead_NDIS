from django.urls import path
from . import views
from .views import signup_view, process_view, logout_view

# Define the application name for namespacing
app_name = 'myapp'
# URL patterns for the application
urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('process/', views.process_view, name='process'),
    path('logout/', logout_view, name='logout'),
    path('download/', views.download_file, name='download_file'),
    path('userguide/', views.user_guide, name='user_guide'),
]
