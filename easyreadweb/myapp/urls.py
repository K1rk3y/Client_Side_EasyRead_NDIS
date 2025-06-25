from django.urls import path
from . import views
from .views import signup_view, process_view, logout_view

urlpatterns = [
    path('', views.login_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('process/', process_view, name='process'),
    path('logout/', logout_view, name='logout'),
]
