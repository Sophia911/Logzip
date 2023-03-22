"""com URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from server import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login),
    path('introduce/', views.introduce),
    path('compress/', views.compress_page),
    path('decompress/', views.decompress_page),
    path('upload/', views.get_file),
    path('compress/start/', views.compress_log_file),
    path('compress/download/', views.download_zip),
    path('compress/template/', views.show_template),
    path('decompress/download/', views.download_dec),
    path('decompress/zipfile/', views.decompress_zip),
    path('login/', views.login)
]
