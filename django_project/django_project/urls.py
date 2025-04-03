"""
엔드 포인트 정의
URL configuration for django_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from email_service.views import generate_and_save_emails, get_emails, hello
from email_service.rpa_email import RPA_Email

urlpatterns = [
    path("", hello, name="hello"),
    path("generate-emails/", generate_and_save_emails, name="generate_emails"),
    path("get-emails/", get_emails, name="get_emails"),
    path("rpa_email/", RPA_Email.rpa_fetch_email, name="rpa_email")
]
