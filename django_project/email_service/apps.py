from django.apps import AppConfig


class EmailServiceConfig(AppConfig):  # ✅ ChatbotConfig → EmailServiceConfig로 수정
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'email_service'
