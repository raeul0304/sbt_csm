from django.http import JsonResponse, HttpResponse
from .models import Email
from .mail_generation import generate_emails
from openai import OpenAI
import json
from .config import gpt4o_OPENAI_API_KEY
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from .serializers import EmailSerializer
from rest_framework.renderers import JSONRenderer

def hello(request):
    """기본 페이지에 hello 출력"""
    return HttpResponse("<h1>hello</h1>")

def generate_and_save_emails(request):
    """이메일 생성 API - OpenAI API를 사용해 20개 이메일 생성 후 저장"""
    emails = generate_emails()
    return JsonResponse({"message": "이메일 20개 저장 완료"})

@api_view(['GET'])
@renderer_classes([JSONRenderer]) # JSON 형식으로 응답
def get_emails(request):
    """저장된 이메일 리스트 조회 API"""
    emails = Email.objects.all().values("id", "sender", "recipient", "sent_time", "subject", "body").order_by("-sent_time")
    serializer = EmailSerializer(emails, many=True)
    return Response(json.loads(json.dumps(serializer.data, ensure_ascii=False)), content_type='application/json; charset=utf-8')
