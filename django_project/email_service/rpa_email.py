import os
import redis
import json
import django
import re
from django.http import JsonResponse
from .models import Email
import requests
from django.views.decorators.csrf import csrf_exempt

FLASK_INTERFACE_URL = "http://127.0.0.1:5001"

class RPA_Email:
    def __init__(self):
        pass
    
    @staticmethod
    def get_param():
        print("**get_param() 진입")
        try:
            response = requests.get(f"{FLASK_INTERFACE_URL}/get_param", params={"app": "rpa_email"})
            print("Flask 응답 상태 코드", response.status_code)
            print("Flask 응답 내용:", response.text)
            data = response.json()
        
            if "error" in data:
                print("Flask에서 에러 응답:", data)
                return None, None
            return data.get("subject"), int(data.get("number", 10))
        except Exception as e:
            print("get param 요청 실패:", str(e))
            return None, None
    
    # 필요한 정보 interface에서 fetch & 사용자 입력에 적힌 작업 수행
    @staticmethod
    @csrf_exempt
    def rpa_fetch_email(request):
        print("Starting Processing A")
        if request.method == "POST":
            try:
                subject, number = RPA_Email.get_param()
                
                if not number:
                    number = 10
                
                print(f"가져온 param number : {number}, subject: {subject}")
                emails = Email.objects.all().order_by("-sent_time")[:number]
                
                email_list = [
                    {
                        "sender": email.sender,
                        "recipient": email.recipient,
                        "sent_time": email.sent_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "subject": email.subject,
                        "body": email.body,
                    }
                    for email in emails
                ]
                print(f"email list 출력: {email_list}")
                result_data = {"emails": email_list}
                
                return RPA_Email.rpa_send_result(request, result_data)
            
            except Exception as e:
                error_msg = f"서버 오류 발생 : {str(e)}"
                print(error_msg)
                return RPA_Email.rpa_send_result(request, {"error" : error_msg})
    
    
    # 결과를 interface로 전송
    @staticmethod
    @csrf_exempt
    def rpa_send_result(request, result_data=None):
        print("Starting Processing B")

        try:
            interface_response = requests.post(
                f"{FLASK_INTERFACE_URL}/send_result",
                json={"app": "rpa_email", "result": result_data}
            )
            print(f"interface로 결과 전송 - 상태 코드 : {interface_response.status_code}")

            if interface_response.status_code == 200:
                return JsonResponse({
                    "status" : "success",
                    "message" : "결과가 성공적으로 인터페이스로 전송되었습니다."
                })
            else:
                return JsonResponse({
                    "status" : "error",
                    "message" : f"인터페이스 전송 실패: {interface_response.text}"
                },
                status = 500)
            
        except Exception as e:
            return JsonResponse(
                {"status": "error", "message": f"인터페이스 연결 오류: {str(e)}"}, 
                status=500
            ) 
        
    
