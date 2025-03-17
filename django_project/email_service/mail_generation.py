from openai import OpenAI
from email_service.config import gpt4o_OPENAI_API_KEY
import os
import django
import re
from datetime import datetime
import json


# Django 설정 로드
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_project.settings")
django.setup()

from email_service.models import Email  # Django 모델 import

# OpenAI API 클라이언트 설정
client = OpenAI(api_key=gpt4o_OPENAI_API_KEY)

# 비즈니스 이메일 생성 프롬프트
def generate_email():
    email_topics = ["협업 요청", "미팅 요청", "중간 보고", "긴급 요청", "회의록 공유", "초청", "안내", "회의 요청", "미팅 일정 조율", "프로젝트 제안"]
    topics_str = ", ".join([f"[{topic}]" for topic in email_topics])
    prompt_text = f"""
    다음 요구사항을 충족하는 비즈니스 이메일을 하나 생성해주세요.
    발신자(이름 + 이메일), 수신자 이메일, 보낸시간, 제목, 본문을 포함해주세요.  
    - 발신자 이메일은 '이름.성@랜덤도메인.com' 형식이어야 합니다.  
    - 수신자 이메일도 'kim.chulsoo@abc-corp.com' 또는 다른 도메인으로 설정하세요.
    - 메일 종류는 다음 중 하나를 선택하세요: {topics_str}
    - 이메일 제목 앞에 선택한 메일 종류를 [메일 종류] 형식으로 표시하세요.
    - 보낸시간은 YYYY-MM-DD HH:MM:SS 형식으로 생성하세요.  
    - 제목은 비즈니스 이메일 주제에 맞게 생성하세요.  
    - 본문은 4~6개의 문장으로 작성해주세요.  
    - 끝에 '감사합니다. [발신자 이름] 드림.' 형식으로 마무리해주세요. 추가적인 점을 더 붙이지 마세요.
    - 다양한 이메일 도메인(xyz-tech.com, aiht.com, qservice.com 등)을 사용해주세요.
    
    아래 JSON 형식으로 출력해주세요.  
    앞에 ```json ... ```절대 적지 말고 **반드시 순수한 JSON만 반환**하고, 추가적인 텍스트(설명, 코드블록 등)는 포함하지 마세요.  
    {{
        "sender": "홍길동 (gildong.hong@xyz-tech.com)",
        "recipient": "kim.chulsoo@abc-corp.com",
        "sent_time": "2025-02-12 09:15:32",
        "subject": "[협업 요청] AI 모델 성능 개선 프로젝트 협업 가능 여부",
        "body": "안녕하세요 김철수 매니저님,\\n현재 저희 팀에서는 AI 모델 성능 개선 프로젝트를 진행 중이며, 협업 가능성을 논의하고자 합니다. 가능하신 일정이 있으시면 회신 부탁드립니다.\\n감사합니다."
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "비즈니스 이메일 생성기"},
            {"role": "user", "content": prompt_text}
        ]
    )

    # GPT의 응답 확인 (디버깅용)
    email_data_json = response.choices[0].message.content
    print("🔍 GPT 응답 (Raw):", email_data_json)  # Debugging: GPT 응답 출력

    # Markdown 코드 블록 제거 (```json ... ```)
    clean_json = re.sub(r"```json\n|\n```", "", email_data_json).strip()

    try:
        email_data = json.loads(clean_json)  # JSON 변환
        return email_data
    except json.JSONDecodeError:
        print("❌ JSON 변환 실패: 응답이 JSON 형식이 아닙니다.")
        return None  # 변환 실패 시 None 반환

# 여러 개의 이메일 생성 함수 (views.py에서 호출됨)
def generate_emails(count=20):
    emails = []
    for _ in range(count):  # 기본값 20개의 이메일 생성
        email_data = generate_email()
        if email_data:  # JSON 변환이 성공한 경우만 추가
            emails.append(email_data)
            
        # 생성된 이메일을 PostgreSQL에 저장
        if email_data:
            Email.objects.create(
                sender=email['sender'],
                recipient=email['recipient'],
                sent_time=datetime.strptime(email['sent_time'], "%Y-%m-%d %H:%M:%S"),
                subject=email['subject'],
                body=email['body']
            )
    
    print(f"\n {len(emails)}개의 이메일이 PostgreSQL에 저장되었습니다!")
    return emails

# 스크립트로 직접 실행될 때만 아래 코드 실행
if __name__ == "__main__":
    # 20개의 이메일 생성
    emails = []
    for _ in range(20):  
        email_data = generate_email()
        if email_data:  # JSON 변환이 성공한 경우만 추가
            emails.append(email_data)

    # 생성된 이메일을 PostgreSQL에 저장
    for email in emails:
        Email.objects.create(
            sender=email['sender'],
            recipient=email['recipient'],
            sent_time=datetime.strptime(email['sent_time'], "%Y-%m-%d %H:%M:%S"),
            subject=email['subject'],
            body=email['body']
        )

    print("\n 20개의 이메일이 PostgreSQL에 저장되었습니다!")