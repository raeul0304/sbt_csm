import streamlit as st
import os
import requests
import logging
import json
from pathlib import Path
import base64

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "chatbot.log"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler(str(log_file))  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

FASTAPI_ENDPOINT = "http://llm_engine:8787/chat" #기존 : 121.133.205.199:8787/chat

class AIAssistant:
    def __init__(self, debug_mode=True, openai_api_key=None):
        self.debug_mode = debug_mode
        self.last_parsed_output = {}
        self.fallback_mode = False # 서버 오류 발생 시 로컬 처리 모드 활성화
        
    def initialize(self):
        pass
    
    def get_ai_response(self, user_input):        
        form_data = {
            "query": user_input,
            "assistant": "ace_ez"
        }

        try:
            logger.info(f"Sending payload: {form_data}")
            logger.info(f"Sending to endpoint: {FASTAPI_ENDPOINT}")
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            
            response = requests.post(
                FASTAPI_ENDPOINT, 
                data={
                    "query": user_input,
                    "assistant": "ace_ez"
                },
                timeout=15
            )
            
            # 디버깅을 위한 응답 상태 코드 및 내용 출력
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            logger.info(f"Response content: {response.text[:500]}")  # 응답 내용 앞부분만 출력
            
            # 오류 발생 시 예외 발생
            if response.status_code != 200:
                st.error(f"FASTAPI 호출 오류 발생 : {response.status_code}")
                st.error(f"응답 내용: {response.text[:500]}")
                return f"서버 응답 오류: {response.status_code}. 자세한 내용은 콘솔 로그를 확인하세요."

            
            response_content = "요청을 처리했습니다"
            answer = response.json().get("response", {})
            output_str = answer.get("output", "")
            response_content = output_str
            reference_urls = answer.get("reference_url", "")
            titles = answer.get("titles", "")

            logger.info(f"FASTAPI 응답 내용: {answer}")
            logger.info(f"[디버깅] titles 타입: {type(titles)}, reference_urls 타입: {type(reference_urls)}")


            if output_str.startswith("<NL>"):
                output_str = output_str.replace("<NL>", "").strip()
            if isinstance(reference_urls, list) and reference_urls and isinstance(titles, list):
                logger.info("[디버깅] 레퍼런스 블록 진입 성공")

                st.session_state.last_titles = titles
                st.session_state.last_urls = reference_urls

                ref_section = "\n\n**레퍼런스:**\n"
                for title, url in zip(titles, reference_urls):
                    if title and url:
                        ref_section += f"- [{title}]({url})\n"
                response_content += ref_section
                return response_content
                        
            # 2. JSON 파싱 시도
            parsed_output = {}
            try:
                if isinstance(output_str, str):
                    parsed_output = json.loads(output_str)
                    logger.info(f"JSON 파싱 결과: {parsed_output}")
                elif isinstance(output_str, dict):
                    parsed_output = output_str
            except Exception as e:
                logger.error(f"output 파싱 오류: {e}")
                st.error("서버 파싱 오류가 발생했습니다.")
                return output_str  # 그냥 출력

            # 3. natural_language_response 있으면 사용
            if isinstance(parsed_output, dict) and "natural_language_response" in parsed_output:
                response_content = parsed_output["natural_language_response"]
                logger.info(f"자연어 응답 추출됨: {response_content[:100]}...")
            else:
                # 없으면 전체 JSON pretty 출력
                response_content = "요청을 처리했습니다."
                logger.info("natural_language_response 없음. 전체 JSON 출력")

            # UI 상태 업데이트
            update_data = parsed_output if parsed_output else self.last_parsed_output
            if update_data:
                self.last_parsed_output = update_data
                self.update_ui_state(update_data)
                logger.info("UI 상태 업데이트 완료")
            else:
                logger.warning("UI 상태 업데이트 건너뜀: 업데이트할 데이터 없음")
            
            return response_content

        except Exception as e:
            st.error(f"FASTAPI 호출 오류 발생 : {e}")
            logger.exception("예외 발생:")
            return "FASTAPI 요청을 처리하는 중 오류가 발생했습니다."
        


    def update_ui_state(self, answer_dict):
        logger.info(f"원본 응답: {answer_dict}")
        
        # 기본값으로 먼저 초기화
        st.session_state.option = None
        st.session_state.option2 = None
        st.session_state.option3 = None
        st.session_state.m10change = 0.0
        st.session_state.m20change = 0.0
        st.session_state.m110change = 0.0
        st.session_state.m50change2 = 0.0
        logger.info("기본값 초기화 완료했습니다.")
        
        # 예외 방지용 유틸 함수
        def safe_set_float(key, attr_name):
            try:
                val = answer_dict.get(key, None)
                if val is not None:
                    val = float(val)
                    setattr(st.session_state, attr_name, val)
                    logger.info(f"{attr_name} 설정됨: {val}")
            except Exception as e:
                logger.warning(f"{attr_name} 변환 실패: {val} / {e}")

        # 제품 옵션 설정
        if 'option' in answer_dict:
            logger.info(f"옵션 업데이트: {answer_dict['option']}")
            st.session_state.option = answer_dict['option']
        
        # 원재료 옵션 설정
        if 'option2' in answer_dict:
            logger.info(f"옵션2 업데이트 : {answer_dict['option2']}")
            st.session_state.option2 = answer_dict['option2']
        
        # 도넛 그래프 상품 설정
        if 'option3' in answer_dict:
            logger.info(f"옵션3 업데이트: {answer_dict['option3']}")
            st.session_state.option3 = answer_dict['option3']
        
        # 수치형 float 변환
        safe_set_float("m10change", "m10change")
        safe_set_float("m20change", "m20change")
        safe_set_float("m110change", "m110change")
        safe_set_float("m50change2", "m50change2")
            
        # # 입고 수량 변화 설정
        # if 'm10change' in answer_dict:
        #     logger.info(f"입고수량 변화: {answer_dict['m10change']}")
        #     st.session_state.m10change = float(answer_dict['m10change'])
        
        # # 판매 수량 변화 설정
        # if 'm20change' in answer_dict:
        #     logger.info(f"판매수량 변화: {answer_dict['m20change']}")
        #     st.session_state.m20change = float(answer_dict['m20change'])
        
        # # USD 환율 변화 설정
        # if 'm110change' in answer_dict:
        #     logger.info(f"USD 환율 변화: {answer_dict['m110change']}")
        #     st.session_state.m110change = float(answer_dict['m110change'])
        
        # # 원재료 구매단가 변화 설정
        # if 'm50change2' in answer_dict:
        #     logger.info(f"원재료 단가 변화: {answer_dict['m50change2']}")
        #     st.session_state.m50change2 = float(answer_dict['m50change2'])

def format_links_as_html(titles, urls):
    if not (titles and urls) or len(titles) != len(urls):
        return ""
    html = "<b>레퍼런스:</b><br>"
    for title, url in zip(titles, urls):
        if title and url:
            html += f"- <a href='{url}' target='_blank'>{title}</a></br>"
    return html  

def render_chatbot():
    """
    챗봇 UI 렌더링
    """
    initialize_session_state()
    
    # AI 어시스턴트 초기화 (첫 실행시에만)
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = AIAssistant(debug_mode=True)
        st.session_state.ai_assistant.initialize()
    
    # 챗봇 컨테이너 생성
    chat_container = st.container(height=500, border=True)
    def image_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = image_to_base64("../data/awslogo.png")
    with chat_container:
        st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <img src='data:image/png;base64,{img_base64}' width='29' style='margin-right: 6px;'/>
            <h4 style='margin: 0; color: #333;'>Bedrock</h4>
        </div>
        """, unsafe_allow_html=True)        
        # 스타일 정의
        st.markdown("""
        <style>
        .chat-area {
            height: 340px;
            width: 100%;
            overflow-y: auto;
            background-color: #f5f7f9;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .user-bubble {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #1E88E5;
            color: white;
            padding: 8px 12px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            font-size: 14px;
        }
        .bot-bubble {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 10px;
        }
        .bot-message {
            background-color: #E9ECEF;
            color: #333;
            padding: 8px 12px;
            border-radius: 18px;
            max-width: 70%;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size:14px;
        }
        .chat-area::-webkit-scrollbar {
            width: 6px;
        }
        .chat-area::-webkit-scrollbar-thumb {
            background-color: #ccc;
            border-radius: 3px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 대화 영역
        chat_placeholder = st.empty()
        
        # 채팅 메시지를 HTML로 렌더링
        messages_html = '<div class="chat-area">'
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages_html += f'<div class="user-bubble"><div class="user-message">{message["content"]}</div></div>'
            elif message["role"] == "assistant":
                if "markdown_links" in message:
                    full_content = message["content"].replace("\n", "<br>") + "<br><br>" + message["markdown_links"]
                    messages_html += f'<div class="bot-bubble"><div class="bot-message">{full_content}</div></div>'
                else:
                    messages_html += f'<div class="bot-bubble"><div class="bot-message">{message["content"]}</div></div>'
        messages_html += '</div>'
        
        # 채팅 메시지 표시
        chat_placeholder.markdown(messages_html, unsafe_allow_html=True)
        
        # 입력 영역
        if user_input := st.chat_input(placeholder="메시지를 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = st.session_state.ai_assistant.get_ai_response(user_input)
            if response:
                # 마크다운 링크 분리 처리
                if "**레퍼런스:**" in response:
                    split_text = response.split("**레퍼런스:**")
                    main_text = split_text[0].strip()
                    titles = st.session_state.get("last_titles", [])
                    urls = st.session_state.get("last_urls", [])
                    html_links = format_links_as_html(titles, urls)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": main_text,
                        "markdown_links": html_links
                    })
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# 세션 상태 초기화 함수
def initialize_session_state():
    """
    Streamlit 세션 상태 초기화
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
        
    if 'active_tab_index' not in st.session_state:
        st.session_state.active_tab_index = 0
        
    # UI 상태 관련 변수들을 모두 초기화
    if 'option' not in st.session_state:
        st.session_state.option = None
        
    if 'option2' not in st.session_state:
        st.session_state.option2 = None
        
    if 'option3' not in st.session_state:
        st.session_state.option3 = None
        
    if 'm10change' not in st.session_state:
        st.session_state.m10change = 0.0
        
    if 'm20change' not in st.session_state:
        st.session_state.m20change = 0.0
        
    if 'm110change' not in st.session_state:
        st.session_state.m110change = 0.0
        
    if 'm50change2' not in st.session_state:
        st.session_state.m50change2 = 0.0