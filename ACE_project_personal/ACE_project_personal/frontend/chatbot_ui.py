import streamlit as st
from csm.ACE_project.backend.llm_w_rag import RAGSystem
import os

class AIAssistant:
    def __init__(self, debug_mode=True, openai_api_key=None):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.rag_system = RAGSystem(openai_api_key=self.openai_api_key)
        self.debug_mode = debug_mode
        
    def initialize(self):
        self.rag_system.initialize()
    
    def get_ai_response(self, user_input):
        """
        사용자 입력에 대한 AI 응답을 생성하는 메인 함수
        
        Args:
            user_input: 사용자 입력 텍스트
        
        Returns:
            응답 텍스트
        """
        # 사용자 입력과 관련된 데이터 검색
        # retrieved_docs = self.rag_system.retrieve_data(user_input)
        
        # 답변 생성
        answer = self.rag_system.generate_answer(user_input)
        
        # LLM이 생성한 자연어 응답이 있는 경우 그것을 사용
        if 'natural_language_response' in answer:
            response_content = answer['natural_language_response']
        else:
            # 메시지 내용 구성 (기존 방식)
            response_content = "요청을 처리했습니다."
            
            # 옵션 값이 있으면 메시지에 포함
            if 'option' in answer:
                response_content = f"{answer['option']} 상품으로 설정했습니다."
            elif 'option2' in answer:
                response_content = f"{answer['option2']} 원재료로 설정했습니다."
            
            # 수량 변경이 있는 경우 메시지에 추가
            if 'm10change' in answer:
                change_value = float(answer['m10change'])
                change_text = "증가" if change_value > 0 else "감소"
                response_content += f" 입고 수량을 {abs(change_value)}% {change_text}시켰습니다."
            
            if 'm20change' in answer:
                change_value = float(answer['m20change'])
                change_text = "증가" if change_value > 0 else "감소"
                response_content += f" 판매 수량을 {abs(change_value)}% {change_text}시켰습니다."
            
            if 'm110change' in answer:
                change_value = float(answer['m110change'])
                change_text = "올림" if change_value > 0 else "내림"
                response_content += f" USD 환율을 {abs(change_value)}% {change_text}으로 설정했습니다."
        
        # UI 상태 업데이트
        self.update_ui_state(answer)
        
        # # 디버깅 정보 추가
        # if self.debug_mode:
        #     debug_str = generate_debug_info(debug_info)
        #     if debug_str:
        #         response_content += f"\n\n[디버깅 정보]\n{debug_str}"
        
        return response_content
    
    def update_ui_state(self, answer_dict):
        """
        답변에 기반하여 UI 상태 업데이트
        
        Args:
            answer_dict: 답변 딕셔너리
        """

        print(f"업데이트할 답변: {answer_dict}")

        # 제품 옵션 설정
        if 'option' in answer_dict:
            print(f"옵션 업데이트: {answer_dict['option']}")
            st.session_state.option = answer_dict['option']
        
        # 원재료 옵션 설정
        if 'option2' in answer_dict:
            print(f"옵션2 업데이트 : {answer_dict['option2']}")
            st.session_state.option2 = answer_dict['option2']
        
        # 도넛 그래프 상품 설정
        if 'option3' in answer_dict:
            print(f"옵션3 업데이트: {answer_dict['option3']}")
            st.session_state.option3 = answer_dict['option3']
        
        # 입고 수량 변화 설정
        if 'm10change' in answer_dict:
            print(f"입고수량 변화: {answer_dict['m10change']}")
            st.session_state.m10change = float(answer_dict['m10change'])
        
        # 판매 수량 변화 설정
        if 'm20change' in answer_dict:
            print(f"판매수량 변화: {answer_dict['m20change']}")
            st.session_state.m20change = float(answer_dict['m20change'])
        
        # USD 환율 변화 설정
        if 'm110change' in answer_dict:
            print(f"USD 환율 변화: {answer_dict['m110change']}")
            st.session_state.m110change = float(answer_dict['m110change'])
        
        # 원재료 구매단가 변화 설정
        if 'm50change2' in answer_dict:
            print(f"원재료 단가 변화: {answer_dict['m50change2']}")
            st.session_state.m50change2 = float(answer_dict['m50change2'])
        

# 대화 관련 함수들
def submit_message():
    """
    사용자 메시지 제출 및 처리
    """
    # AI 어시스턴트 초기화 (첫 실행시에만)
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = AIAssistant(debug_mode=True)
        st.session_state.ai_assistant.initialize()
    
    # 동적 key를 사용하여 현재 입력값 가져오기
    current_input_key = f"user_input_{st.session_state.input_key}"
    if current_input_key in st.session_state and st.session_state[current_input_key].strip():
        user_input = st.session_state[current_input_key]
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # RAG 기반 응답 생성
        response = st.session_state.ai_assistant.get_ai_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 입력창 초기화를 위해 key 값 변경
        st.session_state.input_key += 1
        st.rerun()

def render_chatbot():
    """
    챗봇 UI 렌더링
    """
    # 챗봇 컨테이너 생성
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        st.markdown("<h4 style='text-align: left; color: #333;'>AI chatbot</h4>", unsafe_allow_html=True)
        
        # 스타일 정의
        st.markdown("""
        <style>
        .chat-area {
            height: 200px;
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
            else:
                messages_html += f'<div class="bot-bubble"><div class="bot-message">{message["content"]}</div></div>'
        messages_html += '</div>'
        
        # 채팅 메시지 표시
        chat_placeholder.markdown(messages_html, unsafe_allow_html=True)
        
        # 입력 영역
        input_col1, input_col2 = st.columns([9.5, 3.5])
        
        with input_col1:
            st.text_input(
                "", 
                placeholder="메시지를 입력하세요...", 
                label_visibility="collapsed", 
                key=f"user_input_{st.session_state.input_key}"
            )

        with input_col2:
            if st.button("전송", key="send_button"):
                submit_message()

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