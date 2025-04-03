import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import openai
import json
import datetime


def get_client_request():
    openai.api_key = "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"

    # client 페이지 - 메일 
    st.title("Central Power AI x RPA")

    st_autorefresh(interval=1000, key="log_refresh")

    # 세션 상태 초기화
    if "response" not in st.session_state:
        st.session_state.response = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "log_initialized" not in st.session_state:
        with open("log.txt", "w") as f:
            print("시스템 로그 초기화")
        st.session_state.log_initialized = True
    if "activate_logging" not in st.session_state:
        st.session_state.activate_logging = False
    if "selected_email" not in st.session_state:
        st.session_state.selected_email = None
    

    user_input = st.text_input("가져올 메일의 개수를 입력하세요: ")
    client_request = {"app": "rpa_email", "subject": "fetch_email", "number": user_input}



    if st.button("전송"):
        try:
            st.session_state.activate_logging = True
            st.session_state.processing = True
               
            # Flask(interface) 서버로 요청 전송
            flask_response = requests.post("http://localhost:5000/interface", json=client_request)
            response_data = flask_response.json()

            # 처리 중인 경우
            if "status" in response_data and response_data["status"] == "processing":
                st.info("이메일 데이터를 처리 중입니다. 잠시만 기다려주세요...")
            else:
                # 바로 결과가 반환된 경우
                st.session_state.response = response_data
                st.session_state.processing = False

        except json.JSONDecodeError:
            st.write("JSON Decode Error - llm응답이 json 형식이 아닙니다.")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            st.session_state.processing = False

    # 처리 중이면 결과 폴링
    if st.session_state.processing and not st.session_state.response:
        try:
            result_response = requests.get("http://localhost:5001/get_result")
            if result_response.status_code == 200:
                st.session_state.response = result_response.json()
                st.session_state.processing = False
        except Exception as e:
            st.warning(f"결과 확인 중 오류: {str(e)}")    


    if st.session_state.response:
        display_result(st.session_state.response)
        # st.subheader("가져온 이메일 결과")
        # st.json(st.session_state.response)
        
    show_log()

def display_result(response_data):
    emails = response_data.get("emails", [])

    if not emails:
        st.warning("가져온 이메일이 없습니다")
        return
    
    st.markdown("### 이메일 목록")

    email_list, email_detail = st.columns([1,2])

    with email_list:
        st.markdown("### 받은 이메일")

        #이메일 목록을 카드 형태로 표시
        for idx, email in enumerate(emails):
            with st.container():
                card = f"""
                <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                    <h5 style="margin:0; color:#1E88E5;">{email.get('subject', '제목 없음')}</h5>
                    <p style="margin:0; font-size:0.8em; color:#666;">From: {email.get('sender', '발신자 불명')}</p>
                    <p style="margin:0; font-size:0.8em; color:#666;">{email.get('sent_time', '')}</p>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)

                if st.button(f"이메일 내용 보기 #{idx+1}", key=f"view_email_{idx}"):
                    st.session_state.selected_email = email
                
                # 선택한 이메일의 상세 내용 표시
    with email_detail:
        if st.session_state.selected_email:
            email = st.session_state.selected_email
            st.markdown("#### 이메일 상세 정보")
            
            # 이메일 헤더 정보 (카드 스타일)
            header_card = f"""
            <div style="background-color:#f8f9fa; border-radius:5px; padding:15px; margin-bottom:15px;">
                <h3 style="margin-top:0;">{email.get('subject', '제목 없음')}</h3>
                <table style="width:100%; border-collapse:collapse;">
                    <tr>
                        <td style="padding:5px 10px 5px 0; width:80px;"><strong>보낸 사람:</strong></td>
                        <td style="padding:5px 0;">{email.get('sender', '발신자 불명')}</td>
                    </tr>
                    <tr>
                        <td style="padding:5px 10px 5px 0;"><strong>받는 사람:</strong></td>
                        <td style="padding:5px 0;">{email.get('recipient', '수신자 불명')}</td>
                    </tr>
                    <tr>
                        <td style="padding:5px 10px 5px 0;"><strong>보낸 시간:</strong></td>
                        <td style="padding:5px 0;">{format_date(email.get('sent_time', ''))}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(header_card, unsafe_allow_html=True)
            
            # 이메일 내용
            st.markdown("#### 본문")
            
            # 본문 내용을 스크롤 가능한 컨테이너에 표시
            with st.container():
                st.markdown(f"""
                <div style="border:1px solid #e0e0e0; border-radius:5px; padding:15px; 
                            background-color:white; height:300px; overflow-y:auto;">
                    {format_email_body(email.get('body', '내용 없음'))}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("왼쪽 목록에서 이메일을 선택하면 상세 내용이 여기에 표시됩니다.")

def format_date(date_str):
    """날짜 포맷 변경"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return date_obj.strftime("%Y년 %m월 %d일 %H시 %M분")
    except:
        return date_str

def format_email_body(body):
    """이메일 내용 포맷팅"""
    # HTML 태그가 있는 경우를 대비해 안전하게 처리
    # 줄바꿈 유지하도록 처리
    formatted_body = body.replace('\n', '<br>')
    return formatted_body


def show_log():
    if st.session_state.get("activate_logging", False):
        st.subheader("실시간 인터페이스 로그")

        try:
            with open("log.txt", "r") as f:
                logs = f.readlines()
            st.code("".join(logs[-40:]), language="text") #최근 40줄
        except FileNotFoundError:
            st.warning("아직 로그 파일이 없습니다.")



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    get_client_request()


    # prompt_message = """
    # {user_input}라는 문장에서 'subject'와 'number' 값을 추출해 json 형식으로 반환하세요.
    # input 예시 : 안읽은 메일 3개를 보여줘
    # output 예시 : {"subject" : "fetch_unread_email", "number" : 3}
    # """
    # response = openai.Completion.create(
        #     model = "gpt-4",
        #     messages = [
        #         {"role" : "system", "content" : "You are an AI Agent who has to recognize entities in given sentences."},
        #         {"role" : "user", "content": prompt_message}
        #     ]
        # )

        # llm_ouput = response["choices"][0]["message"]["content"]
    #parsed_data = json.loads(user_input)