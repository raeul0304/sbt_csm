import streamlit as st
import requests
import openai
import json


def get_client_request():
    openai.api_key = "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"

    # client 페이지 - 메일 
    st.title("Central Power AI x RPA")
    user_input = st.text_input("가져올 메일의 개수를 입력하세요: ")
    client_request = {"app": "rpa_email", "subject": "fetch_email", "number": user_input}



    if st.button("전송"):
        try:   
            # Flask(interface) 서버로 요청 전송
            flask_response = requests.post("http://localhost:5000/interface", json=client_request)

            st.write("가져온 이메일:", flask_response.json())
        except json.JSONDecodeError:
            st.write("JSON Decode Error - llm응답이 json 형식이 아닙니다.")


if __name__ == "__main__":
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