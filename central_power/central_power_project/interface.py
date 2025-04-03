from flask import Flask, request, jsonify
import requests
import sqlite3
import streamlit as st
import datetime

app = Flask(__name__)

RPA_SERVER_URL = "http://127.0.0.1:8000"
processed_result = None

#로그 기록
def store_log_event(message):
    with open("log.txt", "a") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%H:%S")
        f.write(f"[{now}] {message}\n")

# SQLite DB 연결 및 저장
def save_param_to_db(app_name, subject, number):
    store_log_event(f"[DB] 저장할 파라미터 : app={app_name}, subject={subject}, number of emails = {number}")
    conn = sqlite3.connect("param.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS params
                 (app_name TEXT, subject TEXT, number INTEGER)''')
    c.execute("DELETE FROM params WHERE app_name=?", (app_name,))
    c.execute("INSERT INTO params VALUES (?, ?, ?)", (app_name, subject, number))
    conn.commit()
    conn.close()

def load_param_from_db(app_name):
    store_log_event(f"[DB] 파라미터 조회 요청 : app={app_name}")
    conn = sqlite3.connect("param.db")
    c = conn.cursor()
    c.execute("SELECT subject, number FROM params WHERE app_name=?", (app_name,))
    row = c.fetchone()
    conn.close()
    if row:
        store_log_event(f"[DB] 파라미터 조회 성공: subject={row[0]}, number={row[1]}")
        return {"subject": row[0], "number": row[1]}
    store_log_event(f"[DB] 해당 파라미터 없음 for app={app_name}")
    return row



@app.route("/interface", methods=["POST"])
def interface():
    data = request.json
    app_name = data.get("app")
    subject = data.get("subject")
    number_of_emails = int(data.get("number", 10))

    store_log_event(f"[interface] 요청 수신: app={app_name}, subject={subject}, number of emails={number_of_emails}")
    print(f"Received request for {app_name} with subject: {subject} and number: {number_of_emails}")

    if app_name == "rpa_email":
        store_log_event(f"[DB] 파라미터 저장 시작")
        save_param_to_db(app_name, subject, number_of_emails)
        store_log_event(f"[DB] 파라미터 저장 완료")

        store_log_event(f"[interface -> RPA] RPA 서버에 요청 전송")
        rpa_response = requests.post(f"{RPA_SERVER_URL}/rpa_email/", json={"subject": subject, "number": number_of_emails})
        store_log_event(f"[PRA -> interface] RPA 응답 수신 완료")
        return jsonify({"status": "processing", "message": "이메일 데이터를 처리 중입니다."})
    
    store_log_event(f"[interface] 알 수 없는 앱 요청으로 인한 에러 : {app_name}")
    return jsonify({"error" : "지원되지 않은 app 정보"})


@app.route("/get_param", methods=["GET"])
def get_param():
    print("get_param 호출됨 in interface!")
    store_log_event(f"[RPA -> interface] 파라미터 요청 수신")
    app_name = request.args.get("app")
    param = load_param_from_db(app_name)
    if param:
        store_log_event(f"[interface -> RPA] 파라미터 반환 : {param}")
        return jsonify(param)
    store_log_event(f"[interface] RPA에서 필요한 파라미터 없음")
    return jsonify({"error": "파라미터를 찾을 수 없습니다."}), 404


@app.route("/send_result", methods=["POST"])
def send_result():
    global processed_result
    data = request.json
    app_name = data.get("app")
    result = data.get("result")
    
    store_log_event(f"[RPA -> interface] {app_name}에서 처리 결과 수신: {result}")
    # 결과 저장
    processed_result = result
    
    return jsonify({"message": "결과 수신 완료"}), 200


@app.route("/get_result", methods=["GET"])
def get_result():
    global processed_result
    if processed_result:
        store_log_event(f"[interface -> Client] 결과 전송: {processed_result}")
        return jsonify(processed_result)
    return jsonify({"error": "아직 결과가 없습니다."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

 