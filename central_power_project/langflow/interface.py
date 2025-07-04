from flask import Flask, request, jsonify
import requests
import sqlite3

app = Flask(__name__)

RPA_SERVER_URL = "http://127.0.0.1:8000"

# SQLite DB 연결 및 저장
def save_param_to_db(app_name, subject, number):
    conn = sqlite3.connect("param.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS params
                 (app_name TEXT, subject TEXT, number INTEGER)''')
    c.execute("DELETE FROM params WHERE app_name=?", (app_name,))
    c.execute("INSERT INTO params VALUES (?, ?, ?)", (app_name, subject, number))
    conn.commit()
    conn.close()

def load_param_from_db(app_name):
    conn = sqlite3.connect("param.db")
    c = conn.cursor()
    c.execute("SELECT subject, number FROM params WHERE app_name=?", (app_name,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"subject": row[0], "number": row[1]}
    return row



@app.route("/interface", methods=["POST"])
def interface():
    data = request.json
    app_name = data.get("app")
    subject = data.get("subject")
    number_of_emails = int(data.get("number", 10))
    print(f"Received request for {app_name} with subject: {subject} and number: {number_of_emails}")

    if app_name == "rpa_email":
        save_param_to_db(app_name, subject, number_of_emails)

        rpa_response = requests.post(f"{RPA_SERVER_URL}/rpa_email/", json={"subject": subject, "number": number_of_emails})
        return jsonify(rpa_response.json())
    
    return jsonify({"error" : "지원되지 않은 app 정보"})


@app.route("/get_param", methods=["GET"])
def get_param():
    print("get_param 호출됨 in interface!")
    app_name = request.args.get("app")
    param = load_param_from_db(app_name)
    if param:
        return jsonify(param)
    return jsonify({"error": "파라미터를 찾을 수 없습니다."}), 404


@app.route("/receive_result", methods=["POST"])
def receive_result():
    global processed_result
    data = request.json
    processed_result = data  # ✅ 결과 저장
    return jsonify({"message": "결과 수신 완료"}), 200

@app.route("/get_result", methods=["GET"])
def get_result():
    global processed_result
    if processed_result:
        return jsonify(processed_result)
    return jsonify({"error": "아직 결과가 없습니다."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

 