from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import os
import logging
import sys
import json
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("fastapi-app")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "received_data.csv"

@app.post("/receive")
async def save_summary(request: Request):
    body = await request.json()
    summary_str = body.get("summary")
    query_type = body.get("query_type", "unknown")

    parsed_summary = json.loads(summary_str)

    # 예: 파일로 저장
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "query_type": query_type,
            "summary": parsed_summary
        }, f, ensure_ascii=False, indent=2)

    logger.info("파일 저장 완료")
    return {"status": "saved"}

   

if __name__ == "__main__":
    try:
        logger.info("Starting server...")
        uvicorn.run(app, host="127.0.0.1", port=8300)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        logger.error(traceback.format_exc())