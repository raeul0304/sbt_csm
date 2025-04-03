from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 추가
import logging
import torch

import config
from langchain_core.messages import SystemMessage as LangChainSystemMessage, HumanMessage as LangChainHumanMessage
from typing import Optional
import json
#agents
from llm_ace import run_agent

logging.basicConfig(level='DEBUG', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)


app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처를 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용
    allow_headers=["*"],  # 모든 헤더를 허용
)


@app.post("/chat") #121.133.205.199:14878/chat
def text_inference(
    query: str = Form(...),
    assistant: str = Form("general"),
    file: Optional[UploadFile] = File(None),
    params: Optional[str] = Form(None)
):
    logger.info(f"Received params: query={query}, assistant={assistant}")
    # inference_llama 모듈을 사용하여 주어진 파라미터로 텍스트 생성 작업을 수행
    try:
        if assistant == 'ace_ez':
            response = run_agent(query)
            print(response) #response 함수 넣기! 
        else:
            raise HTTPException(status_code=400, detail="Invalid assistant type specified.")
        print(json.dumps({"response": response}, ensure_ascii=False, indent=2))
        return {"response": response}
    except HTTPException as e:
        logger.error(f"Error occurred: {e.detail}")
        raise e 
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in 'params'")
        raise HTTPException(status_code=400, detail="Invalid JSON format in 'params'")
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU memory exhausted")
        raise HTTPException(status_code=500, detail="GPU out of memory, please reduce batch size or sequence length")
    finally:
        # 추론 후 GPU 메모리 정리
        torch.cuda.empty_cache()