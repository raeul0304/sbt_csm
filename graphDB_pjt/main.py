from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 추가
import logging
import uvicorn
import torch
import config
from langchain_core.messages import SystemMessage as LangChainSystemMessage, HumanMessage as LangChainHumanMessage
from typing import Optional
import json
import sys
sys.path.append("C:/Users/USER/vscodeProjects/graphDB_pjt")
from db_to_graphDB.graphdb_llm import run_graphDB_schema_generation_llm
from db_to_graphDB.generate_graphDB import run_graphdb_pipeline
from text2cypher.cypher_llm import run_text2cypher_llm, initialize_retriever


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

@app.post("/graphdb_llm") #192.168.0.115:8344/graphdb_llm
def text_inference(
    file_name: Optional[str] = Form(None),
    assistant: str = Form("graphdb"),
    file: Optional[UploadFile] = File(None),
    schema: Optional[str] = Form(None),
    query: Optional[str] = Form(None)
):
    logger.info(f"Received params: file_name={file_name}, assistant={assistant}")
    try:
        if assistant == 'generate_graphdb_schema':
            response = run_graphDB_schema_generation_llm(file_name, file)
            print(response)

        elif assistant == 'generate_graphdb':
            if schema is None:
                print("schema가 없습니다")
            else:
                try:
                    parsed_schema = json.loads(schema)
                    if isinstance(parsed_schema, dict) and "response" in parsed_schema and isinstance(parsed_schema["response"], dict):
                        schema_to_process = parsed_schema["response"]
                    else:
                        schema_to_process = parsed_schema

                    with open("latest_schema.json", "w", encoding="utf-8") as f:
                        json.dump(parsed_schema, f, ensure_ascii=False, indent=2)

                    response = run_graphdb_pipeline(schema_to_process)
                    print(f">> graphDB 저장 성공 여부: {response}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON format in 'schema' for 'generate_graphdb'")
                    raise HTTPException(status_code=400, detail="Invalid JSON format for schema.")
                except Exception as e:
                    logger.error(f"Error processing schema for 'generate_graphdb': {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error processing schema: {str(e)}")

        elif assistant == 'text2cypher':
            with open("latest_schema.json", "r", encoding="utf-8") as f:
                schema_dict = json.load(f)
            retriever = initialize_retriever(schema=schema_dict)
            response = run_text2cypher_llm(retriever, query)
            print(response)

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

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8344)