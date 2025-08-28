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
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

pipeline_instance = None

@app.post("/graphdb_llm") #121.133.205.199:14878/graphdb_llm
def text_inference(
    file_name: Optional[str] = Form(None),
    assistant: str = Form("graphdb"),
    file: Optional[UploadFile] = File(None),
    schema: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    user_edit_text: Optional[str] = Form(None)
):
    global pipeline_instance
    logger.info(f"Received params: file_name={file_name}, assistant={assistant}")
    try:
        if assistant == 'generate_graphdb_schema':
            pipeline_instance = GraphDBSchemaPipeline(file_name, file)
            response = pipeline_instance.run_graphDB_schema_generation_llm()
            print(response)
        
        elif assistant = 'regenerate_graphdb_schema':
            if schema is None:
                print("schema가 없습니다")
            if pipeline_instance is None:
                print("pipeline instance가 초기화되지 않았습니다.")
            else:
                try:
                    parsed_schema = json.loads(schema) 
                    response = pipeline_instance.run_regenerate_schema(parsed_schema, user_edit_text)
                    print(f">>> 스키마 재생성 성공 여부 : {response}")
                except Exception as e:
                    logger.error(f"Error processing schema regeneration : {str(e)}")

        elif assistant == 'generate_graphdb':
            if schema is None:
                print("schema가 없습니다")
            if pipeline_instance is None:
                print("pipeline instance가 초기화되지 않았습니다.")
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
            
            pipeline = GraphDBSchemaPipeline(file_name="", file=None)
            pipeline.schema = schema_dict  # 수동 설정
            pipeline.generate_text2cypher_examples()  # LLM 호출해서 examples 생성

            retriever = Text2CypherRetriever(
                driver=driver,
                llm=pipeline.model,
                neo4j_schema=json.dumps(pipeline.schema),
                examples=pipeline.cypher_examples,
                custom_prompt=config.CYPHER_TEMPLATES
            )
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