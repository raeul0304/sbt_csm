import pandas as pd
import numpy as np
import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import openai
from openai import OpenAI
import os
import ast
import re
from config import gpt4o_OPENAI_API_KEY

# OpenAI API 키 환경 변수 설정
os.environ["OPENAI_API_KEY"] = gpt4o_OPENAI_API_KEY

class RAGSystem:
    """
    RAG(Retrieval-Augmented Generation) 시스템 클래스
    """
    def __init__(self, data_path='parameter_controls_data_for_rag.csv', openai_api_key=None):
        self.data_path = data_path
        self.search_data = None
        self.openai_api_key = openai_api_key or gpt4o_OPENAI_API_KEY
        self.llm = None
        self.client = None
        
    def initialize(self):
        """
        RAG 시스템 초기화
        """
        if self.search_data is None:
            self.search_data = self.generate_vectorDB()
            
        if self.llm is None and self.openai_api_key:
            # OpenAI 클라이언트 초기화
            self.client = OpenAI(api_key=self.openai_api_key)
            
            # ChatOpenAI 모델 초기화
            self.llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-4",
                openai_api_key=self.openai_api_key,
                max_tokens=4048
            )
            print("OpenAI API 연결 완료")
            
        return self.search_data
    
    def generate_vectorDB(self):
        """
        CSV 데이터를 로드하고 벡터 데이터베이스를 생성하는 함수
        """
        print("\n[데이터 로딩 시작]")
        # CSV 파일 로드
        df = pd.read_csv(self.data_path)
        print(f"CSV 파일 로드 완료: {len(df)} 행")
        
        # 문서 준비
        documents = []
        all_texts = []  # BM25 검색기를 위한 텍스트 리스트
        
        for idx, row in df.iterrows():
            # 각 행을 하나의 문서로 처리 (행 단위 chunking)
            content = f"질문: {row['질문']}\n답변: {row['답변']}"
            all_texts.append(content)
            
            # 메타데이터에 원본 답변 JSON과 질문 저장
            try:
                answer_json = json.loads(row['답변'])
            except:
                answer_json = ast.literal_eval(row['답변'])
                
            metadata = {
                "source": idx, 
                "original_answer": row['답변'],
                "original_question": row['질문']
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        print(f"문서 처리 완료: {len(documents)} 문서")
        
        # 임베딩 모델 설정 (한국어를 지원하는 모델 선택)
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        print("임베딩 모델 로드 완료")
        
        # 벡터 저장소 생성 (Dense 검색용)
        vectordb = FAISS.from_documents(documents, embeddings)
        print("벡터 저장소 생성 완료")
        
        # BM25 검색기 생성 (Sparse 검색용)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3  # 상위 3개 결과 반환
        print("BM25 검색기 생성 완료")
        
        # 검색 객체들 저장
        return {
            "vectordb": vectordb,
            "documents": documents,
            "bm25_retriever": bm25_retriever
        }
    
    def hybrid_search(self, user_input, k=3, dense_weight=0.7):
        """
        Dense 검색과 Sparse 검색을 결합한 하이브리드 검색 구현
        
        Args:
            user_input: 사용자 입력 텍스트
            k: 검색할 문서 수
            dense_weight: Dense 검색 결과의 가중치 (0~1)
        
        Returns:
            검색된 문서 리스트와 디버깅 정보
        """
        print(f"\n[검색 시작] 쿼리: {user_input}")
        
        if self.search_data is None:
            self.initialize()
            
        # Dense 검색 (벡터 유사도)
        dense_docs = self.search_data["vectordb"].similarity_search_with_score(user_input, k=k*2)
        print("\n[Dense 검색 결과]")
        for doc, score in dense_docs[:3]:
            print(f"질문: {doc.metadata['original_question']}")
            print(f"유사도 점수: {score}\n")
        
        # Sparse 검색 (BM25)
        sparse_docs = self.search_data["bm25_retriever"].invoke(user_input)
        print("\n[Sparse 검색 결과]")
        for doc, score in sparse_docs[:3]:
            print(f"질문: {doc.metadata['original_question']}\n")
            print(f"유사도 점수: {score}\n")
        
        # 결과 통합 및 점수 계산
        combined_results = {}
        debug_info = {
            "dense_results": [],
            "sparse_results": [],
            "final_results": []
        }
        
        # Dense 검색 결과 처리
        for doc, score in dense_docs:
            doc_id = doc.metadata["source"]
            # FAISS는 거리를 반환하므로 유사도로 변환 (1 - 정규화된 거리)
            similarity = 1 - (score / 100) if score > 0 else 0
            
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "doc": doc,
                    "dense_score": similarity,
                    "sparse_score": 0,
                    "final_score": dense_weight * similarity
                }
            
            # 디버깅 정보에 추가
            debug_info["dense_results"].append({
                "id": doc_id,
                "question": doc.metadata["original_question"],
                "score": similarity
            })
        
        # Sparse 검색 결과 처리
        for i, doc in enumerate(sparse_docs):
            doc_id = doc.metadata["source"]
            # BM25 결과의 순위를 점수로 변환 (역순으로)
            sparse_score = 1 - (i / len(sparse_docs)) if i < len(sparse_docs) else 0
            
            if doc_id in combined_results:
                combined_results[doc_id]["sparse_score"] = sparse_score
                combined_results[doc_id]["final_score"] += (1 - dense_weight) * sparse_score
            else:
                combined_results[doc_id] = {
                    "doc": doc,
                    "dense_score": 0,
                    "sparse_score": sparse_score,
                    "final_score": (1 - dense_weight) * sparse_score
                }
            
            # 디버깅 정보에 추가
            debug_info["sparse_results"].append({
                "id": doc_id,
                "question": doc.metadata["original_question"],
                "score": sparse_score
            })
        
        # 최종 점수로 정렬
        sorted_results = sorted(combined_results.values(), key=lambda x: x["final_score"], reverse=True)
        
        # 상위 k개 결과 반환
        top_results = [item["doc"] for item in sorted_results[:k]]
        
        # 디버깅 정보에 최종 결과 추가
        for item in sorted_results[:k]:
            debug_info["final_results"].append({
                "id": item["doc"].metadata["source"],
                "question": item["doc"].metadata["original_question"],
                "dense_score": item["dense_score"],
                "sparse_score": item["sparse_score"],
                "final_score": item["final_score"]
            })
        
        return top_results, debug_info
    
    def retrieve_data(self, user_input, k=3):
        """
        사용자 입력과 가장 관련성 높은 데이터를 검색하는 함수
        
        Args:
            user_input: 사용자 입력 텍스트
            k: 검색할 문서 수
        
        Returns:
            검색된 문서 리스트와 디버깅 정보
        """
        if self.search_data is None:
            self.initialize()
            
        retrieved_docs, debug_info = self.hybrid_search(user_input, k=k)
        return retrieved_docs, debug_info
    
    def generate_answer(self, user_input):
        """
        검색된 문서를 바탕으로 사용자 입력에 대한 답변을 생성하는 함수
        
        Args:
            retrieved_docs: 검색된 문서 리스트
            user_input: 사용자 입력 텍스트
        
        Returns:
            생성된 답변 (JSON 형식)
        """
        print("\n[답변 생성 시작]")
        
        answer_dict = {"message": "관련 정보를 찾을 수 없습니다."}

        retrieved_docs = self.retrieve_data(user_input)
        
        if not retrieved_docs or not self.llm:
            print("검색된 문서가 없거나 LLM이 초기화되지 않았습니다.")
            return answer_dict
            
        context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
        print(f"\n컨텍스트:\n{context}\n")
        
        # LLM에게 전달할 프롬프트 템플릿 (영어로 작성)
        prompt_template = """
        You are an AI assistant for a Korean financial forecasting system.
        Your task is to understand user requests in Korean and generate the appropriate JSON response.
        
        Use the following context to understand the system capabilities:
        
        Context:
        {context}
        
        User question (in Korean): {question}
        
        Based on the user's request, generate a SINGLE JSON object with the appropriate fields.
        The JSON object must include a field called "natural_language_response" with a friendly message in Korean.
        
        Possible fields to include (only use what's needed based on the user's request):
        - "option": For setting a product (e.g., "FERT101", "FERT102", etc.)
        - "option2": For setting a raw material (e.g., "ROH0001", "ROH0002", etc.)
        - "option3": For setting the FERT to ROH BOM display (e.g., "FERT101", "FERT201", etc.)
        - "m10change": For changing import quantity (as a string percentage, e.g., "5.0", "-3.0")
        - "m20change": For changing sales quantity (as a string percentage, e.g., "4.0", "-2.0")
        - "m110change": For changing USD exchange rate (as a string percentage, e.g., "4.0")
        - "active_tab": For switching to a specific tab (integer: 0 for inventory, 1 for cost calculation, 2 for profit summary)
        - "natural_language_response": A friendly Korean response explaining what was done
        
        Example responses:
        {{
          "option": "FERT101",
          "natural_language_response": "FERT101 상품으로 설정했습니다."
        }}

        {{
            "option3": "FERT201",
            "natural_language_response": "FERT201의 원재료 구성(BOM)을 표시합니다."
        }}
        
        {{
          "option": "FERT102",
          "m10change": "5.0",
          "active_tab": 1,
          "natural_language_response": "FERT102 상품의 입고 수량을 5% 증가시키고 원가계산서를 표시합니다."
        }}
        
        Return ONLY the JSON object with no additional text or explanation.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # LLM에게 응답 생성 요청
            response = chain.run(
                context=context,
                question=user_input
            ).strip()
            
            print(f"\nLLM 응답:\n{response}\n")
            
            # JSON 응답 파싱
            try:
                # 응답에서 JSON 객체만 추출하기 위한 정리
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                # JSON 파싱
                answer_dict = json.loads(response)
                print(f"\n최종 응답:\n{json.dumps(answer_dict, indent=2, ensure_ascii=False)}\n")
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시 직접 응답 사용
                print(f"\nJSON 파싱 오류: {str(e)}")
                answer_dict = {
                    "error": f"JSON 파싱 실패: {str(e)}",
                    "natural_language_response": "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
                }
                
        except Exception as e:
            # 오류 발생 시 기본 응답 유지
            print(f"\n오류 발생: {str(e)}")
            answer_dict["error"] = str(e)
            answer_dict["natural_language_response"] = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
        
        return answer_dict

# # 디버깅 정보 생성 함수
# def generate_debug_info(debug_info):
#     """
#     디버깅 정보를 문자열로 생성
#     """
#     if not debug_info or 'final_results' not in debug_info:
#         return ""
    
#     result = "최종 검색 결과:\n"
#     for i, doc_info in enumerate(debug_info['final_results']):
#         result += f"{i+1}. 질문: {doc_info['question']}\n"
#         result += f"   점수: {doc_info['final_score']:.4f} (Dense: {doc_info['dense_score']:.4f}, Sparse: {doc_info['sparse_score']:.4f})\n"
    
#     return result