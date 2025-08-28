#TODO: Retrieving similary schemas with RAG - elasticSearch

import config
import logging
import json
import os
import pandas as pd
from typing import List, Dict, Any
from uuid import uuid4
from elasticsearch.helpers import BulkIndexError
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, ElasticsearchStore
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings


# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)


_embeddings_cache = None

def get_embedding_model():
    global _embeddings_cache
    if _embeddings_cache is None:
        try:
            # _embeddings_cache = OpenAIEmbeddings(
            #     openai_api_key=config.OPEN_API_KEY_SERVER,
            #     model="text-embedding-3-small"
            # )
            _embeddings_cache = HuggingFaceEmbeddings(
                 model_name="BAAI/bge-base-en-v1.5"
            )
            logger.info("OpenAI Embedding 모델이 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Embedding 모델 초기화 중 오류 발생: {e}")
            raise e
    return _embeddings_cache


def index_exists(index_name: str, es_url: str) -> bool:
    es_client = Elasticsearch(es_url)
    return es_client.indices.exists(index=index_name)

# Vector DB 저장
def add_schema_data_to_vdb(schema_json_path: str, vector_store, index_name, es_url) -> ElasticsearchStore:
    # 인덱스가 이미 존재하면 생략
    # if index_exists(index_name=index_name, es_url=es_url):
    #     print(f"Index '{index_name}' already exists. Skipping document addition.")
    #     return vector_store

    with open(schema_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("DEBUG: type(data) =", type(data))  # <class 'list'>
    print("DEBUG: schema count =", len(data))
    
    documents: List[Document] = []

    for table in data:
        table_name = table["table_name"]
        chunks = []

        # summary 저장
        documents.append(Document(
            page_content=f"[Table: {table_name}]\n{table['schema_explanation']}",
            metadata={
                "table_name": table_name,
                "chunk_type": "table_summary"
            }
        ))

        # node 저장
        for node in table.get("nodes", []):
            documents.append(Document(
                page_content = (
                    f"[Node: {node['label']}]\n"
                    f"Description: {node['description']}\n"
                    f"Properties: {', '.join(node.get('properties', []))}"
                ),
                metadata={
                    "table_name": table_name,
                    "chunk_type": "node",
                    "label": node["label"],
                    "properties_list": ', '.join(node.get("properties", []))
                }
            ))


        # relation 저장
        for rel in table.get("relationships", []):
            documents.append(Document(
                page_content=(
                    f"[Relationship: {rel['from']} - [{rel['type']}] -> {rel['to']} ] \n"
                    f"Description: {rel['description']}\n"
                    f"Properties: {', '.join(rel.get('properties', []))}"
                ),
                metadata={
                    "table_name": table_name,
                    "chunk_type": "relationship",
                    "from": rel["from"],
                    "to": rel["to"],
                    "relationship_type": rel["type"],
                    "properties_list": ', '.join(rel.get("properties", []))
                }
            ))
    
    print(f"총 {len(documents)}개 문서를 처리중입니다")

    uuids = [str(uuid4()) for _ in range(len(documents))]

    try:
        vector_store.add_documents(documents=documents, ids=uuids)
        return vector_store

    except BulkIndexError as e:
        for i, err in enumerate(e.errors):
            reason = err.get('index', {}).get('error', {}).get('reason', 'No reason')
            doc_id = err.get('index', {}).get('_id', 'No ID')
            print(f"[{i}] Failed to index document ID {doc_id} → Reason: {reason}")
            print("Full error:", err)


def store_vector_db():
    embeddings = get_embedding_model()
    index_name = config.ELASTICSEARCH_INDEX_NAME
    es_url = config.ELASTICSEARCH_URL

    # 1. Elasticsearch 클라이언트 생성
    es = Elasticsearch(es_url)

    # 2. 기존 인덱스 삭제 (768차원 모델에 맞추기 위해)
    if es.indices.exists(index=index_name):
        logger.warning(f"Elasticsearch 인덱스 '{index_name}' 가 존재합니다. 삭제 후 재생성합니다.")
        es.indices.delete(index=index_name)
        logger.info(f"인덱스 '{index_name}' 삭제 완료.")

    # 3. 새로운 인덱스 매핑 생성 (768차원 dense vector)
    mapping = {
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 768,  # ✅ BGE 모델에 맞게 변경
                    "index": True,
                    "similarity": "cosine"  # 또는 "l2_norm" 등
                }
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    logger.info(f"인덱스 '{index_name}' 768차원으로 재생성 완료.")

    # 4. ElasticsearchStore 생성
    vector_store = ElasticsearchStore(
        index_name=index_name,
        embedding=embeddings,
        es_url=es_url,
        vector_query_field="vector"
    )

    # 5. 문서 삽입
    schema_path = "/home/pjtl2w01admin/csm/graphDB_pjt/ontology_routing/schema_data.json"
    add_schema_data_to_vdb(
        schema_json_path=schema_path,
        vector_store=vector_store,
        index_name=index_name,
        es_url=es_url
    )



# def store_vector_db():
#     embeddings = get_embedding_model()

#     elastic_vector_search = ElasticsearchStore(
#         index_name = config.ELASTICSEARCH_INDEX_NAME,
#         es_url = config.ELASTICSEARCH_URL,
#         embedding=embeddings
#     )

#     vector_store = ElasticsearchStore(
#         config.ELASTICSEARCH_INDEX_NAME,
#         embedding = embeddings,
#         es_url = config.ELASTICSEARCH_URL,
#         vector_query_field="vector"
#     )

#     add_schema_data_to_vdb(
#         schema_json_path="/home/pjtl2w01admin/csm/graphDB_pjt/ontology_routing/schema_data.json",
#         vector_store=vector_store,
#         index_name = config.ELASTICSEARCH_INDEX_NAME,
#         es_url=config.ELASTICSEARCH_URL
#     )


# 유사도 높은 스키마 가져오기
def get_schemas_from_retrieved_docs(dense_docs: List[Document], sparse_docs: List[Document], schema_json_path: str) -> List[dict]:
    all_docs = dense_docs + sparse_docs
    # 1. table_name 전부 수집 (중복 없이)
    table_names = {
        doc.metadata.get("_source", {}).get("metadata", {}).get("table_name")
        for doc in all_docs
        if doc.metadata.get("_source", {}).get("metadata", {}).get("table_name")
    }

    print(f"[DEBUG] table_names found in retrieved docs: {table_names}")

    # 2. 전체 스키마 로드
    with open(schema_json_path, "r", encoding="utf-8") as f:
        full_schema = json.load(f)

    # 3. 필요한 table_name만 필터링
    matched_schemas = [
        schema for schema in full_schema
        if schema.get("table_name") in table_names
    ]

    return matched_schemas



#Dense Search
def vector_query(search_query: str) -> Dict:
    embeddings = get_embedding_model()
    query_vector = embeddings.embed_query(search_query)
    return {
        "knn": {
            "field": config.VECTOR_FIELD,
            "query_vector": query_vector,
            "k": 8,
            "num_candidates": 15,
        }
    }

def dense_retriever(search_query):
    return ElasticsearchRetriever.from_es_params(
        index_name = config.ELASTICSEARCH_INDEX_NAME,
        body_func=lambda search_query: vector_query(search_query),
        content_field=config.CONTENT_FIELD,
        url=config.ELASTICSEARCH_URL
    )

def dense_search(query):
    vector_retriever = dense_retriever(query)
    dense_search_docs = vector_retriever.invoke(query)
    return dense_search_docs



#Sparse Search
def bm25_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {
                config.TEXT_FIELD : search_query,
            }
        }
    }

def sparse_retriever(query):
    return ElasticsearchRetriever.from_es_params(
        index_name = config.ELASTICSEARCH_INDEX_NAME,
        body_func=lambda query: bm25_query(query),
        content_field=config.TEXT_FIELD,
        url=config.ELASTICSEARCH_URL
    )

def sparse_search(query):
    bm25_retriever = sparse_retriever(query)
    sparse_search_docs = bm25_retriever.invoke(query)
    return sparse_search_docs



# Retrieval - 실행
def search_related_ontologies(query):
    schema_json_path = "/home/pjtl2w01admin/csm/graphDB_pjt/ontology_routing/schema_data.json"
    embeddings = get_embedding_model()

    store_vector_db()

    dense_search_docs = dense_search(query)
    print(f"벡터 유사도 : {dense_search_docs}")
    sparse_search_docs = sparse_search(query)
    print(f"키워드 유사도: {sparse_search_docs}")

    matched_schemas = get_schemas_from_retrieved_docs(dense_search_docs, sparse_search_docs, schema_json_path)
    print(f"최종적으로 유사도가 높은 스키마는 : {matched_schemas} 입니다")
    return matched_schemas