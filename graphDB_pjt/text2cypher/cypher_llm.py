import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import json
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# model은 cypher 쿼리용, llm은 최종 답변 추론용 
model = ChatOpenAI(openai_api_key=config.OPEN_API_KEY_SERVER, model_name="gpt-4o", temperature=0, max_tokens=4048)
llm = ChatOpenAI(openai_api_key=config.OPEN_API_KEY_TEST, model_name="gpt-4o", temperature=0, max_tokens=4048)
#model = ChatGoogleGenerativeAI(google_api_key=config.GEMINI_API_KEY, model="gemini-2.5-flash-preview-05-20", temperature = 0, max_output_tokens = 2024)
#llm = ChatGoogleGenerativeAI(google_api_key=config.GEMINI_API_KEY, model="gemini-1.5-flash", temperature = 0, max_output_tokens = 2024)

# Neo4j 연결
URI = config.NEO4J_URI
AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# 노드 임베딩 > TODO: 속성에 저장을 해두는 건데 실행할 때마다 매번 수행할 필요 없음 
def generate_node_embeddings(driver, token):
    logger.info("Generating node embeddings...")
    query = """
    MATCH (m:Metric)
    WHERE m.name IS NOT NULL AND m.amount IS NOT NULL
    WITH m, m.name + 'level:' + toString(m.level) + 'amount:' + toString(m.amount) AS description
    CALL genai.vector.encode(description, 'OpenAI', { token: $token}) YIELD vector
    CALL db.create.setNodeVector(m, 'embedding', vector)
    RETURN m.name AS name, vector
    """
    with driver.session() as session:
        result = session.run(query, token=token)
        print("임베딩 저장 결과:")
        for record in result:
            print(f"- {record['name']}")

# vector index 생성
def create_vector_index(driver):
    query = """
    CREATE VECTOR INDEX metricEmbedding IF NOT EXISTS
    FOR (m:Metric)
    ON m.embedding
    OPTIONS {
        indexConfig: {
            'vector.dimensions': 1536,
            'vector.similarity_function': 'COSINE'
        }
    }
    """
    with driver.session() as session:
        session.run(query)
        print("metric 벡터 인덱스 생성 완료")


# 유사도 검색
def similarity_search(driver, query_text, token, top_k=5):
    query = """
    CALL genai.vector.encode($query_text, 'OpenAI', { token: $token}) YIELD vector
    CALL db.index.vector.queryNodes('Metric', 'embedding', vector, $top_k)
    YIELD node, score
    RETURN node.name AS name, node.level AS level, node.amount AS amount, score
    """
    with driver.session() as session:
        result = session.run(query, query_text=query_text, token=token, top_k=top_k)
        print(f"{query_text}에 대한 유사도 검색 결과:")
        for record in result:
            print(f" - {record['name']}, level: {record['level']}, amount: {record['amount']}, score: {record['score']:.4f}")


# 리트리버 정의
def initialize_retriever(schema: dict):
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=model,  # type: ignore
        neo4j_schema=json.dumps(schema),
        examples=config.KNA1_NEO4J_EXAMPLES,
        custom_prompt = config.KNA1_CYPHER_TEMPLATES
    )
    return retriever

def get_system_prompt(query):
    
    system_prompt = """You are an assistant that generates clear, professional, and fact-based answers in Korean.

    You must rely **only on the provided "Information"**, which comes from a Neo4j graph database. Do not make assumptions, hallucinate missing information, or use external knowledge. Do not mention the use of a database or describe the query process in your response.

    Important instructions:
    - Do **not** generate any information not present in the provided Information.
    - If no relevant information is found, respond with:
    "이 질문에 대해서는 개인 DB 기반 정보가 없어 정확한 답변을 드릴 수 없습니다."
    - Write all answers in **KOREAN**, with a confident, helpful tone. Avoid passive voice and technical disclaimers.
    - Always try to generate response when cypher information is provided.

    ---

    Information:
    {retrieved_contents}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system_prompt),
            ("human", 
             query)
        ]
    )
    return prompt

def run_text2cypher_llm(retriever, query):
    logger.info(f"자연어 질문 >> {query}")
    retrieved_data= retriever.search(query_text=query)
    if not retrieved_data.items:
        print(f">>>>사용된 Cypher: \n{retrieved_data.metadata.get('cypher')}\n>>>>사이퍼 결과: 없음 (검색 결과 없음)")
        retrieved_contents = "NO INFORMATION FOUND"

    else:
        formatted_items = []
        for i, item in enumerate(retrieved_data.items):
            line = f"[{i+1}] {item.content}"
            formatted_items.append(line)
    
        retrieved_contents = f"[Cypher Query]\n{retrieved_data.metadata.get('cypher')}\n\n[Results]\n" + "\n".join(formatted_items)
        print(f"retrieved_contents:\n {retrieved_contents}")


    system_prompt = get_system_prompt(query)
    chain = system_prompt | llm | StrOutputParser()
    response = chain.invoke({'query': query, 'retrieved_contents': retrieved_contents})
    response_dict = generate_dict_response(response, retrieved_data)
    return response_dict


def generate_dict_response(response, retrieved_data):
    response_dict = {
        "response": response,
        "cypher": retrieved_data.metadata.get('cypher'),
    }
    #print(f">>> json 답안: {response_dict}")
    return response_dict



def run_similarity_search():
    generate_node_embeddings(driver, token=config.OPEN_API_KEY_SERVER)
    create_vector_index(driver)
    similarity_search(driver, query_text="CN과 관련된 회사들이 뭐야?", token=config.gpt4o_OPENAI_API_KEY)


#테스트용
if __name__ == "__main__":
    #TODO : schema 예시 넣고, retriever 연결하기
    schema = '''{'table_name': 'KNA1', 'nodes': ['Customer', 'Address', 'City', 'Country', 'TaxInfo', 'Lifecycle', 'CustomerType', 'Extra'], 'relationships': [{'from': 'Customer', 'to': 'Address', 'type': 'HAS_ADDRESS', 'properties': ['STRAS', 'PSTLZ']}, {'from': 'Address', 'to': 'City', 'type': 'IN_CITY', 'properties': ['ORT01']}, {'from': 'City', 'to': 'Country', 'type': 'PART_OF', 'properties': ['LAND1']}, {'from': 'Customer', 'to': 'TaxInfo', 'type': 'HAS_TAX_INFO', 'properties': ['STCD2', 'STCD5']}, {'from': 'Customer', 'to': 'Lifecycle', 'type': 'HAS_LIFECYCLE', 'properties': ['ERDAT', 'ERNAM', 'DUEFL']}, {'from': 'Customer', 'to': 'CustomerType', 'type': 'HAS_TYPE', 'properties': ['KTOKD']}, {'from': 'Customer', 'to': 'Extra', 'type': 'HAS_EXTRA', 'properties': []}], 'primary_key': {'Customer': 'KUNNR', 'Address': 'ADRNR', 'City': 'ORT01', 'Country': 'LAND1', 'TaxInfo': 'KUNNR', 'Lifecycle': 'KUNNR', 'CustomerType': 'KTOKD', 'Extra': 'KUNNR'}, 'node_properties': {'Customer': ['KUNNR', 'NAME1', 'NAME2', 'SORTL', 'MCOD1', 'MCOD2', 'MCOD3', 'ANRED'], 'Address': ['ADRNR', 'STRAS', 'PSTLZ'], 'City': ['ORT01', 'REGIO'], 'Country': ['LAND1'], 'TaxInfo': ['STCD2', 'STCD5'], 'Lifecycle': ['ERDAT', 'ERNAM', 'DUEFL'], 'CustomerType': ['KTOKD'], 'Extra': ['MANDT', 'BBBNR', 'BBSNR', 'BUBKZ', 'UMSAT', 'UMJAH', 'JMZAH', 'JMJAH', 'UMSA1', 'HZUOR', 'J_1KFREPRE', 'J_1KFTBUS', 'J_1KFTIND', 'UPTIM', 'RIC', 'LEGALNAT', '_VSO_R_PALHGT', '_VSO_R_I_NO_LYR', '_VSO_R_ULD_SIDE', '_VSO_R_LOAD_PREF', 'PSPNR', 'J_3GSTDMON', 'J_3GSTDTAG', 'J_3GTAGMON', 'J_3GVMONAT', 'J_3GEMINBE', 'J_3GFMGUE', 'J_3GZUSCHUE']}, 'used_fields': ['KUNNR', 'NAME1', 'NAME2', 'SORTL', 'MCOD1', 'MCOD2', 'MCOD3', 'ANRED', 'ADRNR', 'STRAS', 'PSTLZ', 'ORT01', 'REGIO', 'LAND1', 'STCD2', 'STCD5', 'ERDAT', 'ERNAM', 'DUEFL', 'KTOKD']}
{
  "response": {
    "table_name": "KNA1",
    "nodes": [
      "Customer",
      "Address",
      "City",
      "Country",
      "TaxInfo",
      "Lifecycle",
      "CustomerType",
      "Extra"
    ],
    "relationships": [
      {
        "from": "Customer",
        "to": "Address",
        "type": "HAS_ADDRESS",
        "properties": [
          "STRAS",
          "PSTLZ"
        ]
      },
      {
        "from": "Address",
        "to": "City",
        "type": "IN_CITY",
        "properties": [
          "ORT01"
        ]
      },
      {
        "from": "City",
        "to": "Country",
        "type": "PART_OF",
        "properties": [
          "LAND1"
        ]
      },
      {
        "from": "Customer",
        "to": "TaxInfo",
        "type": "HAS_TAX_INFO",
        "properties": [
          "STCD2",
          "STCD5"
        ]
      },
      {
        "from": "Customer",
        "to": "Lifecycle",
        "type": "HAS_LIFECYCLE",
        "properties": [
          "ERDAT",
          "ERNAM",
          "DUEFL"
        ]
      },
      {
        "from": "Customer",
        "to": "CustomerType",
        "type": "HAS_TYPE",
        "properties": [
          "KTOKD"
        ]
      },
      {
        "from": "Customer",
        "to": "Extra",
        "type": "HAS_EXTRA",
        "properties": []
      }
    ],
    "primary_key": {
      "Customer": "KUNNR",
      "Address": "ADRNR",
      "City": "ORT01",
      "Country": "LAND1",
      "TaxInfo": "KUNNR",
      "Lifecycle": "KUNNR",
      "CustomerType": "KTOKD",
      "Extra": "KUNNR"
    },
    "node_properties": {
      "Customer": [
        "KUNNR",
        "NAME1",
        "NAME2",
        "SORTL",
        "MCOD1",
        "MCOD2",
        "MCOD3",
        "ANRED"
      ],
      "Address": [
        "ADRNR",
        "STRAS",
        "PSTLZ"
      ],
      "City": [
        "ORT01",
        "REGIO"
      ],
      "Country": [
        "LAND1"
      ],
      "TaxInfo": [
        "STCD2",
        "STCD5"
      ],
      "Lifecycle": [
        "ERDAT",
        "ERNAM",
        "DUEFL"
      ],
      "CustomerType": [
        "KTOKD"
      ],
      "Extra": [
        "MANDT",
        "BBBNR",
        "BBSNR",
        "BUBKZ",
        "UMSAT",
        "UMJAH",
        "JMZAH",
        "JMJAH",
        "UMSA1",
        "HZUOR",
        "J_1KFREPRE",
        "J_1KFTBUS",
        "J_1KFTIND",
        "UPTIM",
        "RIC",
        "LEGALNAT",
        "_VSO_R_PALHGT",
        "_VSO_R_I_NO_LYR",
        "_VSO_R_ULD_SIDE",
        "_VSO_R_LOAD_PREF",
        "PSPNR",
        "J_3GSTDMON",
        "J_3GSTDTAG",
        "J_3GTAGMON",
        "J_3GVMONAT",
        "J_3GEMINBE",
        "J_3GFMGUE",
        "J_3GZUSCHUE"
      ]
    },
    "used_fields": [
      "KUNNR",
      "NAME1",
      "NAME2",
      "SORTL",
      "MCOD1",
      "MCOD2",
      "MCOD3",
      "ANRED",
      "ADRNR",
      "STRAS",
      "PSTLZ",
      "ORT01",
      "REGIO",
      "LAND1",
      "STCD2",
      "STCD5",
      "ERDAT",
      "ERNAM",
      "DUEFL",
      "KTOKD"
    ]
  }
}'''
    retriever = initialize_retriever(schema)
    query = "한국(KR)에 있는 고객들을 리스트해줘"
    print(f">>>>질문: {query}")
    result = run_text2cypher_llm(retriever, query)
    print(f">>>>LLM 결과: {result}")