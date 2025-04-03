import sys
sys.path.insert(0, '/home/pjtl2w01admin/csm/GIT/llm_engine')

from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import OpenAIEncoder
import datetime
from datetime import timedelta
import requests
from typing import List
import logging
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import Tool
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
import config
import os
import json
from llm_w_rag import RAGSystem

# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)
model = ChatOpenAI(openai_api_key=config.gpt4o_OPENAI_API_KEY, temperature=0)
# Weather API Key 설정
OPENWEATHER_API_KEY = config.OPENWEATHERMAP_API_KEY
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = config.GOOGLE_CSE_ID
k=3
global_memory = ConversationBufferWindowMemory(k=k, memory_key="chat_history", return_messages=True, output_key="output" )


# 1. 라우터 정의
routers = [
    Route(name="rag", utterances=[
        "FERT101의 입고 수량을 5% 증가해줘",
        "FERT104의 판매 수량을 9% 증가해줘",
        "FERT201의 환율을 7% 증가해줘"
        "ROH0002 원재료의 구매단가를 7% 증가해줘",
        "FERT201의 BOM을 그래프로 보여줘"]),
    Route(name="search", utterances=[
        "오늘 서울 날씨 알려줘",
        "오늘 판교 날씨 알려줘",
        "현재 원유 가격을 알려줘",
        "현재와 비교하면 12개월동안 원유가격이 몇% 변동된거야?",
        "2025년 3월 석탄 가격은? 뉴캐슬 기준으로 알려줘"]),
    Route(name="llm", utterances=[
        "오늘 기분이 좀 안 좋다",
        "스트레스가 쌓인다",
        "GPT와 BERT의 차이점이 뭐야?",
        "좋은 사업 아이디어를 추천해줘",
        "벚꽃 나들이 가기 좋은 곳을 추천해줘",
        "오늘 같이 날씨 좋은 날은 놀러가야 하는데..",
        "날씨가 안 좋다"
    ])
]

# 2. 인코더 설정
encoder = OpenAIEncoder()

# 3.라우터 생성
router = SemanticRouter(encoder=encoder, routes=routers, auto_sync="local")

# agent가 쓸 도구들 정의
def create_tools():
    @tool("search")
    def search_api(query : str) -> str:
        """Useful for retrieving recent supplementary information from Google."""
        try: 
            search = GoogleSearchAPIWrapper()
            res = search.results(query, num_results=3)
            if not res:
                return "<NL> 검색 결과가 없습니다. 다시 시도해 주세요"
            return res
        except Exception as e:
            return f"오류 발생: {str(e)}"

    @tool("current_weather")
    def current_weather(query: str) -> str:
        """search current weather information for a specific location in English. Input should be the name of a location in English"""
        try:
            weather  = OpenWeatherMapAPIWrapper(openweathermap_api_key = config.OPENWEATHERMAP_API_KEY)
            return weather.run(query)
        except Exception as e:
            return f"Weather tool faild for query '{query}'.\nProceed again or try search tool. Error: {str(e)}"
    
    
    def fetch_content(url, query):
        """
        주어진 URL에서 메인 콘텐츠를 가져옵니다.
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find('article') or soup.find('main') or soup.body
            if main_content:
                text = main_content.get_text(strip=True)
                # LLM 요약 요청
                prompt = f"find the information related to the query in the following text \n\n query:{query} \n text:{text}"
                return model(prompt)
            return "Unable to extract meaningful content."
        except requests.exceptions.RequestException as e:
            return f"Error fetching the URL: {e}"
        
    # URL이 포함되는 search tool이 선행되어서 사용되어야함
    @tool('mclick')
    def mclick(search_results_cache: list, query: str):
        """
        Tool for retrieving detailed content from url in search result 
        
        Parameters:
        - search_results_cache(list): link result of search tool. example ['www.google.com', 'www.naver.com', 'www.langchain.com']
        - query(str): user's query
        Returns:
        - list: summary content of the url 
        """
        results = []

        for index in range(len(search_results_cache)):
            group_result = {}
            match =search_results_cache[index]
            if match:
                content = fetch_content(match, query)
                group_result[index] = {"content": content}
            else:
                group_result[index] = {"error": f"No search result found for ID: {search_results_cache[index]}"}
            results.append(group_result)

        return results
        
    tools = [search_api,current_weather]

    return tools

# rag 연결
def initialize_rag(query):
    answer_dict = {"message": "관련 정보를 찾을 수 없습니다."}

    # (문서들, 메타데이터) 반환
    retrieved_docs, debug_info = RAGSystem().retrieve_data(user_input=query)

    if not retrieved_docs:
        print("검색된 문서가 없거나 LLM이 초기화되지 않았습니다.")
        return answer_dict

    # [문서 1] 형태로 구성 + 중괄호 이스케이프
    context_raw = "\n\n".join([
        f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs[:3])
    ])
    safe_context = context_raw.replace("{", "{{").replace("}", "}}")

    print(f"\n최종 컨텍스트:\n{safe_context}\n")
    return safe_context


# agent 설정
def create_agent(model, tools, prompt):
    return create_openai_functions_agent(model, tools, prompt)


def create_agent_executor(agent, tools, memory):
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
    )

def initialize_agent(query):
    model = ChatOpenAI(
        openai_api_key=config.gpt4o_OPENAI_API_KEY,
        temperature=0
    )
    memory = global_memory
    tools = create_tools()
    prompt = create_agent_prompt(query)
    agent = create_agent(model, tools, prompt)
    agent_executor = create_agent_executor(agent, tools, memory)
    return agent_executor



# Rag Prompt
def create_rag_prompt(context):
    prompt = """
        You are an AI assistant for a Korean financial forecasting system.
        Your task is to understand user requests in Korean and generate the appropriate JSON response.
        
        Use the following context to understand the system capabilities:
        
        Context:
        {context}
        
        User question (in Korean): {question}
        
        Based on the user's request, respond with a **SINGLE JSON object** that contains only the appropriate fields.
        The JSON object **must** include a `"natural_language_response"` field in **friendly Korean**, explaining the action taken.
        Do **not include any explanation or extra text** outside the JSON.
        Use only the fields required for the user's request.
                
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
            "m20change": "8.0",
            "m110change" : "-3.0",
            "option2" : "ROH0002",
            "m50change2" : "10.0",
            "active_tab": 1,
            "natural_language_response": "FERT102 상품의 입고 수량을 5% 증가, 판매 수량을 8% 증가, USD 환율을 3% 감소시키고, ROH0002 원재료의 구매단가를 10% 증가시킨 결과를 표시합니다."              
        }}
        
        Return a valid JSON object only. Do not return text, markdown, or any explanation outside of the JSON.
    """
    return ChatPromptTemplate.from_template(prompt)


# rag 처리
def generate_rag_response(query, context):
    prompt = create_rag_prompt(context)
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(context=context, question=query)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "JSON 파싱 실패", "natural_language_response": response}
    


# Search Prompt
def create_agent_prompt(context):
    today = datetime.datetime.today().strftime("%D")
    print("context", context)

    return ChatPromptTemplate.from_messages([
        ("system", 
        f"""
        You are a powerful assistant. Most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs.
        Today is {today}. Final Answer should be in Korean.

        You have the tool search. Use search in the following circumstances:
        - User is asking about current events or something that requires real-time information (weather, economic data, etc.)
        - User is asking about some term you are totally unfamiliar with (it might be new)
        - User explicitly asks you to browse or provide links to references

        Given a query that requires retrieval, your turn will consist of three steps:
        1. Call the search function to get a list of results.
        2. Write a summary response to the user in sentence, based on these results.
        3. Always ADD <NL> tag infront of the answer if you use any tool(search, current_weather). Do not answer in json format!
                
        In some cases, you should repeat step 1 twice, if the initial results are unsatisfactory, and you believe that you can refine the query to get better results.
        The search tool has the following commands: search(query: str) Issues a query to a search engine and displays the results.
        The curent_weather tool has the following commands: current_weather(query: str) Issues a query to retrieve weather.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

# search 처리
def run_agent(query):
    agent_executor = initialize_agent(query)

    result = agent_executor.invoke({"input": query})
    # 메모리 상태 확인
    print("🧠 현재 메모리:")
    for m in agent_executor.memory.chat_memory.messages[-k:]:
        print(f"{m.type}: {m.content}")
    print("intermediate steps:", result.get("intermediate_steps"))
        
    # intermediate_steps에서 mclick의 URL만 추출
    intermediate_steps = result.get("intermediate_steps", [])

    urls = []
    titles= [] 
    for action, observation in intermediate_steps:
        if action.tool == "search" and isinstance(observation, list):
            for item in observation:
                if isinstance(item, dict) and "link" in item:
                    urls.append(item["link"])
                if isinstance(item, dict) and "title" in item:
                    titles.append(item["title"])

    return {
        "output": result["output"],
        "reference_url": urls,
        "titles": titles,
    }


# llm Prompt
def run_free_chatting(query):
    messages = [
        {"role": "system", "content": "당신은 친절한 한국어 AI 비서입니다."},
        {"role": "user", "content": query}
    ]
    client = OpenAI(api_key=config.gpt4o_OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return {"output": response.choices[0].message.content}


# 전체 라우팅 및 실행
def route_and_respond(query):
    match = router(query)
    category = match.name if match else "llm"
    logger.info(f"[라우팅 결과] {query} -> {category}")

    if category == "rag":
        context = initialize_rag(query)
        return generate_rag_response(query, context)
    
    elif category == "search":
        return run_agent(query)
    
    else:
        return run_free_chatting(query)
    

# 테스트
if __name__ == "__main__":
    user_query = "오늘 날씨 되게 좋다~"
    result = route_and_respond(user_query)
    print("응답")
    print(json.dumps(result, indent=2, ensure_ascii=True))