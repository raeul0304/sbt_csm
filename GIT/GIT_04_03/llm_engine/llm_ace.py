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
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
import config
import re
import os
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


# agent가 쓸 도구들 정의(실제로 쓰는 도구는 tools변수 확인인)
def create_tools():
    # @tool("search")
    # def search_api(query : str) -> str:
    #     """Useful for retrieving recent supplementary information from Google."""
    #     try: 
    #         search = SerpAPIWrapper(serpapi_api_key = config.SERP_API_KEY)
    #         res = search.results(query)
    #         organic = res.get("organic_results", [])

    #         if not organic:
    #             return "<NL> 검색 결과가 없습니다. 다시 시도해 주세요"
    #         return res["organic_results"][:3]
    #     except Exception as e:
    #         return f"오류 발생: {str(e)}"

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




def create_prompt(context):
    """Create the prompt for the agent."""
    today = datetime.datetime.today().strftime("%D")
    print("context",context)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                f"""You are an intelligent assistant that responds to user requests by **either generating a strict JSON object** or by **using the search tool for real-time information**, depending on the nature of the request.
                
                Choose which of the two instruction sets to follow based on the user's request:

                ----------------------------------------------------------------------
                
                 **IF the user’s request is a structured business query** that matches any of the following:
                    - Setting a product (e.g., "FERT101", "FERT102" etc.)
                    - Setting a raw material (e.gg, "ROH0001", etc)
                    - Displaying the BOM structure (Donut Graph)
                    - Adjusting import/export quantity or exchange rates (in percentage)
                    - Switching application tabs (inventory, cost, or profit)

                THEN:
                    - Respond with a **SINGLE JSON object** that contains only the appropriate fields.
                    - The JSON object **must** include a `"natural_language_response"` field in **friendly Korean**, explaining the action taken.
                    - Do **not include any explanation or extra text** outside the JSON.
                    - Use only the fields required for the user's request.
                
                    Use the following retrieved data that you must reference to generate a strict JSON object response:
            
                    Retrieved Data:
                    {context}

                    Possible fields to include (only use what's needed based on the user's request):
                    - "option": For setting a product (e.g., "FERT101", "FERT102", etc.)
                    - "option2": For setting a raw material (e.g., "ROH0001", "ROH0002", etc.)
                    - "option3": For setting the FERT to ROH BOM display (related with Donut Graph or Chart) (e.g., "FERT101", "FERT201", etc.)
                    - "m10change": For changing import quantity (as a string percentage, e.g., "5.0", "-3.0")
                    - "m20change": For changing sales quantity (as a string percentage, e.g., "4.0", "-2.0")
                    - "m110change": For changing USD exchange rate (as a string percentage, e.g., "4.0")
                    - "natural_language_response": A friendly Korean response explaining what was done

                    Example Response1:
                        "option3": "FERT201",
                        "natural_language_response": "FERT201의 원재료 구성(BOM) 도넛 그래프를를 표시합니다."
                    
                    Example Response2:
                        "option": "FERT102",
                        "m10change": "5.0",
                        "m20change": "8.0",
                        "m110change" : "-3.0",
                        "option2" : "ROH0002",
                        "m50change2" : "10.0",
                        "active_tab": 1,
                        "natural_language_response": "FERT102 상품의 입고 수량을 5% 증가, 판매 수량을 8% 증가, USD 환율을 3% 감소시키고, ROH0002 원재료의 구매단가를 10% 증가시킨 결과를 표시합니다."              
        
                -----------------------------------------------------------
                Today is {today}
                
                **If the user’s query requires current events or unfamiliar knowledge**, such as:
                
                    - User is asking about current events or something that requires real-time information (weather, economic data, etc.) 
                    
                    - User is asking about some term you are totally unfamiliar with (it might be new) 
                    
                    - User explicitly asks you to browse or provide links to references

                THEN: 

                    1. Call the search function to get a list of results.

                    2. Write a summary response to the user, based on these results without link.

                    2. Write a summary response to the user in sentence, based on these results. 
                    
                    3. Always ADD <NL> tag infront of the answer if you use any tool(search, current_weather). Do not answer in json format!
                    
                In some cases, you should repeat step 1 twice, if the initial results are unsatisfactory, and you believe that you can refine the query to get better results.

                If no result is retrieved, check whether the retrieved data corresponds to a structured business json in Example Data
                
                The search tool has the following commands: search(query: str) Issues a query to a search engine and displays the results.
                
                The curent_weather tool has the following commands: current_weather(query: str) Issues a query to retrieve weather.
                ----------------------------------------------------------
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),

            (
                "user",
                "query: {input}"
            ),
            (
                "assistant",
                "{agent_scratchpad}"  # agent_scratchpad를 추가하여 중간 단계를 기록
            ),
        ]
    )
    return prompt




def create_agent(model, tools, prompt):
    """Create the OpenAI Functions agent."""
    return create_openai_functions_agent(model, tools, prompt)

def create_agent_executor(agent, tools, memory):
    """Create the agent executor."""
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory = memory,
        return_intermediate_steps=True,
    ) 

def initialize_agent(query):
    """Initialize the agent and return the executor."""
    # 모델 생성
    model = ChatOpenAI(
        openai_api_key=config.gpt4o_OPENAI_API_KEY,
        temperature=0,
    )
    memory = global_memory
    tools = create_tools()
    context = initialize_rag(query)
    prompt = create_prompt(context)
    agent = create_agent(model, tools, prompt)
    agent_executor = create_agent_executor(agent, tools, memory)
    return agent_executor

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
