import datetime
from datetime import timedelta
import requests
from typing import List
import logging
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
import config
import re
import os
# from retriever import retrieve_example
# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# OpenAI Client 설정
client = OpenAI(api_key=config.gpt4o_OPENAI_API_KEY)
model = ChatOpenAI(openai_api_key=config.gpt4o_OPENAI_API_KEY, temperature=0)

# Weather API Key 설정
OPENWEATHER_API_KEY = config.OPENWEATHERMAP_API_KEY
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = config.GOOGLE_CSE_ID


# agent가 쓸 도구들 정의(실제로 쓰는 도구는 tools변수 확인인)
def create_tools():
    # @tool("search")
    # def search_api(query : str) -> str:
    #     """Useful for retrieving recent supplementary information from Google."""
        # os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY

    #     search = TavilySearchResults(k=5) 
    #     return search.run(query)
    @tool("search")
    def search_api(query : str) -> str:
        """Useful for retrieving recent supplementary information from Google."""
        search = GoogleSearchAPIWrapper()
        return search.run(query)
    @tool("current_weather")
    def current_weather(query: str) -> str:
        """search current weather information for a specific location. Input should be the name of a location"""
        try:
            weather  = OpenWeatherMapAPIWrapper(openweathermap_api_key = config.OPENWEATHERMAP_API_KEY)
            return weather.run(query)
        except Exception as e:
            return f"Weather tool faild for query '{query}'.\nProceed again or try other tools. Error: {str(e)}"
    
    @tool("rag_search")
    def rag_search(query: str) -> str:
        """
        Retrieves relevant information from the RAG system based on the query.
        Useful for finding supplementary information from internal knowledge sources.
        """
        try:
            retrieved_titles, retrieved_contents = retrieve_example(query, top_k=1)

            
            # 검색된 문서를 정리하여 반환
            if not retrieved_titles or not retrieved_contents:
                return "No relevant information found from RAG."
            
            result = "\n\n".join(
                        [f"Title: {title}\nContent: {content}" for title, content in zip(retrieved_titles, retrieved_contents)]
                    )    
            return result    
        except Exception as e:
            return f"RAG search failed for query '{query}'. Error: {str(e)}"
    
    
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
        return result
    tools = [search_api,current_weather, mclick]

    return tools


def create_prompt():
    """Create the prompt for the agent."""
    today = datetime.datetime.today().strftime("%D")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a powerful assistant. Most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. 
                
                Today is {today}. Final Answer should be in Korean.
                
                You have the tool search. Use search in the following circumstances: 
                
                - User is asking about current events or something that requires real-time information (weather, economic data, etc.) 
                
                - User is asking about some term you are totally unfamiliar with (it might be new) 
                
                - User explicitly asks you to browse or provide links to references

                Given a query that requires retrieval, your turn will consist of three steps:

                1. Call the search function to get a list of results.

                2. Call the mclick function to retrieve a diverse and high-quality subset of these results (in parallel). Remember to SELECT 2 url when using mclick.

                3. Write a summary response to the user based on these results. 

                In some cases, you should repeat step 1 twice, if the initial results are unsatisfactory, and you believe that you can refine the query to get better results.

                The search tool has the following commands: search(query: str) Issues a query to a search engine and displays the results. mclick(ids: list[str]). Retrieves the contents of the webpages with provided links. You should ALWAYS SELECT 2 links. Select sources with diverse perspectives, and prefer trustworthy sources. Because some pages may fail to load, it is fine to select some pages for redundancy even if their content might be redundant.
                """,
            ),
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

def create_agent_executor(agent, tools,memory):
    """Create the agent executor."""
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
    ) 

def initialize_agent(memory):
    """Initialize the agent and return the executor."""
    # 모델 생성
    model = ChatOpenAI(
        openai_api_key=config.gpt4o_OPENAI_API_KEY,
        temperature=0,
    )

    tools = create_tools()
    prompt = create_prompt()
    agent = create_agent(model, tools, prompt)
    agent_executor = create_agent_executor(agent, tools, memory=memory)
    return agent_executor

def run_agent_search(query):
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    agent_executor = initialize_agent(memory = memory)

    result = agent_executor.invoke({"input": query})
    memory.save_context({"HummanMessage": query}, {"output": result["output"]})
    # intermediate_steps에서 mclick의 URL만 추출
    urls = []
    for action, _ in result.get("intermediate_steps", []):
        if action.tool == "mclick":
            tool_input = action.tool_input
            urls.extend(tool_input.get("search_results_cache", []))

    return {
        "output": result["output"],
        "reference_url": urls
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run agent search with a query.")
    parser.add_argument("query", type=str, help="User query for the agent.")
    args = parser.parse_args()

    response = run_agent_search(args.query)
    print("=== Agent Output ===")
    print(response["output"])
    print("\n=== Reference URLs ===")
    for url in response["reference_url"]:
        print(url) 
