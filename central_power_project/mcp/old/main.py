import asyncio
import os
import json
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from llm.docker_llm_wrapper import DockerRemoteLLMWrapper
from mcp_tools.tool_loader import load_all_mcp_tools

async def main():
    load_dotenv()

    #1. MCP 도구들 불러오기
    tools = await load_all_mcp_tools("C:\\Users\\USER\\vscodeProjects\\central_power_project\\mcp\\config\\config.json")
    

    #2. llm wrapper 준비
    api_url = "http://121.133.205.199:14777/chat"
    llm = DockerRemoteLLMWrapper(api_url=api_url, assistant="llm_f_mcp")

    #3. Langchain Agent 초기화
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    #4. 질의 실행
    result = await agent.run("바탕화면에 있는 mcp-test 폴더에 있는 폴더들을 리스트해줘")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
