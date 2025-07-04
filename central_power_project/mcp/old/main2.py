import json
import asyncio
import asyncio.base_subprocess as _bsp
import asyncio.proactor_events as _pro
import warnings
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from llm.docker_llm_wrapper import DockerRemoteLLMWrapper

# Windows asyncio 오류 방지 (fileno closed pipe 등)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception as e:
        print(f"이벤트 루프 정책 설정 중 오류 발생: {e}")
        pass

def _safe_repr(self):
    try:
        return object.__repr__(self)
    except Exception:
        return "<closed transport>"

# Override __repr__ to avoid fileno errors
_bsp.BaseSubprocessTransport.__repr__ = _safe_repr
_pro._ProactorBasePipeTransport.__repr__ = _safe_repr

# Override __del__ to prevent closing transports after loop shutdown
_bsp.BaseSubprocessTransport.__del__ = lambda self: None
_pro._ProactorBasePipeTransport.__del__ = lambda self: None

# ResourceWarning 무시
warnings.filterwarnings("ignore", category=ResourceWarning)



load_dotenv()

api_url = "http://121.133.205.199:14777/chat"
model = DockerRemoteLLMWrapper(api_url=api_url, assistant="llm_f_mcp")
model = ChatOpenAI(model="gpt-4o-mini")

def load_mcp_config():
    try:
        with open("C:\\Users\\USER\\vscodeProjects\\central_power_project\\mcp\\config\\filesystem_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None

def create_server_config():
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # command가 있으면 stdio 방식
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio"
                }
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse"
                }
    return server_config

# 도구 호출 로그를 남기기 위한 래핑 함수
def wrap_tool(tool):
    async def _wrapped_tool(*args, **kwargs):
        print(f"[TOOL INVOKED] '{tool.name}' 호출됨")
        return await tool.coroutine(*args, **kwargs)
    
    return Tool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_wrapped_tool,
        coroutine=_wrapped_tool
    )


async def main():
    server_config = create_server_config()
    print(f"서버 설정: {server_config}")
    async with MultiServerMCPClient(server_config) as client:
        original_tools = client.get_tools()
        wrapped_tools = [wrap_tool(t) for t in original_tools]
        
        agent = create_react_agent(model, wrapped_tools)

        query = "바탕화면의 프레젠테이션 폴더에 있는 파일들을 리스트해줘"
        print(f"\n[QUERY] {query}\n")

        # Langchain 형식의 입력 사용
        input_message = HumanMessage(content=query)

        # ReAct 에이전트에 메시지 전달
        response = await agent.ainvoke({"messages": [input_message]})

        # 도구 호출 정보 출력
        if "intermediate_steps" in response:
            print("\n[도구 호출 정보]")
            steps = response["intermediate_steps"]
            for i, step in enumerate(steps):
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    print(f"도구: {action.tool}")
                    print(f"이자: {action.tool_input}")
                    print(f"결과: {observation}")
                    print("---")

        print("\n[최종 결과]")

        # 여러 가능한 응답 형식 처리
        if "output" in response:
            # output 필드가 있는 경우 (일반적인 ReAct 에이전트 응답)
            print(response["output"])
        elif "messages" in response and response["messages"]:
            # messages 필드가 있는 경우 (마지막 메시지 추출)
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                print(last_message.content)
            elif isinstance(last_message, dict) and "content" in last_message:
                print(last_message["content"])
        else:
            # 기타 응답 형식 (응답 전체 출력)
            print("응답 형식을 인식할 수 없습니다.")
            print(f"응답 키: {list(response.keys())}")
            
            # 응답의 일부 내용만 출력 시도
            for key in ["answer", "result", "response", "content"]:
                if key in response:
                    print(f"\n가능한 응답({key}):")
                    print(response[key])


if __name__ == "__main__":
    asyncio.run(main())