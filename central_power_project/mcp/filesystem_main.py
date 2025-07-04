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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from llm.docker_llm_wrapper import DockerRemoteLLMWrapper
from mcp_tools.tool_loader import create_server_config

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



model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key="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")

async def main():
    server_config = create_server_config()
    print(f"서버 설정: {server_config}")
    async with MultiServerMCPClient(server_config) as client:
        agent = create_react_agent(model, client.get_tools())
        query = "바탕화면에 있는 폴더명들을 리스트해줘"
        print(f"\n[QUERY] {query}\n")

        # Langchain 형식의 입력 사용
        input_message = HumanMessage(content=query)

        response = await agent.ainvoke({"messages": [input_message]})

        # 전체 message 리스트 가져오기
        messages = response.get("messages", [])

        # 1) tool_calls에 있는 tool name 출력
        for msg in messages:
            if isinstance(msg, AIMessage):
                tool_calls = msg.additional_kwargs.get("tool_calls", [])
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    if tool_name:
                        print("사용된 tool:", tool_name)

        # 2) ToolMessage의 content만 출력
        for msg in messages:
            if isinstance(msg, ToolMessage):
                print("[RESULT] :", msg.content)
        

if __name__ == "__main__":
    asyncio.run(main())