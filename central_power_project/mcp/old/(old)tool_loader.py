import json
import copy
from mcp_use import MCPClient
from langchain_core.tools import Tool

def load_mcp_client_from_config(config:dict, name: str) -> MCPClient:
    mcpServer_raw = config["mcpServers"][name]
    mcpServer_config = copy.deepcopy(mcpServer_raw) # 원본 손상 방지

    #필수 필드 점검 및 보정
    if "env" not in mcpServer_config:
        mcpServer_config["env"] = {}
    if "args" not in mcpServer_config:
        mcpServer_config["args"] = []
    
    return MCPClient.from_dict(mcpServer_config)


async def load_all_mcp_tools(config_path:str) -> list[Tool]:
    """"config에 정의된 모든 MCP 서버에 대해 세션을 만들고, 각 세션에서 도구를 불러와 Langchain Tool 객체로 변환"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    all_mcp_tools = []
    for mcpServer_name in config["mcpServers"].keys():
        try:
            print(f"MCP 서버에 연결: {mcpServer_name}")
            mcp_client = load_mcp_client_from_config(config, mcpServer_name)

            #1. 세션 생성 및 획득
            await mcp_client.create_session(mcpServer_name)
            session = mcp_client.get_session(mcpServer_name)
            print("세션 생성 및 획득 완료")

            #2. 도구 목록 불러오기
            tools = session.list_tools()
            print(f"도구 목록: {tools}")

            #3. LangChain Tool 객체로 변환
            for tool_def in tools:
                name = tool_def.name
                desc = getattr(tool_def, "description", "No description provided.")

                def make_tool(name=name):
                    return lambda **kwargs: session.call_tool(name, kwargs)
                
                tool = Tool.from_function(
                    func=make_tool(name),
                    name=name,
                    description=f"[{mcpServer_name}] {desc}"
                )
                all_mcp_tools.append(tool)
                print(f"도구 추가: {name}")

        except Exception as e:
            print(f"Error loading MCP client for {mcpServer_name}: {e}")
            continue
    return all_mcp_tools