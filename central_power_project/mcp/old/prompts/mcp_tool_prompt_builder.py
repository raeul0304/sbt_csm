from langchain_core.messages import SystemMessage
from mcp_use import MCPClient

def get_mcp_tool_schema_description(client: MCPClient) -> str:
    tool_descriptions=[]
    for tool in client.get_tools():
        schema = tool.schema.schema()
        props = schema.get("properties", {})
        param_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        example_args = ", ".join(f'{k}="..."' for k in props.keys())

        block = f"""
- {tool.name}({param_str}): {tool.description}
  Example:
  {{
    "action": "{tool.name}",
    "action_input": {{ {example_args} }}
  }}
"""
        tool_descriptions.append(block.strip())
    return "\n\n".join(tool_descriptions)


def get_available_server_names(client: MCPClient) -> list[str]:
    server_names = list({tool.metadata["serverName"] for tool in client.get_tools() if "serverName" in tool.metadata})
    return server_names


def build_mcp_tool_prompt(client: MCPClient) -> SystemMessage:
    tool_info = get_mcp_tool_schema_description(client)
    available_servers = ", ".join(get_available_server_names(client))
    prompt = f"""
You are an AI assistant integrated with the MCP (Model Context Protocol) system. 
You have access to tools provided by the following MCP servers: {available_servers}

Do NOT hallucinate or invent any information. Instead, always choose the most appropriate tool listed below to answer user's request.

ONLY Respond in the following JSON format. Other fomats are NOT allowed:
{{
  "action": "<tool_name>",
  "action_input": {{
    "param1": "...",
    "param2": "..."
  }}
}}

Available tools:

{tool_info}
"""
    return SystemMessage(content=prompt.strip())