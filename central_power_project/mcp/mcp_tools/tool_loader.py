import json
import subprocess
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "ko_KR.UTF-8"

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
