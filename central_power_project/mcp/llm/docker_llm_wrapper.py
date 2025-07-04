from mcp_use import MCPClient
import requests
from typing import Sequence, Optional
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool

class DockerRemoteLLMWrapper(BaseChatModel):
    api_url: str = Field(...)
    assistant: str = Field(default="llm_f_mcp")
    _bound_tools: Optional[Sequence[BaseTool]] = None

    
    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun = None) -> ChatResult:
        
        prompt = "\n".join([msg.content for msg in messages if isinstance(msg, (HumanMessage))])

        payload = {
            "query": prompt,
            "assistant": self.assistant,
            "temperature": 0,
            "top_p": 0.9,
            "max_seq_len": 2048,
            "max_batch_size": 16,
            "max_gen_len": 256
        }
        try:
            response = requests.post(self.api_url, data=payload)
            response.raise_for_status()
            result_text = response.json()["response"]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result_text))])
        except requests.exceptions.HTTPError as e:
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)
            raise e
        
    
    def bind_tools(self, tools: Sequence[BaseTool]) -> BaseChatModel:
        self._bound_tools = tools
        return self
    
    @property
    def _llm_type(self) -> str:
        return "custom_remote_llm"