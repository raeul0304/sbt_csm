from typing import Annotated, Literal, Sequence, TypedDict
import operator
import functools

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import tool

from yhs_llm import create_tools  # text2Cypher, executeSQL, llm_direct 포함된 LangChain Tool 목록
from ontology_routing import run_ontology_selection_prompt  # 사용자가 제공한 ontology 기반 reasoning 함수
from config import ONTOLOGY_INSTRUCTION, EXAMPLE_SCHEMAS, ONTOLOGY_EXAMPLE, APPLIED_EXAMPLE, ONTOLOGY_SCHEMA, ONTOLOGY_MAPPING

# ---------------------------
# 1. 상태 정의
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    used_schemas: list[str]
    cypher: str
    intermediate_steps: list

# ---------------------------
# 2. Tool + LLM 설정
# ---------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = create_tools()

# ---------------------------
# 3. ontology 기반 사전 reasoning 수행
# ---------------------------
def inject_ontology_reasoning(user_query: str) -> str:
    return run_ontology_selection_prompt(
        ontology_instruction=ONTOLOGY_INSTRUCTION,
        example_schemas=EXAMPLE_SCHEMAS,
        ontology_example=ONTOLOGY_EXAMPLE,
        applied_example=APPLIED_EXAMPLE,
        schema_data=ONTOLOGY_SCHEMA,
        mapping_information=ONTOLOGY_MAPPING,
        user_query=user_query
    )

# ---------------------------
# 4. ReAct Agent 생성 (create_react_agent 사용)
# ---------------------------
planner_system_prompt_template = """
You are a schema reasoning planner. You must follow the predefined reasoning path based on the ontology.

Required schemas and reasoning chain:
{ontology_reasoning}

Your job is to perform Thought → Action → Observation steps according to this reasoning plan.
Only use the following tools: [text2Cypher, executeSQL, llm_direct]

Always use this format:
Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
...
Thought: I now know the final answer
Final Answer: (final answer in KOREAN)
"""

# ---------------------------
# 5. 노드 함수 정의
# ---------------------------

def planner_agent_factory(ontology_reasoning: str):
    planner_prompt = planner_system_prompt_template.format(ontology_reasoning=ontology_reasoning)
    return create_react_agent(
        llm=llm,
        tools=tools,
        system_prompt=planner_prompt
    )

def agent_node(state: AgentState) -> AgentState:
    user_query = state["messages"][0].content
    ontology_reasoning = inject_ontology_reasoning(user_query)
    planner_agent = planner_agent_factory(ontology_reasoning)

    result = planner_agent.invoke(state)
    tool_calls = result.get("tool_calls", [])
    outputs = result.get("messages", [])
    new_steps = state.get("intermediate_steps", [])

    for tool_call, output in zip(tool_calls, outputs):
        new_steps.append(
            f"Tool: {tool_call.name}, Input: {tool_call.args}, Output: {output.content}"
        )

    return {
        "messages": result["messages"],
        "next": "PlannerAgent",
        "used_schemas": state.get("used_schemas", []),
        "cypher": state.get("cypher", ""),
        "intermediate_steps": new_steps
    }

def should_continue(state: AgentState) -> Literal["PlannerAgent", "Analyzer"]:
    if "Final Answer:" in state["messages"][-1].content:
        return "Analyzer"
    return "PlannerAgent"

def analyzer_node(state: AgentState) -> AgentState:
    final_answer = state["messages"][-1].content
    steps = "\n".join(state.get("intermediate_steps", []))
    full_result = f"[Final Answer]\n{final_answer}\n\n[Steps]\n{steps}"
    return {"messages": [AIMessage(content=full_result)]}

# ---------------------------
# 6. LangGraph 구성
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("PlannerAgent", agent_node)
graph.add_node("Analyzer", analyzer_node)

graph.set_entry_point("PlannerAgent")
graph.add_conditional_edges("PlannerAgent", should_continue, {
    "PlannerAgent": "PlannerAgent",
    "Analyzer": "Analyzer"
})
graph.add_edge("Analyzer", END)

app = graph.compile()

# ---------------------------
# 7. 실행
# ---------------------------
if __name__ == "__main__":
    inputs = {
        "messages": [HumanMessage(content="2025년 1분기에 가장 많이 팔린 제품의 부품과 공급처 위치 알려줘")],
        "next": "PlannerAgent",
        "used_schemas": [],
        "cypher": "",
        "intermediate_steps": []
    }

    for step in app.stream(inputs, stream_mode="values"):
        print("\n🔹 Step result:")
        print(step["messages"][-1].content)
