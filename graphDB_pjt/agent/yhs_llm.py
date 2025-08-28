import config
import logging
import psycopg2
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent, AgentExecutor

# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# Neo4j 연결
URI = config.NEO4J_URI
AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    
#sql 연결
# sql_db = SQLDatabase.from_uri(config.DB_CONNECTION)
conn = psycopg2.connect(
    dbname=config.DB_NAME,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT
)

#history 
k=3
global_memory = ConversationBufferWindowMemory(k=k, memory_key="chat_history", return_messages=True, output_key="output" )

#llm
llm = ChatOpenAI(
        openai_api_key=config.OPEN_API_KEY,
        model="gpt-4o",
        temperature=0,
        max_tokens=4048
    )

def get_tables(_: str = "") -> str:
    """Return all table names as a formatted string."""
    try:
    
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """)
            rows = cursor.fetchall()

        if not rows:
            return "No tables found in the database."
        else:
            return "\n".join(f"- {schema}.{table}" for schema, table in rows)

    except Exception as e:
        return f"Error occurred while fetching tables: {e}"
    finally:
        if 'conn' in locals():
            conn.close()
            
              
def get_system_prompt():
    table_list = get_tables()
    print(table_list)
    system_prompt = """
    You are an intelligent agent that helps route natural language queries to the appropriate data system: SQL or Graph.
    
    ## Your Goal
    Given a user's question, follow these steps to determine how to answer it:

    1. Analyze the question and identify what information is required.
    2. Refer to the schemas below to understand what data is available in:
        - The **Relational Database (RDB)**, and
        - The **Graph Database (Neo4j)**.
    3. Determine which system(s) contain the necessary data.
    4. Use the appropriate tool:
        - Use `text2Cypher` if the required data is in the graph database.
        - Use `executeSQL` if the required data is in the relational database.
    5. **Important**: Only use the columns explicitly listed in the schema below.  
        - **Do not assume or generate column names that are not present.**
        - If the required information is not in the schema, Use other data to gain the information!
    6. Becareful, if the tex2Cypher or executeSQL is likely to return many results, consider generating a COUNT query first.
       Then decide whether to proceed with full retrieval or use LIMIT to avoid token overflow.    
    7. Some entities are linked across both systems via shared fields.
    8. Therefore, you may need to **use multiple data** to fully answer a question.
    
    
    
    Then call the corresponding tool among:
    {tools}

    ##  Database Information

    ### [1] Relational DB (PostgreSQL)
    This is a structured relational database with tabular schema.
    
    Possible table list name:
    """ +table_list+ """

    "custom_sales_plan" SCHEMA:
    """ +config.CUSTOM_SALES_PLAN_SCHEMA +"""
    
    "eina" SCHEMA:
    """ +config.EINA_SCHEMA + """
    
    "lfa1" SCHEMA:
    """ +config.LFA1_SCHEMA + """
    
    ## Business data is connected across tables
    ONTOLOGY SCHEMA: 
    """ +config.ONTOLOGY_SCHEMA + """
    
    ### [2] Graph DB (Neo4j)
    This database models product structure (BoM) as a graph.
    MAST_STPO_BOM SCHEMA:
    """ +config.NEO4J_SCHEMA+ """

    ---
    YOU MUST Use the following format especially Final Answer! :

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question in KOREAN. User is Korean.

    Agent Example:
    """ +config.AGENT_EXAMPLE +"""

    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )



def create_tools():
    @tool("text2Cypher")
    def text2Cypher(query: str) -> str:
        """
        Translates a natural language question into a Cypher query, executes it on the graph DB,
        and returns both the generated Cypher and search results.
        """
        try:
            retriever = Text2CypherRetriever(
                driver=driver,
                llm=llm,  # type: ignore
                neo4j_schema=config.NEO4J_SCHEMA,
                examples=config.NEO4J_EXAMPLES,
                custom_prompt=config.CYPHER_TEMPLATES
            )
            
            result = retriever.search(query_text=query)

            # Cypher 쿼리 추출
            cypher_query = result.metadata.get("cypher", "[No Cypher query generated]")

            # Cypher 결과 포맷팅
            if not result.items:
                results_str = "NO INFORMATION FOUND"
            else:
                formatted_items = [
                    f"[{i+1}] {item.content}" for i, item in enumerate(result.items)
                ]
                results_str = "\n".join(formatted_items)

            
            output = (
                f"[Cypher Query]\n{cypher_query}\n\n"
                f"[Results]\n{results_str}"
            )
            return output

        except Exception as e:
            return f"text2Cypher failed for query '{query}'. Error: {str(e)}"


    @tool("executeSQL")
    def executeSQL(sql_query: str) -> str:
        """
        Executes a raw SQL query on the relational database and returns the results.
    
        Note: Input must be a clean SQL string. No backticks, no markdown formatting.
        """
        try:            
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query)
                    result = cursor.fetchall()
                    return str(result)
        except Exception as e:
            try:
                conn.rollback()  # rollback 추가 SQL 한번 에러나면 롤백해줘야됨
            except:
                pass
            return f"executeSQL failed: {str(e)}"
        finally:
            if 'conn' in locals():
                conn.close()
    
    
    @tool("llm_direct")
    def llm_direct(input_text: str) -> str:
        """
        Use this tool when you do not need to use any actual tool
        and can answer the question directly using your own knowledge.
        
        This tool helps maintain proper ReAct formatting.
        """
        return f"LLM will answer directly: {input_text}"



    return [text2Cypher, executeSQL, llm_direct]

def create_agent(model, tools, prompt):
    """Create the OpenAI Functions agent."""
    return create_react_agent(model, tools, prompt)

def create_agent_executor(agent, tools, memory):
    """Create the agent executor."""
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    ) 
    
    
def initialize_agent():
    """Initialize the agent and return the executor."""
    memory = global_memory
    tools = create_tools()
    print(tools)
    prompt = get_system_prompt()
    agent = create_agent(llm, tools, prompt)
    agent_executor = create_agent_executor(agent, tools, memory)
    return agent_executor

def generate_dict_response(result):
    def extract_cypher_from_intermediate_steps(steps):
        """
        Extracts Cypher query from intermediate steps if tool is 'text2Cypher'.
        """
        for step in steps:
            action, observation = step
            if hasattr(action, 'tool') and action.tool == 'text2Cypher':
                if isinstance(observation, str) and "[Cypher Query]" in observation:
                    parts = observation.split("[Cypher Query]")
                    if len(parts) > 1:
                        cypher_block = parts[1].split("[Results]")[0].strip()
                        return cypher_block
        return None  # No Cypher found
    def extract_logs_from_intermediate_steps(steps):
        """
        Extracts the 'log' attribute from each action in intermediate steps.
        """
        logs = []
        for step in steps:
            action, _ = step
            if hasattr(action, 'log'):
                logs.append(action.log)
        return logs
    cypher_query = extract_cypher_from_intermediate_steps(result.get("intermediate_steps", []))
    logs = extract_logs_from_intermediate_steps(result.get("intermediate_steps", []))
    response_dict = {
        "response": result.get("output"),
        "cypher": cypher_query,
        "intermediate_steps": logs
    }

    return response_dict



def run_SAP_llm(query):
    
    agent_executor = initialize_agent()
    result = agent_executor.invoke({"input": query})
    response_dict = generate_dict_response(result)
    return response_dict

