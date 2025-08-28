import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import logging
import json
import pandas as pd
from io import BytesIO
from fastapi import UploadFile
from langchain_openai import ChatOpenAI
from db_to_graphDB.extract_preprocessed_data import list_preprocessed_columns, read_excel_file, preprocess_data


# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# TMP 디렉토리 설정
TMP_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)  # 디렉토리가 없으면 생성


def extract_fields_description(table_name: str, fields: list) -> dict:
    fields_and_description = {}
    try:
        with open("/home/pjtl2w01admin/csm/graphDB_pjt/rag/column_description.json", "r", encoding="utf-8") as f:
            column_descriptions = json.load(f)
        
        table_specific_columns= {}
        for entry in column_descriptions:
            if entry.get("table_name").upper() == table_name.upper():
                table_specific_columns = {k.upper(): v for k, v in entry.get("columns", {}).items()}
                break
        
        for field in fields:
            description = table_specific_columns.get(field.upper(), field)
            fields_and_description[field] = description
        print(fields_and_description)
        return fields_and_description
    
    except Exception as e:
        logger.error(f"컬럼 설명을 추출하는 중 오류가 발생했습니다: {str(e)}")
        
        

def get_data_info(file_name, file: UploadFile):
    table_name = file_name

    try:
        #file.file.seek(0)
        #df = pd.read_excel(BytesIO(file.file.read()), dtype=str)
        df = pd.read_excel(file, dtype=str)
        df.columns = [col.strip().upper() for col in df.columns]
        df = df.fillna("").applymap(str.strip)
    except Exception as e:
        logger.error(f"엑셀 파일 로딩 실패: {str(e)}")
        raise e
    
    cleaned_df = preprocess_data(df)

    store_file(table_name, cleaned_df, version_name="cleaned")

    fields = list_preprocessed_columns(cleaned_df)
    fields_and_description = extract_fields_description(table_name, fields)
    sample_rows = cleaned_df.head(10).to_dict(orient='records')

    data_info = {
        "table_name": table_name,
        "fields_and_description" : fields_and_description,
        "sample_rows": sample_rows
    }
    logger.info("Data information successfully extracted")

    return data_info




def create_graphdb_schema(model, data_info, retrieved_context: str):
    required_keys = ['table_name', 'fields_and_description', 'sample_rows']
    
    for key in required_keys:
        if key not in data_info:
            raise ValueError(f"Missing required key in data_info: {key}")
    
    try:
        # sample_rows 문자열화
        sample_str = json.dumps(data_info['sample_rows'], indent=2)
        logger.info(f"retrieve된 문서들 : {retrieved_context}")

        schema_prompt = f"""
You are a data modeling expert with deep understanding of SAP systems and business processes. Your task is to analyze the following SAP table and design a comprehensive and semantically meaningful GraphDB schema.

# Data Information : 
- Table Name: {data_info['table_name']}
- Sample Rows:
{sample_str}
- Extra Important information:
{retrieved_context}

Your task consists of the following steps:

---
**Step 1: Understand Inherent Data Information & Column Semantics**

First, meticulously analyze each column (field) in the provided table.
- **Define Column Meaning:** For each column, explicitly describe its business meaning and  and what real-world concept or attribute it represents.
- **Identify Key Entities & Attributes:** Determine what primary business objects and their essential attributes are implied by these columns.
- **Infer Relationships & Structure:** Based on the column values, enterprise data patterns and SAP context, infer the inherent relationships between these entities. Recognize if there's a hierarchical or many-to-many structure implied.

---
**Step 2: Identify All Business Purposes / Usage Scenarios of the Data**

Based on your comprehensive understanding from Step 1, identify **all distinct and valuable business usage contexts or scenarios** for this data. Think broadly about how this data could empower various organizational functions. Consider:
- **Core Business Processes:** How does this data support fundamental operations (e.g., order fulfillment, financial reporting, material management, human resources, sales analytics)?
- **Analytical Insights:** What types of insights can be derived (e.g., performance metrics, trend analysis, root cause analysis, predictive modeling)?
- **Decision Support:** How can this data inform strategic or operational decisions (e.g., resource allocation, risk assessment, process optimization, market segmentation)?
- **Master Data Management:** How does it define, maintain, or relate master data records (e.g., customer, vendor, product master)?
- **Transactional Context:** How does it capture and describe specific business events or transactions?
- **Reporting & Dashboards:** What key performance indicators (KPIs) or reports could be generated?
- **Integration & Data Flow:** How might this data interact with or be enriched by other systems or datasets (e.g., external market data, IoT sensor data, CRM systems, HR systems)?

You are not limited to one usage scenario; enumerate all relevant and valuable ones.


---

**Step 3: Design the GraphDB Schema**

Translate your comprehensive understanding from Step 1 and Step 2 into a GraphDB schema, following the specified JSON output format.

- **schema description**: write a summarized explanation of the schema. include what entities and relationships (with direction) are included in natural language.

- **Nodes (Entities)**: Identify all significant business entities that should be represented as nodes.
  - For each node:
    - `label`: A clear, semantically meaningful name.
    - `primary_key`: The column(s) that uniquely identify this node.
    - `description`: A rich, business-contextual explanation of what the entity represents, its key identifying properties, how it typically relates to other entities, and any overall/specific usage.
    - `properties`: A list of all relevant original column names that describe this node.

- **Relationships (Edges)**: Define explicit connections between your nodes based on the inherent relationships and identified business usages.
  - For each relationship:
    - `from`: The label of the source node.
    - `to`: The label of the target node.
    - `type`: A clear, semantically meaningful relationship type (e.g., `HAS_ITEM`, `PERFORMS_TRANSACTION`, `BELONGS_TO`).
    - `properties`: A list of original column names that describe the relationship itself -> ONLY include the core ones if needed.
    - `description`: A detailed explanation of the business logic or connection it represents, the direction's implication, and how it enables specific data navigation, classification, or aggregation.

- **'Extra' Node:** Create an 'Extra' node to capture any original columns that are not used in your primary graph schema. This node ensures no data is lost.
  - Connect this 'Extra' node to the most relevant primary entity (e.g., `Material`, `Order`, `Document`) via a generic relationship (e.g., `HAS_EXTRA_INFO`). Clearly state which primary node it connects to.

---

**Output Format**
Return the result as **valid JSON only**, following this structure:

{{
  "table_name": "{data_info['table_name']}",
  "schema_explanation" : "The following is an ontology consisting of the entities Person and Food. There exists a FRIEND_OF relationship between persons, and a LIKES relationship between persons and food. The friendship relationship is bidirectional."
  "nodes": [
    {{
      "label": "EntityA",
      "primary_key": "field2",
      "description": "EntityA represents the main business object in this dataset. It is uniquely identified by a combination of field2, field4, field11, and field12 (stored as a hash in concatenatedFields). The node typically serves as the starting point in relationships with other entities such as EntityB. The 'name' property is used for human-readable display.",
      "properties": ["name", "field2", "field4", "field11", "field12"]
    }},
    {{
      "label": "EntityB",
      "primary_key": "concatenatedFields",
      "description": "EntityB contains field3, which is used as an external lookup key to find matching weather or cost data",
      "properties": ["name", "field1", "field8", "field3"]
    }},
    {{
      "label": "EntityC",
      "primary_key": "field9",
      "description": "EntityC stores auxiliary classification or category data. It is identified by 'field9' and used primarily for lookup, filtering, or enrichment operations in queries.",
      "properties": ["name", "field9"]
    }},
    {{
      "label": "Extra",
      "primary_key": "field2",
      "description": "This node stores all unused or supplementary fields not captured in the main schema. It connects to the primary entity (EntityA) via a general-purpose relationship. It helps preserve full row fidelity.",
      "properties": ["name", "UnusedField1", "UnusedField2"]
    }}
  ],
  "relationships": [
    {{
      "from": "EntityA",
      "to": "EntityB",
      "type": "RELATION_TYPE",
      "properties": ["field3"],
      "description": "This relationship indicates that EntityA is associated with EntityB through a linkage defined by 'field3'. It reflects a core business connection such as assignment, ownership, or dependency."
    }},
    {{
      "from": "EntityB",
      "to": "EntityC",
      "type": "CLASSIFIED_AS",
      "properties": [],
      "description": "This relationship indicates that each EntityB instance can be categorized under a corresponding EntityC class. EntityB instances are classified under EntityC via CLASSIFIED_AS."
    }}
    ],
  "used_fields" : ["field2", "field4", "field11", "field12", "field1", "field8", "field3", "field9"]
}}

---

Guidelines:
- **Domain Agnostic:** Design the schema to be adaptable to various SAP modules (FI, CO, SD, MM, PP, HR, etc.) without hardcoding domain-specific terms unless explicitly derived from the *sample data provided*.
- **Exhaustive:** Uncover **all meaningful entities, relationships, and their properties** embedded in the dataset.
- **Business Context Focus:** Every element in the schema should clearly map back to a real-world business concept or process, regardless of the specific SAP module.
- **Logical Flow:** The schema should be intuitive and logically support all identified business usages.
- **Respond ONLY with the JSON object, without any explanation or formatting (no markdown, no code blocks)**
"""
        
        response = model.invoke(schema_prompt)

        try:
            parsed = json.loads(response.content)
            logger.info("GraphDB schema 생성 성공")
            return parsed
        except json.JSONDecodeError:
            logger.warning("LLM 응답이 JSON 형식이 아닙니다")
            return response
        
    except Exception as e:
        logger.error(f"GraphDB schema 생성 중 오류 발생: {str(e)}")
        return None



def edit_graphdb_schema(schema, user_edit_text):
  model = get_model()
  retrieved_documents = search_related_ontologies(json.dumps(data_info))
  
  schema_prompt = f"""
  These were the schema data you had to observe : {retrieved_documents}
  You have generated the schema like this : {schema}
  However, 
  """
  response = model.invoke(schema_prompt)

  try:
            parsed = json.loads(response.content)
            logger.info("GraphDB schema 재생성 성공")
            return parsed
        
  except Exception as e:
        logger.error(f"GraphDB schema 재생성 중 오류 발생: {str(e)}")
        return None
  




def create_cypher_template(model, graphdb_schema):
    parsed_schema = graphdb_schema if isinstance(graphdb_schema, dict) else json.loads(graphdb_schema)
    graphdb_schema_dict = parsed_schema.get("response", parsed_schema)
    data_info = graphdb_schema_dict.get("table_name", "KNA1")
    cypher_prompt = f"""
You are an expert in Cypher query language for Neo4j GraphDB.

Table name used in SAP system:
{data_info}

Given the following GraphDB schema:
{json.dumps(graphdb_schema, indent=2)}

---

### Goal :
Generate Cypher MERGE templates **based on the given GraphDB schema** for:
1. Each node (entity) : MERGE and SET n += {{...}} using the node properties info in the schema.
2. Each relationship : MATCH and MERGE with relationships info in the schema.

---
### RULES

1. **Use Exact Field Names**  

- Use field names exactly as given in the schema — no renaming, mapping, or formatting.
- Apply these names in both Cypher properties and parameters (e.g., FIELD_X → $FIELD_X).


2. **Node MERGE Logic**  

- Use the primary key field from the schema **only if it is a strong identifier** for that node.
- A field is **not** a strong identifier if:
  - It is reused across different nodes (e.g., shared customer ID).
  - It represents a category, type, or descriptive flag (e.g., KTOKD, STATUS).
  - It lacks uniqueness or is insufficient to distinguish one node from another.
  - exception : 'Extra' node's primary key must be 'KUNNR', 'City' node's pk must be 'ORT01'

- If the primary key does not satisfy these conditions, use the following instead:
  - MERGE with `ID: $concatenatedFields`
  - Add `"pk": "concatenatedFields"`
  - Add `"fields": [fieldA, fieldB, ...]` to define the components properties
  - For example, if CustomerType node has "ERDAT", "ERNAM", "DUEFL" as its property in the schema,
  its template would look like : "CustomerType": {{
      "cypher": "MERGE (n:CustomerType {{ID: $concatenatedFields}}) SET n += {{name: $concatenatedFields, KTOKD: $KTOKD, LIFNR: $LIFNR, ANRED: $ANRED, SPRAS: $SPRAS}}",
      "pk": "concatenatedFields",
      "fields": ["KTOKD", "LIFNR", "ANRED", "SPRAS"],
    }} >> reference the example output for more examples of this.

    
3. **'name' Property Logic** 

- Each node must include a `name` property in the SET clause (placed first).  
- Choose a human-readable field (e.g., NAME1, STRAS) that best represents the entity.  
- If not, use the value of `concatenatedFields` as the name.
- For the "Extra" node, always use `name: "extra"` (literal string) as the first field.


4. **Cypher Template Output Requirements**  

For each node, include:
- `"cypher"`: the full MERGE and SET Cypher statement  
- `"pk"`: either the proper field name, or `"concatenatedFields"`
- `"fields"`: list of all fields used in the SET clause

5. **Relationship MATCH/MERGE Logic**  

- Use the primary keys (or concatenatedFields if applicable) of the source and target nodes for MATCH.
- Use only the necessary fields to establish the relationship - you don't need to have the property in the relationship if not neccessary.
- Provide `"cypher"`, `"from_key"`, and `"to_key"` explicitly.

----
### Example Output (when the data is about customers):
{{
  "node_merge_templates": {{
    "Customer": {{
      "cypher": "MERGE (n:Customer {{KUNNR: $KUNNR}}) SET n += {{name: $NAME1, KUNNR: $KUNNR, NAME2: $NAME2}}",
      "pk": "KUNNR",
      "fields": ["KUNNR", "NAME1", "NAME2", "ANRED"]
    }},
    "Address" : {{
      "cypher": "MERGE (n:Address {{ADRNR: $ADRNR}}) SET n += {{name: $STRAS, ADRNR: $ADRNR, PSTLZ: $PSTLZ, ORT02: $ORT02}},
      "pk": "ADRNR",
      "fields: ["STRAS", "ADRNR", "PSTLZ"]
    }},
    "City" : {{
      "cypher": "MERGE (n:City {{ORT01: $ORT01}}) SET n += {{name: $ORT01, REGIO: $REGIO}}",
      "pk": "ORT01",
      "fields": ["ORT01", "REGIO"]
    }},
    "Country": {{
      "cypher": "MERGE (n:Country {{LAND1: $LAND1}}) SET n += {{name: $LAND1, LAND1: $LAND1}}",
      "pk": "LAND1",
      "fields": ["LAND1"]
    }},
    "TaxInfo": {{
      "cypher": "MERGE (n:TaxInfo {{ID: $concatenatedFields}}) SET n += {{name: $STCD2, STCD2: $STCD2, STCD5: $STCD5}}",
      "pk": "concatenatedFields",
      "fields": ["STCD2", "STCD5"],
    }},
    "Lifecycle": {{
      "cypher": "MERGE (n:Lifecycle {{ID: $concatenatedFields}}) SET n += {{name: $ERNAM, ERDAT: $ERDAT, DUEFL: $DUEFL}}",
      "pk": "concatenatedFields",
      "fields": ["ERDAT", "ERNAM", "DUEFL"],
    }},
    "CustomerType: {{
      "cypher": "MERGE (n:CustomerType {{ID: $concatenatedFields}} SET n += {{name: $concatenatedFields, KTOKD: $KTOKD, LIFNR: $LIFNR, ANRED: $ANRED, SPRAS: $SPRAS}}),
      "pk": "concatenatedFields",
      "fields": ["KTOKD", "LIFNR", "ANRED", "SPRAS"],
    }},
    "Extra" : {{
      "cypher": "MERGE (n:Extra {{KUNNR: $KUNNR}}) SET n += {{name: \"extra\", MANDT: $MANDT,  MCOD1: $MCOD1, MCOD2: $MCOD2, MCOD3: $MCOD3}}",
      "pk": "KUNNR",
      "fields": ["MANDT", "MCOD1", "MCOD2", "MCOD3"]
    }}
  }},
  "relationship_merge_templates": [
    {{
      "cypher": "MATCH (a:Customer {{KUNNR: $KUNNR}}) MATCH (b:Address {{ADRNR: $ADRNR}}) MERGE (a)-[:HAS_ADDRESS]->(b)",
      "from_key": "KUNNR",
      "to_key": "ADRNR"
    }},
    {{
      "cypher": "MATCH (a:Address {{ADRNR: $ADRNR}}) MATCH (b:City {{ORT01: $ORT01}}) MERGE (a)-[:IN_CITY]->(b)",
      "from_key": "ADRNR",
      "to_key": "ORT01"
    }},
    {{
      "cypher": "MATCH (a:City {{ORT01: $ORT01}}) MATCH (b:Country {{LAND1: $LAND1}}) MERGE (a)-[:PART_OF]->(b)",
      "from_key": "ORT01",
      "to_key": "LAND1"
    }},
    {{
      "cypher": "MATCH (a:Customer {{KUNNR: $KUNNR}}) MATCH (b:TaxInfo {{STCD2: $STCD2}}) MERGE (a)-[:HAS_TAX_INFO]->(b)",
      "from_key": "KUNNR",
      "to_key": "STCD2"
    }},
    {{
      "cypher": "MATCH (a:Customer {{KUNNR: $KUNNR}}) MATCH (b:Lifecycle {{ID: $concatenatedFields}}) MERGE (a)-[:HAS_LIFECYCLE]->(b)",
      "from_key": "KUNNR",
      "to_key": "concatenatedFields"
    }},
    {{
      "cypher": "MATCH (a:Customer {{KUNNR: $KUNNR}}) MATCH (b:CustomerType {{ID: $concatenatedFields}}) MERGE (a)-[:HAS_TYPE]->(b)",
      "from_key": "KUNNR",
      "to_key": "concatenatedFields"
    }},
    {{
      "cypher": "MATCH (a:Customer {{KUNNR: $KUNNR}}) MATCH (b:Extra {{KUNNR: $KUNNR}}) MERGE (a)-[:HAS_EXTRA]->(b)",
      "from_key": "KUNNR",
      "to_key": "KUNNR"
    }}
  ]
}}

**Respond ONLY with the JSON object, without any explanation or formatting (no markdown, no code blocks)**
"""

    response = model.invoke(cypher_prompt)
    try:
        parsed = json.loads(response.content)
        logger.info("Cypher template 생성 성공")
        return parsed
    except json.JSONDecodeError:
        logger.warning("LLM 응답이 JSON 형식이 아닙니다")
        return response




def generate_text2cypher_examples(schema: dict, model) -> list:
  prompt = f"""
  You are a GraphDB expert. Given the following schema: {schema}
  
  Generate example sets of user's questions and cypher queries that matches to the questions in following format:
  [
    "USER INPUT: ...., QUERY : .....",
    ...
  ]
  
  **IMPORTANT**
  The examples should include all possible paths that exist in the schema.
  So it should include 1-hop to N-hop questions and answers(cypher queries).
  
  """
  response = model.invoke(prompt)
  try:
    examples = json.loads(response.content)
    return examples
  except Exception as e:
    logger.error(f"Failed to generate examples from LLM : {str(e)}")
    return []



class GraphDBSchemaPipeline:
  def __init__(self, file_name: str, file: UploadFile):
    self.file_name = file_name
    self.file = file
    self.data_info = None
    self.schema = None
    self.retrieved_documents = None
    self.cypher_examples = None
    self.model = self.get_model()
    
  def get_model(self):
    try:
        model = ChatOpenAI(
            openai_api_key=config.OPEN_API_KEY_SERVER,
            model="gpt-4o",
            temperature=0,
            max_tokens=4048
        )
        logger.info("OpenAI 모델이 성공적으로 초기화되었습니다.")
        return model
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        raise e

  def run_graphDB_schema_generation_llm(self):
    try:
      self.data_info = get_data_info(self.file_name, self.file)
      self.retrieved_documents = search_related_ontologies(json.dumps(self.data_info))
      schema = create_graphdb_schema(self.model, self.data_info, self.retrieved_documents)
      return schema
    except Exception as e:
      logger.error(f"run_graphDB_schema_generation_llm() 파이프라인 실행 중 오류 : {str(e)}")
      raise e
  
  
  def run_regenerate_schema(self, schema: dict, user_edit_text: str):
    regenerated_schema = edit_graphdb_schema(self.model, schema, user_edit_text)
    self.schema = regenerated_schema
    return regenerated_schema
  
  def generate_cypher_templates(self):
    if self.schema is None:
      raise ValueError("schema가 비어있습니다")
    return create_cypher_template(self.model, self.schema)
  
  def generate_text2cypher_examples(self):
    if self.schema is None:
      raise ValueError("schema가 비어있습니다.")
    self.cypher_examples = generate_text2cypher_examples(self.schema, self.model)
    return self.cypher_examples
  
    



def run_graphDB_schema_generation_llm(
  file_name: Optional[str] = None, 
  file: Optional[UploadFile] = None,
  ):
    try:
        model = get_model()
        data_info = get_data_info(file_name, file)
        retrieved_documents = search_related_ontologies(json.dumps(data_info))
        graphdb_schema = create_graphdb_schema(model, data_info, retrieved_documents)
        return graphdb_schema
    except Exception as e:
        logger.error(f"GraphDB schema generation failed: {str(e)}")
        return {"error": str(e)}



def run_cypher_template_generation_llm(schema):
    model = get_model()
    cypher_template = create_cypher_template(model, schema)
    print(f">> Cypher Template: {cypher_template}")
    return cypher_template



if __name__ == "__main__":
    excel_path = "C:\\Users\\USER\\OneDrive\\Desktop\\KNA1_testfile.xlsx"

    #UploadFile 흉내
    with open(excel_path, "rb") as f:
        excel_bytes = f.read()
        file_like = BytesIO(excel_bytes)
        fake_upload_file = UploadFile(filename="Table_KNA1.XLSX", file=file_like)

    data_info = get_data_info(file_name="KNA1", file=fake_upload_file)
    print("Data Information: {data_info}")
    model = get_model()
    graphdb_scehma = create_graphdb_schema(model, data_info)
    print(f">> 스키마 : {graphdb_scehma}")
