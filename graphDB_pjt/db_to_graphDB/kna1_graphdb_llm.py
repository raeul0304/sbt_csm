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

def get_model():
    try:
        model = ChatOpenAI(
            openai_api_key=config.OPEN_API_KEY_TEST,
            model="gpt-4o",
            temperature=0,
            max_tokens=4048
        )
        logger.info("OpenAI 모델이 성공적으로 초기화되었습니다.")
        return model
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        raise e


def get_data_info(file_name, file: UploadFile):
    table_name = file_name
    table_description = "customer master data in SAP system"

    try:
        file.file.seek(0)
        df = pd.read_excel(BytesIO(file.file.read()), dtype=str)
        df.columns = [col.strip().upper() for col in df.columns]
        df = df.fillna("").applymap(str.strip)
    except Exception as e:
        logger.error(f"엑셀 파일 로딩 실패: {str(e)}")
        raise e
    
    cleaned_df = preprocess_data(df)

    #엑셀 파일로 저장
    xlsx_path = os.path.join(TMP_DIR, f"{file_name}_cleaned.xlsx")
    cleaned_df.to_excel(xlsx_path, index=False)
    logger.info(f"전처리된 cleaned_df 저장 완료 (.xlsx): {xlsx_path}")

    fields = list_preprocessed_columns(cleaned_df)
    sample_rows = cleaned_df.head(10).to_dict(orient='records')

    data_info = {
        "table_name": table_name,
        "table_description": table_description,
        "fields": fields,
        "sample_rows": sample_rows
    }
    logger.info("Data information successfully extracted")

    return data_info
    

def create_graphdb_schema(model, data_info):
    required_keys = ['table_name', 'table_description', 'fields', 'sample_rows']
    
    for key in required_keys:
        if key not in data_info:
            raise ValueError(f"Missing required key in data_info: {key}")
    
    try:
        # sample_rows 문자열화
        sample_str = json.dumps(data_info['sample_rows'], indent=2)

        schema_prompt = f"""
You are a data modeling expert with deep understanding of SAP systems and business processes. Your task is to analyze the following SAP table and design a comprehensive and semantically meaningful GraphDB schema.

Table Name: {data_info['table_name']}
Description: {data_info['table_description']}
Fields: {', '.join(data_info['fields'])}
Sample Rows:
{sample_str}

Your task consists of the following steps:

---

1. **Identify All Business Purposes / Usage Scenarios**

Analyze the table and infer all distinct usage contexts in which this data may be involved. These may include, but are not limited to:

- Customer master data management
- Contact and address resolution
- Location classification and hierarchy
- Tax/identification tracking
- Lifecycle tracking (created by, date, flags)
- Customer segmentation by geography, type, or status

Be creative in identifying usage contexts. **You are not limited to one.**

---

2. **Extract Nodes, Relationships, and Properties per Purpose**

> For each purpose you identify, extract:
- Entities (nodes)
- Relationships (edges)
- Properties (fields as attributes)

> You may not rename the fields, but you may rename the entities and relationships to be more semantically menaingful.
So properties' names should be the same as the original field names.

> Also consider structural/semantic patterns.

▶ Example (one of many possible usages — do not limit yourself to this):

When the data is used for organizing customers by location purpose(the relation you must include at least):
- Nodes: Customer, Address, City, Country
- Relationships:
  - (:Customer)-[:HAS_ADDRESS]->(:Address)
  - (:Address)-[:IN_CITY]->(:City)
  - (:Region)-[:PART_OF]->(:Country)

This example demonstrates **hierarchical geographic modeling**. You should look for **other patterns** such as authorship, categorization, or workflow tracking.

---

3. **Design the Full Graph Schema**
Return the following fields:
- nodes
- relationships (with direction and property names)
- node_properties
- primary_key
> Do not use the same fields as properties in different nodes when constructing the schema
> When there's no appropriate field for primary key for certain node, set its primary key as "concatenatedFields" so that the primary key can be generated with combined fields.
---

4. **Create 'Extra' Node for the fields that are not used in the graph schema**
Include 'Extra' node for fields that are not used in any purpose. This node should contain all unused fields as properties.
Always include relationship between 'Extra' and the main entities(e.g, when data is about customers, 'Extra' should be connected to 'Customer' node).
---

5. **Output Format**
Return the result as **valid JSON only**, following this structure:

{{
  "table_name": "{data_info['table_name']}",
  "nodes": ["EntityA", "EntityB", "EntityC", "EntityD", "Extra"],
  "relationships": [
    {{
      "from": "EntityA",
      "to": "EntityB",
      "type": "RELATION_TYPE",
      "properties": ["field3"]
    }}
  ],
  "primary_key": {{
    "EntityA": "concatenatedFields",
    "EntityB": "field1",
    "EntityC": "field9",
    "EntityD": "field10",
    "Extra": "concatenatedFields"
  }},
  "node_properties": {{
    "EntityA": ["name", "field2", "field4", "field11", "field12"],
    "EntityB": ["name", "field1", "field8", "field3"],
    "EntityC": ["name", "field9],
    "EntityD": ["name", "field6", "field7"],
    "Extra" : ["name", "UnusedField1", "UnusedField2"]
  }}
  "used_fields" : ["UsedField1", "UsedField2", "UsedField3"]
}}

---

Guidelines:
- Be exhaustive — uncover **all meaningful entities and connections** embedded in the dataset.
- Do not limit yourself to one business purpose.
**Respond ONLY with the JSON object, without any explanation or formatting (no markdown, no code blocks)**
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



def run_cypher_template_generation_llm(schema):
    model = get_model()
    cypher_template = create_cypher_template(model, schema)
    print(f">> Cypher Template: {cypher_template}")
    return cypher_template


def run_graphDB_schema_generation_llm(file_name, file):
    try:
        model = get_model()
        data_info = get_data_info(file_name, file)
        graphdb_schema = create_graphdb_schema(model, data_info)
        return graphdb_schema
    except Exception as e:
        logger.error(f"GraphDB schema generation failed: {str(e)}")
        return {"error": str(e)}


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
