import os
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경변수 로드

OPEN_API_KEY_SERVER = os.getenv("OPEN_API_KEY_SERVER")

NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")
AURA_INSTANCEID=os.getenv("AURA_INSTANCEID")
AURA_INSTANCENAME=os.getenv("AURA_INSTANCENAME")


# Elasticsearch 관련 설정
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "infoactive_vdb")
OPEN_API_KEY = os.getenv("OPEN_API_KEY_TEST")

# 필드 이름
TEXT_FIELD = "text"
CONTENT_FIELD = "text"
VECTOR_FIELD = "vector"

# Vector DB 저장 경로
PERSIST_DIRECTORY = "inforactive_vdb"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# PostgreSQL DB Config
DB_NAME = "ontology_db"
DB_USER = "llm_user"
DB_PASSWORD = "1234"
DB_HOST = "192.168.1.154"
DB_PORT = 8004

# Optional: Create full SQLAlchemy-style URI if needed elsewhere
DB_CONNECTION = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


KNA1_NEO4J_SCHEMA = """
Node properties:
Customer {name: STRING, KUNNR: STRING, KTOKD: STRING, NAME1: STRING, NAME2: STRING, STORL: STRING, SPRAS: STRING, ANRED: STRING}
Address {name: STRING, REGIO: STRING, ORT01: STRING, ADRNR: STRING, PSTLZ: STRING}
City {name: STRING}
Country {name: STRING, REGIO: STRING}
TaxInfo {name: STRING, KUNNR: STRING, STCD2: STRING, STCD5: STRING}
Lifecycle {name: STRING, KUNNR: STRING, DUEFL: STRING, ERDAT: STRING, ERNAM: STRING}
Extra {}

Relationship properties:
LOCATED_IN {name: STRING}
HAS_TAX_INFO {name: STRING}
HAS_ADDRESS {name: STRING}
PART_OF {name: STRING}
HAS_EXTRA_INFO {name: STRING}

The relationships:
(:Customer)-[:HAS_ADDRESS]->(:Address)
(:Customer)-[:HAS_TAX_INFO]->(:TaxInfo)
(:Customer)-[:HAS_LIFECYCLE]->(:Lifecycle)
(:Customer)-[:HAS_EXTRA]->(:Extra)
(:Address)-[:IN_CITY]->(:City)
(:City)-[:PART_Of]->(:Country)
"""

KNA1_NEO4J_EXAMPLES = [
    "USER_INPUT: '주소가 서울인 고객들을 보여줘' QUERY: MATCH (c:Customer)-[ha:HAS_ADDRESS]->(a:Address)-[ic:IN_CTIY]->(ci:City) WHERE ci.name = '서울' RETURN c, ha, a, ic, ci",
    "USER_INPUT: '세금 정보 STCD2가 DE123456인 고객을 찾아줘' QUERY: MATCH (c:Customer)-[h:HAS_TAX_INFO]->(t:TaxInfo) WHERE t.STCD2 = 'DE123456' RETURN c, h, t",
    "USER_INPUT: '2024년에 생성된 고객 목록 보여줘' QUERY: MATCH (c:Customer)-[h:HAS_LIFECYCLE]->(l:Lifecycle) WHERE l.ERDAT STARTS WITH '2024' RETURN c, h, l",
    "USER_INPUT: '지역 코드가 01인 고객들의 이름과 우편번호를 보여줘' QUERY: MATCH (c:Customer)-[h:HAS_ADDRESS]->(a:Address)-[i:IN_CTIY]->(ci:City) WHERE ci.REGIO = '01' RETURN c, h, a, i, ci",
    "USER_INPUT: '서울에 거주하며 우편번호가 12345인 고객을 찾아줘' QUERY: MATCH (c:Customer)-[h:HAS_ADDRESS]->(a:Address)-[i:IN_CTIY]->(ci:City) WHERE ci.name = '서울' AND a.PSTLZ = '12345' RETURN c, h, a, i, ci",
    "USER_INPUT: '고객 이름과 해당 고객이 속한 국가명을 보여줘' QUERY: MATCH (c:Customer)-[h:HAS_ADDRESS]->(a:Address)-[i:IN_CTIY]->(ci:City)-[p:PART_OF]->(co:Country) RETURN c, h, a, i, ci, p, co",
    "USER_INPUT: '담당자가 SAPADMIN인 고객의 이름과 생성일을 보여줘' QUERY: MATCH (c:Customer)-[h:HAS_LIFECYCLE]->(l:Lifecycle) WHERE l.ERNAM = 'SAPADMIN' RETURN c, h, l",
    "USER_INPUT: '모든 고객의 이름과 고객 유형을 보여줘' QUERY: MATCH (c:Customer) RETURN c",
    "USER_INPUT: '국가가 한국인 고객을 찾아줘' QUERY: MATCH (c:Customer)-[h:HAS_ADDRESS]->(a:Address)-[i:IN_CTIY]->(ci:City)-[p:PART_OF]->(co:Country) WHERE co.LAND1 = 'KR' RETURN c, h, a, i, ci, p, co",
    "USER_INPUT: '모든 고객과 그들의 주소를 보여줘' QUERY: MATCH (c:Customer)-[h:HAS_ADDRESS]->(a:Address) RETURN c, h, a"
]


KNA1_CYPHER_TEMPLATES = """
Task: Generate a VALID Cypher statement for querying a Neo4j graph database from a user input.
---
You are a Neo4j Cypher expert.

You are given a schema for a knowledge graph and your task is to convert natural language questions into valid Cypher queries using this schema. You MUST follow the schema exactly as defined.

Do NOT invent or rename any property names, node labels, or relationship types.

---
Example paths generated in graphDB:
(:Customer)-[:HAS_ADDRESS]->(:Address)  
(:Address)-[:IN_CITY]->(:City)  
(:City)-[:PART_OF]->(:Country)  
(:Customer)-[:HAS_TAX_INFO]->(:TaxInfo)  
(:Customer)-[:HAS_LIFECYCLE]->(:Lifecycle)  
(:Customer)-[:HAS_EXTRA]->(:Extra)
---
Schema:
{schema}

Examples:
{examples}

Input:
{query_text}

Do not use any properties or relationships not included in the scehma.
Match and return relevant variables and properties clearly.
Do NOT include any explanations or comments — only return the Cypher query.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Cypher query:
"""


SCHEMA_LINK_INFO = """"

- bom.Parent ≡ mast.MATNR: 완제품(Parent)이 MAST의 제품 자재 번호(MATNR)와 의미적으로 동일하며, BOM 연결의 출발점입니다.

- mast.STLNR ≡ stpo.STLNR: BOM 번호(STLNR)를 기준으로 구성 자재(STPO.IDNRK)를 조회할 수 있습니다.

- stpo.IDNRK ≡ bom.Child: 구성 자재(IDNRK)는 BOM 테이블에서 구성품(Child)과 동일한 자재를 의미합니다.

- bom.Child ≡ eina.MATNR: 구성품은 EINA의 자재 번호(MATNR)와 의미적으로 같으며, 공급처 정보 조회에 사용됩니다.

- eina.LIFNR ≡ lfa1.LIFNR: 공급처 ID(LIFNR)는 LFA1에서 위치(ORT01) 정보를 조회하는 데 사용됩니다.

- bom.Parent ≡ custom_sales_plan.MATNR: 완제품은 판매계획 테이블(custom_sales_plan)의 MATNR과 동일한 제품입니다.

"""


NEO4J_SCHEMA ="""
[
        "table_name": "MAST_STPO_BOM",
        "schema_explanation": "This schema represents a Bill of Materials (BOM) hierarchy using data from SAP's MAST and STPO tables. All materials—whether they are final products, semi-finished, or components—are represented as nodes with the label 'Material'. Each node is uniquely identified by its material number (MATNR). The functional role (parent or component) is inferred from the relationship context, not the node label. Nodes are connected using HAS_COMPONENT relationships, which describe assembly structure and include metadata such as quantity (MENGE) in the BOM.",
        "nodes": [
            "label": "Material",
            "primary_key": "MATNR",
            "description": "Represents any material in the BOM hierarchy—whether it acts as a parent (assembly) or component. This includes raw materials, semi-finished products, and final goods. All nodes are labeled 'Material', and a single material can act both as a parent and as a component depending on the BOM context.",
            "properties": ["MATNR", "STLNR"]
          ],
        ],
        "relationships": [
        [
            "(parents:Material)-[:HAS_COMPONENT]->(component:Material) "
            "type": "HAS_COMPONENT" ,
            "properties": ["MENGE"],
            "description": "Represents a directed relationship where a parent material (assembly) includes a component material in its structure. The relationship models BOM-specific metadata: the required unit quantity (MENGE) for the parent material. This relationship allows for recursive material decomposition across multiple levels. Becareful of the direction."
        ]
        ],
        "used_fields": [
        "MATNR",
        "STLNR",
        "MENGE"
        ]
    ]
"""


NEO4J_EXAMPLES = [
  "USER_INPUT: 'BoM 넘버가 41일 때 어떤 구성품들로 이루어져 있어?' QUERY: MATCH (parent:Material {{STLNR: '41'}})-[:HAS_COMPONENT]->(child:Material) RETURN child.MATNR AS Component",

  "USER_INPUT: '46번 BOM 구조와 각 구성품의 수량을 보여줘.' QUERY: MATCH (parent:Material {{STLNR: '46'}})-[r:HAS_COMPONENT]->(child:Material) RETURN child.MATNR AS Component, r.MENGE AS Quantity, r.MEINS AS Unit ORDER BY r.STPOZ",

  "USER_INPUT: 'GE612B2000 자재가 어떤 BoM 넘버에 포함되는지 알려줘.' QUERY: MATCH (parent:Material)-[r:HAS_COMPONENT]->(child:Material {{MATNR: 'GE612B2000'}}) RETURN parent.STLNR AS BOMNumber, r.MENGE AS Quantity",

  "USER_INPUT: '100번 BoM 전체 트리를 그래프로 보여줘.' QUERY: MATCH path = (parent:Material {{STLNR: '100'}})-[:HAS_COMPONENT*]->(child:Material) RETURN path",
  
  "USER_INPUT: 'K300987번 재료를 쓰는 제품을 보자.' QUERY: MATCH path=(parent:Material) -[:HAS_COMPONENT*]-> (child:Material {{MATNR:'K300987'}}) WHERE NOT (child)-[:HAS_COMPONENT]->() RETURN path",

  "USER_INPUT: '최상위 완제품 리스트를 알려줘.' QUERY: MATCH (parent:Material) WHERE NOT (parent)<-[:HAS_COMPONENT]-(:Material) RETURN parent",

  "USER_INPUT: 'K300987,K300989 재료를 쓰는 최상위 제품을 알려줘.' QUERY: MATCH (parent:Material)-[:HAS_COMPONENT*]->(child:Material) WHERE child.MATNR IN ['K300987', 'K300989'] AND NOT (parent)<-[:HAS_COMPONENT]-(:Material) RETURN DISTINCT parent.MATNR AS TopLevelProduct, child.MATNR AS UsedMaterial",

]

EINA_SCHEMA = """
This table shows the mapping between materials (matnr) and their corresponding suppliers (lifnr).
최하위 자재만 존재. 반제품과 완제품은 공급업체번호가 존재하지 않음.
[
  "matnr: Material Number",
  "lifnr: Vendor Number (LFA1)",
]
"""
LFA1_SCHEMA = """
This table shows the supplier numbers (lifnr) and their corresponding city locations (ort01).
[
  "lifnr: Vendor Number (Primary Key)",
  "ort01: City(only in English)"
]
"""
CUSTOM_SALES_PLAN_SCHEMA = """
This table presents planned sales data by material (matnr), year (year), and quarter (quarter), including quantity (quantity), unit price (unit price), and total amount (amount).
[
  "matnr: Material Number",
  "year: Fiscal Year",
  "quarter: Fiscal Quarter (Q1 ~ Q4)",
  "quantity: Sales Quantity (Units)",
  "unitPrice: Price per Unit (dollar)",
  "amount: Total Sales Amount (Quantity × Unit Price)"
]
"""
ONTOLOGY_SCHEMA = """
GraphDB: MAST_STPO_BOM
- Each `Material` node may have a `:HAS_COMPONENT` relationship to another `Material` node, representing component hierarchy.
- `parent:Material` has `MATNR` and links to `child:Material` via `HAS_COMPONENT`, with relationship property `MENGE` (quantity).
- MENGE is the unit quantity of the component. So the total quantity is the quantity of the parent material multiplied by the MENGE.

RDB Tables:
- `EINA`: purchasing info; each component (material) may be found here via `MATNR`.
- `LFA1`: vendor info table; connected via `LIFNR` from `EINA`, includes `ORT01` (vendor city).
- `CUSTOM_SALES_PLAN`: future sales plans by material (`MATNR`), quarter, and year.


Relationships:
- custom_sales_plan.matnr → eina.matnr
- custom_sales_plan.matnr → Neo4j Material.matnr
- eina.lifnr → lfa1.lifnr
- eina.matnr → custom_sales_plan.matnr
- eina.matnr → Neo4j Material.matnr
"""



CYPHER_TEMPLATES = """
Task: Generate a VALID Cypher statement for querying a Neo4j graph database from a user input.
---
You are an expert in generating Cypher queries for a Neo4j graph database that stores Bill of Materials (BoM) data.
Make sure to name the path variable as path and return path
---

Schema:
{schema}

Examples:
{examples}

Input:
{query_text}


Do not use any properties or relationships not included in the schema.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

if you 
Cypher query:
"""


SQL_TEMPLATES = """
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Given an input question, convert it to a syntactically correct SQL query.

Input:
{input}

SQL:

"""

AGENT_EXAMPLE="""
Question: How many people are in the database and can you give me some examples?

Thought: I should first check how many nodes match this query.
Action: text2Cypher
Action Input: Find the people node and count
Observation: 3,248

Thought: This is a large result set. I will retrieve only a sample.
Action: text2Cypher
Action Input: Find the people node and limit 20
Observation: 
[1] <Record People='JAMES'>
[2] <Record People='ANDREW'>
[3] <Record People='LILY'>
...
[20] <Record People='ANNA'>
Thought: I now know the final answer
Final Answer: There are 3,248 people in the database. Here are 20 sample entries with their names and ages.



Question: If there is a flood in Daejeon in August, how would the sales plan be affected?
Thought: 
Find suppliers located in Daejeon (ort01 = 'daejeon') using RDB.
Identify the materials (matnr) those suppliers supply via eina using RDB
To find the indirectly affected materials (matnr), I will query the graph database.
Use those material (matnr) to query the sales plan in Q3 from custom_sales_plan in RDB
Action: executeSQL
Action Input: SELECT eina.matnr FROM eina JOIN lfa1 ON eina.lifnr = lfa1.lifnr WHERE lfa1.ort01 = 'Daejeon'
Observation:
[('MAT0053',), ('MAT0072',), ('MAT0084',), ('MAT0095',), ('MAT0100',), ('MAT0101',), ('MAT0184',), ('MAT0182',), ('MAT0194',), ('MAT0183',)]
Thought: I have identified the materials supplied by vendors in Daejeon. Now, I will check the parents material for these materials in MAST_STPO_BOM data. I will query the graph database. This will help identify indirectly affected products that use these components. 
Action: text2Cypher
Action Input: What are the most parent materials that use the following components:
['MAT0053', 'MAT0072', 'MAT0084', 'MAT0095', 'MAT0100', 'MAT0101', 'MAT0184', 'MAT0182', 'MAT0194', 'MAT0183']?
Observation: 
[1] <Record m.MATNR='MAT0001'>
[2] <Record m.MATNR='MAT0003'>
Thought:
Now I know which parent materials might be indirectly affected by supply disruptions in Daejeon.
I will query the sales plan for these parent materials in Q3 to understand the potential impact.
Action: executeSQL
Action Input: SELECT * FROM custom_sales_plan WHERE matnr IN ('MAT0001', 'MAT0003') AND UPPER(quarter) = 'Q3'
Observation:
[
    ('MAT0001', 2025, 'Q3', 1452, 285, 413820),
    ('MAT0001', 2026, 'Q3',  699, 297, 207603),
    ('MAT0003', 2025, 'Q3',  574, 166,  95284),
    ('MAT0003', 2026, 'Q3', 1376, 138, 189888)
]
Thought:
I now know the final answer.
Final Answer:
If there is a flood in Daejeon in August, it may disrupt the supply of materials from vendors in that region.
('MAT0001', 2025, 'Q3', 1452, 285, 413820),
('MAT0001', 2026, 'Q3',  699, 297, 207603),
('MAT0003', 2025, 'Q3',  574, 166,  95284),
('MAT0003', 2026, 'Q3', 1376, 138, 189888)
These materials are used as components in higher-level products, which we identified via the BOM structure in the graph database.
The sales plans for those parent materials in Q3 could be delayed or impacted, particularly in August.
Thus, both direct and indirect disruptions in the supply chain would likely affect the planned sales volumes and revenues for those products.

"""
STPO_SCHEMA = """
[
  "STLNR: Bill of Material",
  "STPOZ: BOM Item Counter (Internal Number)",
  "IDNRK: BOM Component (Material Number)",
  "MEINS: Base Unit of Measure (T006)",
  "MENGE: Quantity of Component"
]
"""

MAST_SCHEMA = """
[
  "MANDT: Client (T000)",
  "MATNR: Material Number (MARA)",
  "WERKS: Plant (T001W)",
  "STLAN: BOM Usage (T416)",
  "STLNR: Bill of Material",
  "STLAL: Alternative BOM",
  "LOSVN: From Lot Size",
  "LOSBS: To Lot Size",
  "ANDAT: Date Record Created On",
  "ANNAM: User who created record",
  "AEDAT: Last Changed On",
  "AENAM: Name of Person Who Changed Object",
  "CSLTY: Indicator: configured material (material variant)",
  "MATERIAL_BOM_KEY: Material BOM Concatenated Key"
]
"""

ONTOLOGY_INSTRUCTION = """
You are a schema reasoning assistant.

Given a user's question and a list of available graph schemas (each with nodes, relationships, and descriptions), your job is :
- to select **all schemas that are needed** to answer the question. 
- explain how they are connected together to form a logical reasoning chain

---

### What you must do:

1. Understand the user's question and break it down into the key data elements needed to answer it.
2. **Determine which schemas are requried,** including those that provide:
  - Final answers
  - Intermediate values necessary to reach them
3. **Describe the reasoning chain**, step by step:
  - What data is needed at each step
  - Which schema provides it
  - How outputs from one schema connect to inputs in the next
  - Include any field mappings where names differ but meanings align
---

### Output Format (in natural language, not JSON):

**Required schemas (in order):**  
- FirstSchemaName  
- SecondSchemaName  
- ThirdSchemaName  
...

**Reasoning chain:**  
1. First, use `FirstSchemaName` to find X (e.g., the components of a product).  
2. Then, map the value `component_id` from `FirstSchemaName` to `material_id` in `SecondSchemaName`, which allows you to find Y (e.g., which plant produces it).  
3. Finally, use `ThirdSchemaName` to find the location of the plant using `plant_id`.  

Be precise and include field name mappings if relevant.

"""

ONTOLOGY_EXAMPLE = """
## ProductStructure
- Material {{parent_id: "m1", material_id: "m1", name: "Cake"}}
- Material {{component_id: "m2", name: "Flour"}}
- Material {{component_id: "m3", name: "Egg"}}
- Material {{component_id: "m4", name: "Whipped Cream"}}
- Material {{component_id: "m5", name: "Milk"}}

- MADE_OF {{from: "parent_id:m1", to: "component_id:m2"}}
- MADE_OF {{from: "parent_id:m1", to: "component_id:m3"}}
- MADE_OF {{from: "parent_id:m1", to: "component_id:m4"}}
- MADE_OF {{from: "component_id:m4", to: "component_id:m5"}}

## ProductSupplier
- Material {{material_id: "m2", name: "Flour"}}
- Material {{material_id: "m3", name: "Egg"}}
- Material {{material_id: "m4", name: "Whipped Cream"}}
- Material {{material_id: "m5", name: "Milk"}}
- Plant {{plant_id: "plnt1", name: "CJ"}}
- Plant {{plant_id: "plnt2", name: "Otoki"}}
- Plant {{plant_id: "plnt3", name: "Lotte"}}

- PRODUCED_IN {{from: "material_id:m2", to: "plant_id:plnt1"}}
- PRODUCED_IN {{from: "material_id:m3", to: "plant_id:plnt2"}}
- PRODUCED_IN {{from: "material_id:m4", to: "plant_id:plnt3"}}
- PRODUCED_IN {{from: "material_id:m5", to: "plant_id:plnt3"}}

## PlantLocation
- Plant {{plant_id: "plnt1", name: "CJ"}}
- Plant {{plant_id: "plnt2", name: "Otoki"}}
- Plant {{plant_id: "plnt3", name: "Lotte"}}
- City {{city_id: "c1", name: "Seoul"}}
- City {{city_id: "c2", name: "Busan"}}
- City {{city_id: "c3", name: "Ulsan"}}
- City {{city_id: "c4", name: "Gumi"}}

- LOCATED_IN {{from: "plant_id:plnt1", to: "city_id:c4"}}
- LOCATED_IN {{from: "plant_id:plnt2", to: "city_id:c3"}}
- LOCATED_IN {{from: "plant_id:plnt3", to: "city_id:c1"}}
- LOCATED_IN {{from: "plant_id:plnt3", to: "city_id:c2"}}

## GeoLocation
- City {{city_id: "c1"}}
- City {{city_id: "c2"}}
- City {{city_id: "c3"}}
- City {{city_id: "c4"}}
- City {{city_id: "c5"}}
- Country {{country_id: "k1"}}
- Country {{country_id: "k2"}}

- PART_OF {{from: "city_id:c1", to: "country_id:k1"}}
- PART_OF {{from: "city_id:c2", to: "country_id:k1"}}
- PART_OF {{from: "city_id:c3", to: "country_id:k1"}}
- PART_OF {{from: "city_id:c4", to: "country_id:k1"}}
- PART_OF {{from: "city_id:c5", to: "country_id:k2"}}

## PersonalNetwork
- Person {{person_id: "p1", id: "p1", name: "Amy", job: "Engineer"}}
- Person {{person_id: "p2", id: "p2", name: "Bob", job: "Designer"}}
- Person {{person_id: "p3", id: "p3", name: "Charlie", job: "Chef"}}
- Person {{person_id: "p4", id: "p4", name: "Diana", job: "Manager"}}
- Hobby {{hobby_id: "h1", name: "Climbing"}}
- Hobby {{hobby_id: "h2", name: "Cooking"}}
- Address {{address_id: "a1", city_id: "c1", name: "Seoul"}}
- Address {{address_id: "a2", city_id: "c3", name: "Ulsan"}}
- Address {{address_id: "a3", city_id: "c5", name: "Osaka"}}

- FRIEND_OF {{from: "person_id:p1", to: "person_id:p2"}}
- FRIEND_OF {{from: "person_id:p2", to: "person_id:p4"}}
- COWORKER_OF {{from: "person_id:p2", to: "person_id:p3"}}
- RELATIVE_OF {{from: "person_id:p1", to: "person_id:p3"}}
- FAMILY_OF {{from: "person_id:p3", to: "person_id:p4"}}
- ENJOYS {{from: "person_id:p1", to: "hobby_id:h1"}}
- ENJOYS {{from: "person_id:p3", to: "hobby_id:h2"}}
- LIVES_IN {{from: "person_id:p1", to: "address_id:a1"}}
- LIVES_IN {{from: "person_id:p2", to: "address_id:a1"}}
- LIVES_IN {{from: "person_id:p3", to: "address_id:a2"}}
- LIVES_IN {{from: "person_id:p4", to: "address_id:a3"}}

## EnergyUsagePlan
- Material {{material_id: "m1", name: "Cake"}}
- TimePeriod {{period_id: "tp1", year: "2025", quarter: "Q1"}}
- UsagePlan {{plan_id: "plan1", kwh: 1200, unit_cost: 0.2, estimated_cost: 240}}

- HAS_USAGE_PLAN {{from: "material_id:m1", to: "plan_id:plan1"}}
- FOR_PERIOD {{from: "plan_id:plan1", to: "period_id:tp1"}}
"""

APPLIED_EXAMPLE = """
Q1. 무슨 지역에 홍수가 나면 Cake라는 완성품을 위한 공급에 차질이 생길 수 있나요?

Answer :
**Required schemas (in order):**
- ProductStructure  
- ProductSupplier  
- PlantLocation  
- GeoLocation

**Reasoning chain:**  
1. First, use `ProductStructure` to find which sub-materials (components) are used to make the product `"Cake"`.  
   → This is done by traversing `MADE_OF` relationships from `parent_id = m1`.

2. Then, map each `component_id` from `ProductStructure` to `material_id` in `ProductSupplier`.  
   → This lets us find the `plant_id` where each component is produced via the `PRODUCED_IN` relationship.

3. Next, use `PlantLocation` to find which `city_id` each `plant_id` is located in via the `LOCATED_IN` relationship.

4. Finally, use `GeoLocation` to map each `city_id` to its `country_id`, which allows us to understand the regional scope of supply risk (e.g., flood-prone countries or cities).

-----
Q2. Bob과 가족 관계에 있는 사람이 즐기는 취미가 Cooking일 때, 그 사람이 사는 지역의 공장에서 생산되는 원재료가 들어간 완제품은 무엇인가요?

**Required schemas (in order):**
- PersonalNetwork  
- ProductSupplier  
- PlantLocation  
- ProductStructure

**Reasoning chain:**  
1. First, use `PersonalNetwork` to find Bob (`person_id:p2`) and trace all people who are `FAMILY_OF` or `RELATIVE_OF` Bob.  
   → This leads to `person_id:p3` (Charlie).

2. Still in `PersonalNetwork`, check if that person (Charlie) has the hobby `Cooking` via the `ENJOYS` relationship.  
   → Confirmed: `person_id:p3` → `hobby_id:h2` (Cooking)

3. Use the `LIVES_IN` relationship in `PersonalNetwork` to get the `address_id` of Charlie.  
   → Charlie lives at `address_id:a2`, which has `city_id:c3`.

4. Use `PlantLocation` to find all plants located in that same `city_id:c3`.  
   → `plant_id:plnt2` is located in Ulsan.

5. Use `ProductSupplier` to find materials that are `PRODUCED_IN` `plant_id:plnt2`.  
   → This gives `material_id:m3` (Egg)

6. Use `ProductStructure` to find any `parent_id` (product) that is `MADE_OF` the given `component_id = m3`.  
   → `m1 (Cake)` is composed of `m3 (Egg)`

7. Therefore, the final product affected is `"Cake"` — it contains raw material(s) produced in the region where Charlie lives and works.
"""