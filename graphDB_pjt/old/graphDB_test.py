import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import logging
import json
import re
import pandas as pd
import uuid
import csv
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from db_to_graphDB.graphdb_llm import run_cypher_template_generation_llm, create_cypher_template

# Logging 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# Neo4j 연결
URI = config.NEO4J_URI
AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# 저장 실패 로그 파일 경로
FAILED_LOG_PATH = os.path.join(os.getcwd(), "graphdb_failed_rows.csv")

def extract_data_according_to_schema(schema):
    # schema가 문자열로 넘어올 수 있으므로 JSON 파싱
    if isinstance(schema, str):
        schema = json.loads(schema)

    used_columns = schema.get("used_fields", [])
    used_columns = [col.strip().upper() for col in used_columns]
    file_name = "KNA1"

    TMP_DIR = os.path.join(os.getcwd(), "tmp")
    xlsx_path = os.path.join(TMP_DIR, f"{file_name}_cleaned.xlsx")

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {xlsx_path}")

    # 문자열 강제 적용하여 읽기
    cleaned_df = pd.read_excel(xlsx_path, dtype=str)
    cleaned_df.columns = [col.strip().upper() for col in cleaned_df.columns]
    cleaned_df = cleaned_df.fillna("").applymap(str.strip)

    final_df = cleaned_df[used_columns].copy()
    #logger.info(f">>> {file_name} cleaned_df 엑셀 불러옴: {xlsx_path}")
    return cleaned_df



def get_parent_fields_for_node(schema: dict, current_node: str) -> list:
    # schema가 문자열로 넘어올 수 있으므로 JSON 파싱
    if isinstance(schema, str):
        schema = json.loads(schema)

    #schema의 relationship을 분석해 현재 노드의 상위 노드에서 연결된 property들을 추출
    parent_nodes = [
        rel["from"] for rel in schema.get("relationships", [])
        if rel["to"] == current_node
    ]

    # 상위 노드들의 node_properties를 모두 합침
    parent_fields = []
    for parent in parent_nodes:
        parent_props = schema.get("node_properties", {}).get(parent, [])
        parent_fields += parent_props

    return list(set(parent_fields))



def generate_fallback_identifier(row_dict:dict, field_list: list, separator: str = "_") -> str :
    """
    row_dict에서 field_list에 해당하는 컬럼 중 실제 값이 존재하는 것만 추출하여 이어붙이기
    """
    values = [
        str(row_dict.get(field, "")).strip()
        for field in field_list
        if str(row_dict.get(field, "")).strip().lower() not in ["", "nan", "none"]
    ]

    return separator.join(values) if values else str(uuid.uuid4())



def fill_missing_primary_keys_with_fallback(row_dict, node_properties, primary_keys, schema):
    """
    pk가 누락된 경우의 처리
    """
    transformed = row_dict.copy()

    for entity, props in node_properties.items():
        pk = primary_keys.get(entity)
        
        # 'pk'가 없거나, 'concat_fields'가 정의되어 있는 경우 'ID'를 기본 PK로 사용
        if "concat_fields" in (primary_keys.get(entity, {}) if isinstance(primary_keys.get(entity), dict) else {}):
            pk = "ID" # LLM의 응답에서 concat_fields가 있는 경우 pk를 "ID"로 설정함
            
        if not pk:
            continue

        pk_val = transformed.get(pk, "").strip()
        
        # 'concat_fields'가 존재하는 경우 해당 필드들을 조합하여 'ID' 값 생성
        if "concat_fields" in (primary_keys.get(entity, {}) if isinstance(primary_keys.get(entity), dict) else {}):
            concat_fields = primary_keys[entity].get("concat_fields", [])
            concatenated_value = generate_fallback_identifier(transformed, concat_fields)
            transformed[pk] = concatenated_value
            logger.info(f"[PK 대체 - Concatenated] {entity}.{pk} -> {concatenated_value}")
            
            # concat_fields로 생성된 ID를 name에도 할당
            transformed["name"] = concatenated_value
            logger.info(f"[name 대체 - Concatenated] {entity}.name → {concatenated_value}")

        elif pk_val == "" or pk_val.lower() == "nan":
            fallback_val = generate_fallback_identifier(transformed, props)
            transformed[pk] = fallback_val
            logger.info(f"[PK 대체] {entity}.{pk} -> {fallback_val}")

            # name도 무조건 fallback_val 또는 parent 필드 기반으로 설정
            parent_fields = get_parent_fields_for_node(schema, entity)
            name_val = generate_fallback_identifier(transformed, parent_fields) or fallback_val
            transformed["name"] = name_val
            logger.info(f"[name 대체] {entity}.name → {name_val}")
    
    return transformed


def safe_bindings(row_dict):
    """
    NaN, 빈 문자열, "nan" 문자열을 제거하여 실제 바인딩 가능한 값만 남긴다.
    """
    return {
        k: v for k, v in row_dict.items()
        if v is not None and str(v).strip() != "" and str(v).strip().lower() != "nan"
    }


def build_safe_set_clause(template_info: dict, bindings: dict):
    """
    SET n += {...} 구문을 안전하게 만들어줌 — 바인딩 없는 필드 제거
    template_info는 LLM 응답에서 얻은 노드 템플릿 정보 (cypher, pk, set_fields 등 포함)
    """
    template = template_info["cypher"]
    set_fields = template_info.get("set_fields", [])
    
    safe_template = template
    safe_bindings = bindings.copy()

    # MERGE (n:Label {pk: $pk}) 부분은 유지하고 SET n += {...} 부분만 처리
    # SET 절을 찾는 정규식 변경: SET n += { ... } 또는 SET n.prop = $param, n.prop2 = $param2
    # LLM이 SET n += {name: $name, prop: $prop} 이런 식으로 주기 때문에, 해당 패턴에 맞춤
    match = re.search(r"SET\s+n\s+\+=\s+{(.+?)}", template)
    
    if not match: # SET 절이 아예 없는 경우
        return safe_template, safe_bindings

    current_set_clause_content = match.group(1)
    
    # set_fields에 있는 필드들만 바인딩에 존재하는지 확인하여 유효한 SET 절 생성
    filtered_pairs = []
    for field in set_fields:
        # LLM이 name: $NAME1 같은 형식으로 주기 때문에, 여기서 $를 붙여줘야 함.
        # 그러나, concat_fields로 생성된 'ID'의 경우, 'name' 필드에 해당 값이 들어가므로 'ID' 자체를 바인딩 키로 사용
        if field == "ID" and "concat_fields" in template_info:
            if bindings.get("ID") is not None and str(bindings.get("ID")).strip().lower() != "nan":
                filtered_pairs.append(f"name: $ID") # name 필드는 ID 바인딩으로 설정
        elif field in bindings and bindings.get(field) is not None and str(bindings.get(field)).strip().lower() != "nan":
             # set_fields에 포함된 필드 중 바인딩에 값이 있는 것만 추가
            if field == "name" and "concat_fields" in template_info: # concat_fields가 있는 노드의 name 필드는 건너뜀 (ID로 처리됨)
                continue
            
            filtered_pairs.append(f"{field}: ${field}")

    if filtered_pairs:
        safe_set_content = ", ".join(filtered_pairs)
        # 기존 SET 절을 새롭게 구성된 SET 절로 대체
        safe_template = re.sub(r"SET\s+n\s+\+=\s+{[^}]*}", f"SET n += {{{safe_set_content}}}", template)
    else:
        # SET 절이 비어있으면 MERGE만 실행
        safe_template = re.sub(r"SET\s+n\s+\+=\s+{[^}]*}", "", template).strip()

    return safe_template, safe_bindings


def log_failure_to_csv(entity_type: str, mode: str, template: str, bindings: dict, error: str):
    """
    실패한 저장 데이터를 CSV로 기록 (append-only)
    """
    headers = ["mode", "entity_type", "template", "bindings", "error"]
    row = {
        "mode": mode,
        "entity_type": entity_type,
        "template": template,
        "bindings": json.dumps(bindings, ensure_ascii=False),
        "error": error
    }

    file_exists = os.path.exists(FAILED_LOG_PATH)
    with open(FAILED_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def patch_missing_primary_keys(df: pd.DataFrame, schema):
    """
    node_properties에 정의된 노드 중 primary_key가 누락된 경우,
    해당 노드가 사용할 수 있는 기본 컬럼이 존재하면 primary_key에 자동으로 설정

    현재는 'Extra' → 'KUNNR' 만 처리하지만, 확장 가능.
    """
    node_properties = schema.get("node_properties", {})
    primary_keys = schema.get("primary_key", {})

    default_key_map = {
        "Extra": "KUNNR",
    }

    for node, default_key in default_key_map.items():
        if node in node_properties and node not in primary_keys:
            if default_key in df.columns:
                primary_keys[node] = default_key
                #logger.info(f"[{node}] primary_key가 없어 기본값 '{default_key}'로 설정함")
    
    schema["primary_key"] = primary_keys
    return schema


def store_data_in_graphdb(df, node_templates, relationship_templates, node_properties_schema, primary_keys_schema):
    URI = config.NEO4J_URI
    AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            for _, row in df.iterrows():
                original_row = row.to_dict()
                
                # 'schema' 객체가 필요하므로, `fill_missing_primary_keys_with_fallback`에 전달할 스키마 재구성
                # patch_missing_primary_keys에서 업데이트된 primary_keys를 포함하도록 schema를 전달
                temp_schema = {
                    "node_properties": node_properties_schema,
                    "primary_key": primary_keys_schema,
                    "relationships": [] # 관계 정보는 현재 함수에서 사용하지 않으므로 빈 리스트
                }

                transformed_row = fill_missing_primary_keys_with_fallback(original_row, node_properties_schema, primary_keys_schema, temp_schema)
                bindings = safe_bindings(transformed_row)

                # 노드 저장
                for entity, template_info in node_templates.items():
                    template_str = template_info["cypher"] # 실제 Cypher 쿼리 문자열 추출
                    pk_field_from_template = template_info["pk"] # LLM에서 제시한 PK 필드명
                    concat_fields = template_info.get("concat_fields", [])

                    pk_val = transformed_row.get(pk_field_from_template, "")

                    # concat_fields가 있으면 해당 필드들을 조합하여 PK 값을 생성하고 바인딩에 추가
                    if concat_fields:
                        concatenated_id = generate_fallback_identifier(transformed_row, concat_fields)
                        bindings[pk_field_from_template] = concatenated_id # ex) ID: $concatenatedFields
                        # 'name' 필드에 'ID'와 동일한 값을 할당 (LLM 프롬프트에 정의된 방식)
                        bindings["name"] = concatenated_id 
                        pk_val = concatenated_id # PK 값도 업데이트
                    else:
                        # concat_fields가 없는 경우 기존 로직 유지 (PK 값 바인딩)
                        if pk_field_from_template and pk_field_from_template not in bindings:
                            bindings[pk_field_from_template] = pk_val

                    # PK 값 유효성 검사
                    if not pk_field_from_template or str(pk_val).strip() == "" or str(pk_val).strip().lower() == "nan":
                        logger.warning(f"[{entity}] PK({pk_field_from_template}) 값이 유효하지 않음 → 노드 MERGE 스킵. 값: '{pk_val}'")
                        log_failure_to_csv(entity, "node", template_str, bindings, f"Primary key '{pk_field_from_template}' invalid or missing → 값: '{pk_val}'")
                        continue

                    try:
                        safe_template, safe_bind = build_safe_set_clause(template_info, bindings) # template_info 전체를 전달
                        if safe_template.strip():
                            session.run(safe_template, safe_bind)
                        logger.info(f"[노드 저장 성공] {entity} → {safe_bind}")
                    except Exception as e:
                        logger.warning(f"[노드 저장 실패] {entity}: {e} → 바인딩: {bindings}")
                        log_failure_to_csv(entity, "node", safe_template, bindings, str(e))

                # 관계 저장
                for rel_info in relationship_templates:
                    rel_template_str = rel_info["cypher"] # 실제 Cypher 쿼리 문자열 추출
                    from_key = rel_info["from_key"]
                    to_key = rel_info["to_key"]

                    # 필수 바인딩 키 (from_key, to_key)가 존재하는지 확인
                    if from_key not in bindings or to_key not in bindings:
                        msg = f"관계 생성에 필요한 바인딩 키({from_key}, {to_key}) 누락 → 건너옴. 현재 바인딩: {bindings.keys()}"
                        logger.warning(f"[관계 MERGE] {msg}")
                        log_failure_to_csv("N/A", "relationship", rel_template_str, bindings, msg)
                        continue

                    try:
                        # 관계 템플릿은 SET 절처럼 동적으로 변경할 필요가 없으므로, LLM이 생성한 템플릿과 바인딩을 바로 사용
                        session.run(rel_template_str, bindings)
                        logger.info(f"[관계 저장 성공] {rel_template_str} → {bindings}")
                    except Exception as e:
                        logger.warning(f"[관계 저장 실패] {rel_template_str}: {e} → 바인딩: {bindings}")
                        log_failure_to_csv("N/A", "relationship", rel_template_str, bindings, str(e))

    return "GraphDB에 데이터 저장 완료"



def transform_data_to_graphdb(schema):
    # schema가 문자열로 넘어올 수 있으므로 JSON 파싱
    if isinstance(schema, str):
        schema_dict = json.loads(schema)
    else:
        schema_dict = schema
    
    filtered_df = extract_data_according_to_schema(schema_dict)
    print(f"filtered_df: {filtered_df.head()}")

    # LLM 호출하여 Cypher 템플릿 생성
    cypher_template = run_cypher_template_generation_llm(schema_dict)
    print(f">> Cypher Template: {json.dumps(cypher_template, indent=2, ensure_ascii=False)}")
    
    node_templates = cypher_template.get("node_merge_templates", {})
    relationship_templates = cypher_template.get("relationship_merge_templates", {})
    
    node_properties = schema_dict.get("node_properties", {})
    primary_keys = schema_dict.get("primary_key", {})

    # patch_missing_primary_keys를 호출하여 schema의 primary_key를 업데이트
    updated_schema = patch_missing_primary_keys(filtered_df, schema_dict)
    
    result = store_data_in_graphdb(filtered_df, node_templates, relationship_templates, updated_schema.get("node_properties", {}), updated_schema.get("primary_key", {}))
    return result



####################################################################################
if __name__ == "__main__":
  # 예시 스키마
  #schema =  {'table_name': 'KNA1', 'nodes': ['Customer', 'Address', 'Region', 'Country', 'TaxInfo', 'Lifecycle', 'Extra'], 'relationships': [{'from': 'Customer', 'to': 'Address', 'type': 'HAS_ADDRESS', 'properties': ['STRAS', 'PSTLZ', 'ORT01', 'ORT02']}, {'from': 'Address', 'to': 'Region', 'type': 'IN_REGION', 'properties': ['REGIO']}, {'from': 'Region', 'to': 'Country', 'type': 'PART_OF', 'properties': ['LAND1']}, {'from': 'Customer', 'to': 'TaxInfo', 'type': 'HAS_TAX_INFO', 'properties': ['STCD1', 'STCD2', 'STCD3', 'STCD5', 'STCEG']}, {'from': 'Customer', 'to': 'Lifecycle', 'type': 'HAS_LIFECYCLE', 'properties': ['ERDAT', 'ERNAM', 'LOEVM', 'AEDAT', 'USNAM']}], 'primary_key': {'Customer': 'KUNNR', 'Address': 'ADRNR', 'Region': 'REGIO', 'Country': 'LAND1', 'TaxInfo': 'KUNNR', 'Lifecycle': 'KUNNR'}, 'node_properties': {'Customer': ['KUNNR', 'NAME1', 'NAME2', 'SORTL', 'ANRED', 'KTOKD', 'LIFNR', 'SPRAS', 'DUEFL'], 'Address': ['ADRNR', 'STRAS', 'PSTLZ', 'ORT01', 'ORT02'], 'Region': ['REGIO'], 'Country': ['LAND1'], 'TaxInfo': ['STCD1', 'STCD2', 'STCD3', 'STCD5', 'STCEG'], 'Lifecycle': ['ERDAT', 'ERNAM', 'LOEVM', 'AEDAT', 'USNAM'], 'Extra': ['MANDT', 'MCOD1', 'MCOD2', 'MCOD3', 'BBBNR', 'BBSNR', 'BUBKZ', 'TELF1', 'TELFX', 'TELF2', 'LZONE', 'VBUND', 'DEAR2', 'UMSAT', 'UMJAH', 'JMZAH', 'JMJAH', 'STKZA', 'STKZN', 'UMSA1', 'WERKS', 'HZUOR', 'FITYP', 'CASSD', 'J_1KFREPRE', 'J_1KFTBUS', 'J_1KFTIND', 'UPTIM', 'DEAR6', 'RIC', 'LEGALNAT', '/VSO/R_PALHGT', '/VSO/R_I_NO_LYR', '/VSO/R_ULD_SIDE', '/VSO/R_LOAD_PREF', 'DUNS', 'J_1IEXRN', 'J_1IPANNO', 'PSPNR', 'J_3GSTDMON', 'J_3GSTDTAG', 'J_3GTAGMON', 'J_3GVMONAT', 'J_3GEMINBE', 'J_3GFMGUE', 'J_3GZUSCHUE']}}
  schema = '''
{
  "response": {
    "table_name": "KNA1",
    "nodes": [
      "Customer",
      "Address",
      "City",
      "Country",
      "TaxInfo",
      "Lifecycle",
      "CustomerType",
      "Extra"
    ],
    "relationships": [
      {
        "from": "Customer",
        "to": "Address",
        "type": "HAS_ADDRESS",
        "properties": [
          "STRAS",
          "PSTLZ"
        ]
      },
      {
        "from": "Address",
        "to": "City",
        "type": "IN_CITY",
        "properties": [
          "ORT01"
        ]
      },
      {
        "from": "City",
        "to": "Country",
        "type": "PART_OF",
        "properties": [
          "LAND1"
        ]
      },
      {
        "from": "Customer",
        "to": "TaxInfo",
        "type": "HAS_TAX_INFO",
        "properties": [
          "STCD2",
          "STCD5"
        ]
      },
      {
        "from": "Customer",
        "to": "Lifecycle",
        "type": "HAS_LIFECYCLE",
        "properties": [
          "ERDAT",
          "ERNAM",
          "DUEFL"
        ]
      },
      {
        "from": "Customer",
        "to": "CustomerType",
        "type": "HAS_TYPE",
        "properties": [
          "KTOKD"
        ]
      },
      {
        "from": "Customer",
        "to": "Extra",
        "type": "HAS_EXTRA",
        "properties": []
      }
    ],
    "primary_key": {
      "Customer": "KUNNR",
      "Address": "ADRNR",
      "City": "ORT01",
      "Country": "LAND1",
      "TaxInfo": "KUNNR",
      "Lifecycle": "KUNNR",
      "CustomerType": "KTOKD",
      "Extra": "KUNNR"
    },
    "node_properties": {
      "Customer": [
        "KUNNR",
        "NAME1",
        "NAME2",
        "SORTL",
        "MCOD1",
        "MCOD2",
        "MCOD3",
        "ANRED"
      ],
      "Address": [
        "ADRNR",
        "STRAS",
        "PSTLZ"
      ],
      "City": [
        "ORT01",
        "REGIO"
      ],
      "Country": [
        "LAND1"
      ],
      "TaxInfo": [
        "STCD2",
        "STCD5"
      ],
      "Lifecycle": [
        "ERDAT",
        "ERNAM",
        "DUEFL"
      ],
      "CustomerType": [
        "KTOKD"
      ],
      "Extra": [
        "MANDT",
        "BBBNR",
        "BBSNR",
        "BUBKZ",
        "UMSAT",
        "UMJAH",
        "JMZAH",
        "JMJAH",
        "UMSA1",
        "HZUOR",
        "J_1KFREPRE",
        "J_1KFTBUS",
        "J_1KFTIND",
        "UPTIM",
        "RIC",
        "LEGALNAT",
        "_VSO_R_PALHGT",
        "_VSO_R_I_NO_LYR",
        "_VSO_R_ULD_SIDE",
        "_VSO_R_LOAD_PREF",
        "PSPNR",
        "J_3GSTDMON",
        "J_3GSTDTAG",
        "J_3GTAGMON",
        "J_3GVMONAT",
        "J_3GEMINBE",
        "J_3GFMGUE",
        "J_3GZUSCHUE"
      ]
    },
    "used_fields": [
      "KUNNR",
      "NAME1",
      "NAME2",
      "SORTL",
      "MCOD1",
      "MCOD2",
      "MCOD3",
      "ANRED",
      "ADRNR",
      "STRAS",
      "PSTLZ",
      "ORT01",
      "REGIO",
      "LAND1",
      "STCD2",
      "STCD5",
      "ERDAT",
      "ERNAM",
      "DUEFL",
      "KTOKD"
    ]
  }
}'''
  # schema가 문자열로 넘어올 수 있으므로 JSON 파싱
  if isinstance(schema, str):
    schema_dict = json.loads(schema)
  else:
    schema_dict = schema

  filtered_df = extract_data_according_to_schema(schema_dict)
  print(f"filtered_df: {filtered_df.head()}")

  # LLM 호출하여 Cypher 템플릿 생성
  cypher_template = run_cypher_template_generation_llm(schema_dict)
  print(f">> Cypher Template: {json.dumps(cypher_template, indent=2, ensure_ascii=False)}")
    
  node_templates = cypher_template.get("node_merge_templates", {})
  relationship_templates = cypher_template.get("relationship_merge_templates", {})
    
  node_properties = schema_dict.get("node_properties", {})
  primary_keys = schema_dict.get("primary_key", {})

  # patch_missing_primary_keys를 호출하여 schema의 primary_key를 업데이트
  updated_schema = patch_missing_primary_keys(filtered_df, schema_dict)
    
  result = store_data_in_graphdb(filtered_df, node_templates, relationship_templates, updated_schema.get("node_properties", {}), updated_schema.get("primary_key", {}))
  print(result)

   