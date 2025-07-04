import config
import logging
import json
import re
import os
import pandas as pd
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

def extract_data_according_to_schema(schema):
    used_columns = schema.get("used_fields", [])
    file_name = "KNA1" #schema.get("table_name")

    # 현재 작업 디렉토리 기준 tmp 디렉토리 경로
    TMP_DIR = os.path.join(os.getcwd(), "tmp")
    os.makedirs(TMP_DIR, exist_ok=True)

    # 저장된 cleaned_df 불러오기
    pickle_path = os.path.join(TMP_DIR, f"{file_name}_cleaned.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle 파일을 찾을 수 없습니다: {pickle_path}")
    
    cleaned_df = pd.read_pickle(pickle_path)
    cleaned_df.columns = [col.strip().upper() for col in cleaned_df.columns]
    logger.info(f">>> {file_name} cleaned_df 불러옴: {pickle_path}")

    final_df = cleaned_df[used_columns].copy()
    return cleaned_df



def transform_row_fields(row_dict, field_to_graph_map):
    transformed = {}
    for excel_col, value in row_dict.items():
        if excel_col in field_to_graph_map:
            mapped_property = field_to_graph_map[excel_col].get("property")
            if mapped_property and pd.notnull(value) and str(value).strip() != "":
                if mapped_property in transformed:
                    logger.debug(f"[중복매핑] {excel_col} → {mapped_property} (기존 값 덮어씀)")
                transformed[mapped_property] = value
    return transformed


import uuid


def ensure_uuid_for_missing_keys(bindings, template):
    #Cypher 템플릿에서 사용된 변수 중 bindings에 없는 키가 있고, 해당 키가 ID 또는 유사 식별자일 경우 UUID를 생성하여 추가한다.
    required_keys = set(re.findall(r"\$([a-zA-Z0-9_]+)", template))

    for key in required_keys:
        if key not in bindings:
            if key.lower().endswith(("id", "_id", "code", "key")):
                bindings[key] = str(uuid.uuid4())
                logger.info(f"[UUID 자동 생성] {key} → {bindings[key]}")

    return bindings



def store_data_in_graphdb(df, node_templates, relationship_templates, field_to_graph_map, primary_keys):
    # Neo4j 연결 설정
    URI = config.NEO4J_URI
    AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

    def safe_bindings(row_dict):
        # 값이 비어있지 않은 것만 추출
        return {k: v for k, v in row_dict.items() if pd.notnull(v) and str(v).strip() != ""}

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            for _, row in df.iterrows():
                original_row = row.to_dict()
                transformed_row = transform_row_fields(original_row, field_to_graph_map)
                bindings = safe_bindings(transformed_row)

                # 노드 저장
                for entity, template in node_templates.items():
                    bindings_with_uuid = ensure_uuid_for_missing_keys(bindings.copy(), template)

                    pk = primary_keys.get(entity)
                    if not pk or pk not in bindings_with_uuid:
                        logger.warning(f"[{entity}] PK({pk}) 누락 → 노드 MERGE 스킵")
                        continue

                    try:
                        session.run(template, bindings_with_uuid)
                        logger.info(f"[노드 저장 성공] {entity} → {bindings_with_uuid}")
                    except Exception as e:
                        logger.warning(f"[노드 저장 실패] {entity}: {e} → 바인딩: {bindings_with_uuid}")

                # 관계 저장
                for rel_template in relationship_templates:
                    rel_bindings = ensure_uuid_for_missing_keys(bindings.copy(), rel_template)
                    required_keys = re.findall(r"\$([a-zA-Z0-9_]+)", rel_template)

                    if len(required_keys) < 2:
                        logger.warning(f"[관계 MERGE] 파라미터 부족 → 템플릿 생략됨: {rel_template}")
                        continue

                    from_key, to_key = required_keys[:2]  # 전제: 첫 2개가 관계 양 끝 노드 PK
                    if from_key not in rel_bindings or to_key not in rel_bindings:
                        logger.warning(f"[관계 MERGE] 누락된 바인딩 키: {[from_key, to_key]} → 건너뜀")
                        continue

                    try:
                        session.run(rel_template, rel_bindings)
                        logger.info(f"[관계 저장 성공] {rel_template}")
                    except Exception as e:
                        logger.warning(f"[관계 저장 실패] {rel_template}: {e}")

    return "GraphDB에 데이터 저장 완료"



def tranform_data_to_graphdb(schema):
    filtered_df = extract_data_according_to_schema(schema)
    cypher_template = run_cypher_template_generation_llm(schema)
    field_to_graph_map = schema["FieldToGraphMapping"]
    node_templates = cypher_template.get("node_merge_templates", {})
    relationship_templates = cypher_template.get("relationship_merge_templates", {})
    primary_keys = schema.get("primary_key", {})
    result = store_data_in_graphdb(filtered_df, node_templates, relationship_templates, field_to_graph_map, primary_keys)
    return result


if __name__ == "__main__":
    # 예시 스키마
    schema =  {'table_name': 'KNA1', 'nodes': ['Customer', 'Address', 'Region', 'Country', 'TaxInfo', 'Lifecycle', 'Extra'], 'relationships': [{'from': 'Customer', 'to': 'Address', 'type': 'HAS_ADDRESS', 'properties': ['STRAS', 'PSTLZ', 'ORT01', 'ORT02']}, {'from': 'Address', 'to': 'Region', 'type': 'IN_REGION', 'properties': ['REGIO']}, {'from': 'Region', 'to': 'Country', 'type': 'PART_OF', 'properties': ['LAND1']}, {'from': 'Customer', 'to': 'TaxInfo', 'type': 'HAS_TAX_INFO', 'properties': ['STCD1', 'STCD2', 'STCD3', 'STCD5', 'STCEG']}, {'from': 'Customer', 'to': 'Lifecycle', 'type': 'HAS_LIFECYCLE', 'properties': ['ERDAT', 'ERNAM', 'LOEVM', 'AEDAT', 'USNAM']}], 'primary_key': {'Customer': 'KUNNR', 'Address': 'ADRNR', 'Region': 'REGIO', 'Country': 'LAND1', 'TaxInfo': 'KUNNR', 'Lifecycle': 'KUNNR'}, 'node_properties': {'Customer': ['KUNNR', 'NAME1', 'NAME2', 'SORTL', 'ANRED', 'KTOKD', 'LIFNR', 'SPRAS', 'DUEFL'], 'Address': ['ADRNR', 'STRAS', 'PSTLZ', 'ORT01', 'ORT02'], 'Region': ['REGIO'], 'Country': ['LAND1'], 'TaxInfo': ['STCD1', 'STCD2', 'STCD3', 'STCD5', 'STCEG'], 'Lifecycle': ['ERDAT', 'ERNAM', 'LOEVM', 'AEDAT', 'USNAM'], 'Extra': ['MANDT', 'MCOD1', 'MCOD2', 'MCOD3', 'BBBNR', 'BBSNR', 'BUBKZ', 'TELF1', 'TELFX', 'TELF2', 'LZONE', 'VBUND', 'DEAR2', 'UMSAT', 'UMJAH', 'JMZAH', 'JMJAH', 'STKZA', 'STKZN', 'UMSA1', 'WERKS', 'HZUOR', 'FITYP', 'CASSD', 'J_1KFREPRE', 'J_1KFTBUS', 'J_1KFTIND', 'UPTIM', 'DEAR6', 'RIC', 'LEGALNAT', '/VSO/R_PALHGT', '/VSO/R_I_NO_LYR', '/VSO/R_ULD_SIDE', '/VSO/R_LOAD_PREF', 'DUNS', 'J_1IEXRN', 'J_1IPANNO', 'PSPNR', 'J_3GSTDMON', 'J_3GSTDTAG', 'J_3GTAGMON', 'J_3GVMONAT', 'J_3GEMINBE', 'J_3GFMGUE', 'J_3GZUSCHUE']}}
    filtered_df = extract_data_according_to_schema(schema)
    #cypher_template = run_cypher_template_generation_llm(schema)
    #print("Cypher Template:", json.dumps(cypher_template, indent=2, ensure_ascii=False))
    #result = tranform_data_to_graphdb(schema)
    #print(result)

    #cypher template 예시
    cypher_template = {
        "node_merge_templates": {
            "Customer": "MERGE (n:Customer {CustomerID: $CustomerID}) SET n += {CustomerID: $CustomerID, Name: $Name, ShortName: $ShortName, CustomerType: $CustomerType, DeletionFlag: $DeletionFlag}",
            "Address": "MERGE (n:Address {AddressID: $AddressID}) SET n += {AddressID: $AddressID, Street: $Street, PostalCode: $PostalCode}",
            "City": "MERGE (n:City {CityName: $CityName}) SET n += {CityName: $CityName}",
            "Region": "MERGE (n:Region {RegionCode: $RegionCode}) SET n += {RegionCode: $RegionCode}",
            "Country": "MERGE (n:Country {CountryCode: $CountryCode}) SET n += {CountryCode: $CountryCode}",
            "Contact": "MERGE (n:Contact {ContactID: $ContactID}) SET n += {ContactID: $ContactID, Phone1: $Phone1, Phone2: $Phone2, Fax: $Fax}",
            "TaxInfo": "MERGE (n:TaxInfo {TaxID: $TaxID}) SET n += {TaxID: $TaxID, TaxNumber1: $TaxNumber1, TaxNumber2: $TaxNumber2, TaxNumber3: $TaxNumber3, TaxNumber5: $TaxNumber5}",
            "LifecycleEvent": "MERGE (n:LifecycleEvent {EventID: $EventID}) SET n += {EventID: $EventID, CreatedDate: $CreatedDate, CreatedBy: $CreatedBy, UpdatedTime: $UpdatedTime}"
        },
        "relationship_merge_templates": [
            "MATCH (a:Customer {CustomerID: $CustomerID}), (b:Address {AddressID: $AddressID}) MERGE (a)-[:HAS_ADDRESS]->(b)",
            "MATCH (a:Address {AddressID: $AddressID}), (b:City {CityName: $CityName}) MERGE (a)-[:IN_CITY]->(b)",
            "MATCH (a:City {CityName: $CityName}), (b:Region {RegionCode: $RegionCode}) MERGE (a)-[:PART_OF]->(b)",
            "MATCH (a:Region {RegionCode: $RegionCode}), (b:Country {CountryCode: $CountryCode}) MERGE (a)-[:PART_OF]->(b)",
            "MATCH (a:Customer {CustomerID: $CustomerID}), (b:Contact {ContactID: $ContactID}) MERGE (a)-[:HAS_CONTACT]->(b)",
            "MATCH (a:Customer {CustomerID: $CustomerID}), (b:TaxInfo {TaxID: $TaxID}) MERGE (a)-[:HAS_TAX_INFO]->(b)",
            "MATCH (a:Customer {CustomerID: $CustomerID}), (b:LifecycleEvent {EventID: $EventID}) MERGE (a)-[:HAS_LIFECYCLE_EVENT]->(b)"   
        ]
    }

    field_to_graph_map = schema["FieldToGraphMapping"]
    node_templates = cypher_template.get("node_merge_templates", {})
    relationship_templates = cypher_template.get("relationship_merge_templates", {})
    primary_keys = schema.get("primary_key", {})
    result = store_data_in_graphdb(filtered_df, node_templates, relationship_templates, field_to_graph_map, primary_keys)