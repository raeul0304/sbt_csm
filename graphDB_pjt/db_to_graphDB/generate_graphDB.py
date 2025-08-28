import pandas as pd
import uuid
import os
import json
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from db_to_graphDB.graphdb_llm import run_cypher_template_generation_llm

FAILED_LOG_PATH = os.path.join(os.getcwd(), "vectorized_graphdb_failed_rows.csv")

#schema에 맞게 데이터 추출
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


#0. cypher 구문 파싱
def extract_merge_label_from_cypher(cypher: str, fallback:str) -> str:
    match = re.search(r"MERGE\s*\(\w+:(\w+)", cypher)
    return match.group(1) if match else fallback

def extract_node_labels_and_reltype_from_cypher(cypher: str):
    match_labels = re.findall(r"MATCH\s*\(\w+:(\w+)", cypher)
    rel_match = re.search(r'-\[:(\w+)(?:\s*\{[^}]*\})?\]->', cypher)
    from_label = match_labels[0] if len(match_labels) > 0 else "UNKNOWN_FROM"
    to_label = match_labels[1] if len(match_labels) > 1 else "UNKNOWN_TO"
    rel_type = rel_match.group(1) if rel_match else "UNKNOWN_REL"
    print(f"시작 노드 : {from_label}, 도착 노드: {to_label}, 관계명 : {rel_type}")
    return from_label, to_label, rel_type


# 1. cypher template 파싱
def parse_node_templates_to_df(node_templates: dict) -> pd.DataFrame:
    records = []
    for node, spec in node_templates.items():
        merge_label = extract_merge_label_from_cypher(spec.get("cypher", ""), node)
        name_field_match = re.search(r"name:\s*\$(\w+)", spec.get("cypher", ""))
        name_field = name_field_match.group(1) if name_field_match else None

        records.append({
            "node": node,
            "merge_label": merge_label,
            "pk": spec["pk"],
            "fields": spec["fields"],
            "name_field": name_field
        })
        #print(f">>>>node records: {records}")
    return pd.DataFrame(records)


def parse_relationship_templates_to_df(rel_templates: list) -> pd.DataFrame:
    records = []
    for rel in rel_templates:
        from_label, to_label, rel_type = extract_node_labels_and_reltype_from_cypher(rel.get("cypher", ""))
        records.append({
            "cypher": rel.get("cypher"),
            "from_key": rel.get("from_key"),
            "to_key": rel.get("to_key"),
            "from": from_label,
            "to": to_label,
            "rel_type": rel_type
        })
        #print(f">>>>relation records : {records}")
    return pd.DataFrame(records)



# 2. 전처리 및 필드 보정
## concatenatedFields를 hash화해서 primary key 생성 -> 중복된 값이 있을 수 있기 때문
def generate_hashed_uuid(series: pd.Series) -> pd.Series:
    temp_series = series.fillna("").astype(str)
    #print("DEBUG: temp_series head after fillna and astype:", temp_series.head().tolist())
    
    result = temp_series.apply(lambda x: hashlib.md5(x.encode()).hexdigest() if x.strip() else "EMPTY_HASH")
    print("DEBUG: result head after apply:", result.head().tolist())
    
    return result

## fields 조합 별로 join/hash 캐싱
def build_concat_cache(df: pd.DataFrame, node_df: pd.DataFrame) -> dict:
    cache = {}
    for _, row in node_df.iterrows():
        node = row["node"]
        key = tuple(sorted(row["fields"]))
        if not key:
            continue
        
        cols = [f for f in key if f in df.columns]
        if not cols:
            print(">>> WARNING : no valid concat fields found in df for key: {key}")
            continue
        
        raw = df[cols].astype(str).agg("_".join, axis=1)

        if "NAME1" in df.columns:
            customer_name = df["NAME1"]
            to_be_hashed = customer_name + f"_{node}" + raw
            hashed = generate_hashed_uuid(to_be_hashed)
        else:
            hashed = generate_hashed_uuid(raw)

        cache[key] = {"raw": raw, "hash": hashed}

    #print(f"concat fields 조합별 cache 빌드 결과 : {cache}")
    return cache


## 각 노드의 name 필드 지정 로직 처리
def resolve_node_name(node: str, sub_df: pd.DataFrame, name_field: str, raw_series, df) -> pd.Series:
    if node.lower() == "extra":
        return pd.Series(["extra"] * len(sub_df))
    elif node.lower() == "lifecycle" and "ERNAM" in sub_df.columns:
        return sub_df["ERNAM"]
    # name 처리
    if name_field == "concatenatedFields":
        return raw_series.astype(str) + f"_{node}"

    elif name_field and name_field in sub_df.columns:
        name_series = sub_df[name_field].fillna("").astype(str).str.strip()
        fallback = df["NAME1"].astype(str).str.strip() + f"_{node}" if "NAME1" in df.columns else f"{node}_UNKNOWN"
        return name_series.mask(name_series.eq(""), fallback)

    else:
        return (
            df["NAME1"].astype(str).str.strip() + f"_{node}"
            if "NAME1" in df.columns else f"{node}_UNKNOWN"
        )


## 중복으로 사용된 field 삭제하기
def remove_duplicate_fields(sub_df: pd.DataFrame, node: str, node_df: pd.DataFrame, fields: list) -> pd.DataFrame:
    for field in fields:
        used_by = node_df[node_df["fields"].apply(lambda s: field in s)]["node"].tolist()
        if len(used_by) > 1 and node != used_by[0] and field in sub_df.columns:
            sub_df.drop(columns=[field], inplace=True)
    return sub_df


## 노드별 df 만들기
def build_node_subdf(df: pd.DataFrame, node_df_row: pd.Series, concat_cache: dict) -> pd.DataFrame:
    node = node_df_row["node"]
    pk = node_df_row["pk"]
    fields = node_df_row["fields"]
    name_field = node_df_row["name_field"]
    
    # fields 기준으로 먼저 필요 컬럼만 추출
    fields_needed = list(set(fields))
    valid_fields = [f for f in fields_needed if f in df.columns]
    sub_df = df[valid_fields].copy()
    sub_df.index = df.index

    key = tuple(sorted(fields))
    raw_series = concat_cache[key]["raw"]

    # pk처리
    if not pk or pk == "concatenatedFields":
        sub_df[pk] = raw_series + "_" + df["KUNNR"] #값이 비어있을 경우 pk를 fields join + kunnr
    else:
        sub_df[pk] = df[pk].copy()    
    
    sub_df["name"] = resolve_node_name(node, sub_df, name_field, raw_series, df)


    if node not in ["City", "Address"]:
        is_all_empty = sub_df[fields].fillna("").astype(str).apply(lambda row: all(val.strip() == "" for val in row), axis=1)
        sub_df = sub_df[~is_all_empty]

    return sub_df


## 벡터화를 위한 전처리
def preprocess_for_vectorization(df: pd.DataFrame, node_df: pd.DataFrame) -> dict:
    processed_dfs = {}
    df = df.fillna("").copy()
    concat_cache = build_concat_cache(df, node_df)

    for _, row in node_df.iterrows():
        sub_df = build_node_subdf(df, row, concat_cache)
        sub_df = remove_duplicate_fields(sub_df, row["node"], node_df, row["fields"])
        sub_df = sub_df.fillna("")
        processed_dfs[row["node"]] = sub_df

    return processed_dfs

################################################################################################################
# 3. 실패 로깅
def log_failure_to_csv(mode: str, entity_type: str, template: str, row: dict, error: str):
    headers = ["mode", "entity_type", "template", "bindings", "error"]
    log_row = {
        "mode": mode,
        "entity_type": entity_type,
        "template": template, 
        "bindings": json.dumps(row, ensure_ascii=False),
        "error": error
    }
    log_df = pd.DataFrame([log_row])
    if not os.path.exists(FAILED_LOG_PATH):
        log_df.to_csv(FAILED_LOG_PATH, index=False, encoding='utf-8-sig')
    else:
        log_df.to_csv(FAILED_LOG_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
    print(f"Failed {mode} for {entity_type}. Error: {error}. Data: {row}")


# 4. UNWIND 멀티스레딩 저장
def _save_node(session, node, pk, rows):
    cypher = f"""
    UNWIND $rows AS row
    MERGE (n:{node} {{{pk}: row.{pk}}})
    SET n += row
    """
    try:
        session.run(cypher, {"rows": rows})
    except Exception as e:
        for r in rows:
            log_failure_to_csv("node", node, cypher.strip(), r, str(e))


def _save_relationship_vectorized(
    session,
    rel_row: pd.Series,
    source_df: pd.DataFrame,
    processed_dfs: dict
):
    from_key_col = rel_row["from_key"]
    to_key_col = rel_row["to_key"]
    from_node_label = rel_row["from"]
    to_node_label = rel_row["to"]
    rel_type = rel_row["rel_type"]
    cypher_template = rel_row["cypher"]

    # 관계 속성 파싱
    rel_properties = []
    match = re.search(r"\[.+?:.+?\{(.+?)\}\]", cypher_template)
    if match:
        properties_str = match.group(1)
        rel_properties = [p.split(":")[0].strip() for p in properties_str.split(",") if ":" in p]

    try:
        if from_node_label not in processed_dfs or from_key_col not in processed_dfs[from_node_label]:
            log_failure_to_csv("relationship", f"{from_node_label}->{to_node_label}", cypher_template.strip(), {}, f"{from_key_col} not found")
            return
        if to_node_label not in processed_dfs or to_key_col not in processed_dfs[to_node_label]:
            log_failure_to_csv("relationship", f"{from_node_label}->{to_node_label}", cypher_template.strip(), {}, f"{to_key_col} not found")
            return

        from_df = processed_dfs[from_node_label]
        to_df = processed_dfs[to_node_label]

        # 관계 키 조합
        df_rel = pd.DataFrame({
            from_key_col: from_df[from_key_col],
            to_key_col: to_df[to_key_col]
        })

        # 관계 속성 채우기 (예: STRAS, PSTLZ 등)
        for prop in rel_properties:
            if prop in source_df.columns:
                df_rel[prop] = source_df[prop]
            elif prop in from_df.columns:
                df_rel[prop] = from_df[prop]
            elif prop in to_df.columns:
                df_rel[prop] = to_df[prop]
            else:
                df_rel[prop] = pd.NA  # 없는 경우는 빈 값

        # 결측치 제거 및 중복 제거
        df_rel.replace('', pd.NA, inplace=True)
        df_rel.dropna(subset=[from_key_col, to_key_col], how='any', inplace=True)
        df_rel.drop_duplicates(subset=[from_key_col, to_key_col] + rel_properties, inplace=True)

        if df_rel.empty:
            print(f"⚠️ 유효한 관계 데이터 없음: {from_node_label} → {to_node_label}")
            return

        rows = df_rel.to_dict(orient="records")

        # 관계 Cypher 구성
        set_clause = ""
        if rel_properties:
            set_exprs = [f"r.{p} = row.{p}" for p in rel_properties]
            set_clause = f"ON CREATE SET {', '.join(set_exprs)} ON MATCH SET {', '.join(set_exprs)}"

        final_cypher = f"""
        UNWIND $rows AS row
        MATCH (from:{from_node_label} {{{from_key_col}: row.{from_key_col}}})
        MATCH (to:{to_node_label} {{{to_key_col}: row.{to_key_col}}})
        MERGE (from)-[r:{rel_type}]->(to)
        {set_clause}
        """

        session.run(final_cypher, {"rows": rows})
        print(f">>>> 관계 저장 완료: {from_node_label} → {to_node_label} ({len(rows)}개)")

    except Exception as e:
        log_failure_to_csv("relationship", f"{from_node_label}→{to_node_label}", cypher_template.strip(), {}, str(e))
        print(f"!!! 관계 저장 실패: {from_node_label} → {to_node_label}, 에러: {str(e)}")



def store_processed_dfs_to_neo4j(
    processed_dfs: dict,
    node_template_df: pd.DataFrame,
    relationship_template_df: pd.DataFrame,
    source_df: pd.DataFrame, #관계 필터링용 원본 dataframe
    neo4j_uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> str:
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    with driver.session(database=database) as session:
        # 1. 노드 저장 (벡터화된 DataFrame 기반)
        for _, row in node_template_df.iterrows():
            node = row["merge_label"]
            pk = row["pk"]
            #print(f">>>> 저장되는 node: {node}, pk: {pk}")
            if row["node"] not in processed_dfs:
                continue
            rows = processed_dfs[row["node"]].to_dict(orient="records")
            #print(f">>>> 저장되는 rows: {rows}")
            _save_node(session, node, pk, rows) # 직접 호출

        # 2. 관계 저장 (source_df에서 실제 관계 필터링)
        for _, rel_row in relationship_template_df.iterrows():
            #print(f">>>> 관계 저장: {rel_row}")
            _save_relationship_vectorized(session, rel_row, source_df, processed_dfs) # 직접 호출

    driver.close()
    return "GraphDB 저장 완료 (노드 + 행기반 관계 + 벡터화 저장)"


def _ensure_dict(obj):
    """str 이면 json.loads, dict 이면 그대로 돌려준다."""
    return json.loads(obj) if isinstance(obj, str) else obj


def run_graphdb_pipeline(schema: dict) -> str:
    schema = _ensure_dict(schema)
    # 1. 원본 데이터 불러오기 (필요 필드만)
    df = extract_data_according_to_schema(schema)

    # 2. Cypher 템플릿 생성
    cypher_template = run_cypher_template_generation_llm(schema)

    # 3. 노드/관계 템플릿 파싱
    node_templates = cypher_template["node_merge_templates"]
    relationship_templates = cypher_template["relationship_merge_templates"]
    node_template_df = parse_node_templates_to_df(node_templates)
    relationship_template_df = parse_relationship_templates_to_df(relationship_templates)

    # 4. 벡터화를 위한 전처리 (PK 보정, name 생성 등 포함)
    processed_dfs = preprocess_for_vectorization(df, node_template_df)

    # 5. Neo4j 저장 (노드 + 관계 멀티스레딩 저장)
    result = store_processed_dfs_to_neo4j(
        processed_dfs=processed_dfs,
        node_template_df=node_template_df,
        relationship_template_df=relationship_template_df,
        source_df=df,  # 관계 필터링용 원본 DataFrame
        neo4j_uri=config.NEO4J_URI,
        user=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    print(f"결과>> {result}")

    return result



if __name__ == "__main__" :
    schema = {
  "table_name": "KNA1",
  "nodes": [
    "Customer",
    "Address",
    "City",
    "Country",
    "Contact",
    "TaxInfo",
    "Lifecycle",
    "Extra"
  ],
  "relationships": [
    {
      "from": "Customer",
      "to": "Address",
      "type": "HAS_ADDRESS",
      "properties": ["ADRNR"]
    },
    {
      "from": "Address",
      "to": "City",
      "type": "IN_CITY",
      "properties": ["ORT01"]
    },
    {
      "from": "City",
      "to": "Country",
      "type": "PART_OF",
      "properties": ["LAND1"]
    },
    {
      "from": "Customer",
      "to": "Contact",
      "type": "HAS_CONTACT",
      "properties": ["TELF1", "TELFX", "TELF2"]
    },
    {
      "from": "Customer",
      "to": "TaxInfo",
      "type": "HAS_TAX_INFO",
      "properties": ["STCD1", "STCD2", "STCD3", "STCD5", "STCEG"]
    },
    {
      "from": "Customer",
      "to": "Lifecycle",
      "type": "HAS_LIFECYCLE",
      "properties": ["ERDAT", "ERNAM", "LOEVM", "AEDAT", "USNAM"]
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
    "City": "concatenatedFields",
    "Country": "LAND1",
    "Contact": "concatenatedFields",
    "TaxInfo": "concatenatedFields",
    "Lifecycle": "concatenatedFields",
    "Extra": "KUNNR"
  },
  "node_properties": {
    "Customer": ["KUNNR", "NAME1", "NAME2", "SORTL", "KTOKD", "LIFNR"],
    "Address": ["ADRNR", "STRAS", "PSTLZ", "LZONE"],
    "City": ["ORT01", "REGIO"],
    "Country": ["LAND1"],
    "Contact": ["TELF1", "TELFX", "TELF2"],
    "TaxInfo": ["STCD1", "STCD2", "STCD3", "STCD5", "STCEG"],
    "Lifecycle": ["ERDAT", "ERNAM", "LOEVM", "AEDAT", "USNAM"],
    "Extra": [
      "MANDT", "MCOD1", "MCOD2", "MCOD3", "ANRED", "BBBNR", "BBSNR", "BUBKZ", "ORT02",
      "SPRAS", "STKZA", "VBUND", "DEAR2", "UMSAT", "UMJAH", "JMZAH", "JMJAH", "STKZN",
      "UMSA1", "WERKS", "DUEFL", "HZUOR", "FITYP", "CASSD", "J_1KFREPRE", "J_1KFTBUS",
      "J_1KFTIND", "UPTIM", "DEAR6", "RIC", "LEGALNAT", "_VSO_R_PALHGT", "_VSO_R_I_NO_LYR",
      "_VSO_R_ULD_SIDE", "_VSO_R_LOAD_PREF", "DUNS", "J_1IEXRN", "J_1IPANNO", "PSPNR",
      "J_3GSTDMON", "J_3GSTDTAG", "J_3GTAGMON", "J_3GVMONAT", "J_3GEMINBE", "J_3GFMGUE",
      "J_3GZUSCHUE"
    ]
  },
  "used_fields": [
    "KUNNR", "NAME1", "NAME2", "SORTL", "KTOKD", "LIFNR", "ADRNR", "STRAS", "PSTLZ", "LZONE",
    "ORT01", "REGIO", "LAND1", "TELF1", "TELFX", "TELF2", "STCD1", "STCD2", "STCD3", "STCD5",
    "STCEG", "ERDAT", "ERNAM", "LOEVM", "AEDAT", "USNAM"
  ]
}

    df = extract_data_according_to_schema(schema)
    cypher_template = {'node_merge_templates': {'Customer': {'cypher': 'MERGE (n:Customer {KUNNR: $KUNNR}) SET n += {name: $NAME1, KUNNR: $KUNNR, NAME1: $NAME1, NAME2: $NAME2, SORTL: $SORTL, KTOKD: $KTOKD, LIFNR: $LIFNR}', 'pk': 'KUNNR', 'fields': ['KUNNR', 'NAME1', 'NAME2', 'SORTL', 'KTOKD', 'LIFNR']}, 'Address': {'cypher': 'MERGE (n:Address {ADRNR: $ADRNR}) SET n += {name: $STRAS, ADRNR: $ADRNR, STRAS: $STRAS, PSTLZ: $PSTLZ, LZONE: $LZONE}', 'pk': 'ADRNR', 'fields': ['ADRNR', 'STRAS', 'PSTLZ', 'LZONE']}, 'City': {'cypher': 'MERGE (n:City {ID: $ORT01}) SET n += {name: $ORT01, ORT01: $ORT01, REGIO: $REGIO}', 'pk': 'ORT01', 'fields': ['ORT01', 'REGIO']}, 'Country': {'cypher': 'MERGE (n:Country {LAND1: $LAND1}) SET n += {name: $LAND1, LAND1: $LAND1}', 'pk': 'LAND1', 'fields': ['LAND1']}, 'Contact': {'cypher': 'MERGE (n:Contact {ID: $concatenatedFields}) SET n += {name: $TELF1, TELF1: $TELF1, TELFX: $TELFX, TELF2: $TELF2}', 'pk': 'concatenatedFields', 'fields': ['TELF1', 'TELFX', 'TELF2']}, 'TaxInfo': {'cypher': 'MERGE (n:TaxInfo {ID: $concatenatedFields}) SET n += {name: $STCD1, STCD1: $STCD1, STCD2: $STCD2, STCD3: $STCD3, STCD5: $STCD5, STCEG: $STCEG}', 'pk': 'concatenatedFields', 'fields': ['STCD1', 'STCD2', 'STCD3', 'STCD5', 'STCEG']}, 'Lifecycle': {'cypher': 'MERGE (n:Lifecycle {ID: $concatenatedFields}) SET n += {name: $ERNAM, ERDAT: $ERDAT, ERNAM: $ERNAM, LOEVM: $LOEVM, AEDAT: $AEDAT, USNAM: $USNAM}', 'pk': 'concatenatedFields', 'fields': ['ERDAT', 'ERNAM', 'LOEVM', 'AEDAT', 'USNAM']}, 'Extra': {'cypher': 'MERGE (n:Extra {KUNNR: $KUNNR}) SET n += {name: "extra", MANDT: $MANDT, MCOD1: $MCOD1, MCOD2: $MCOD2, MCOD3: $MCOD3, ANRED: $ANRED, BBBNR: $BBBNR, BBSNR: $BBSNR, BUBKZ: $BUBKZ, ORT02: $ORT02, SPRAS: $SPRAS, STKZA: $STKZA, VBUND: $VBUND, DEAR2: $DEAR2, UMSAT: $UMSAT, UMJAH: $UMJAH, JMZAH: $JMZAH, JMJAH: $JMJAH, STKZN: $STKZN, UMSA1: $UMSA1, WERKS: $WERKS, DUEFL: $DUEFL, HZUOR: $HZUOR, FITYP: $FITYP, CASSD: $CASSD, J_1KFREPRE: $J_1KFREPRE, J_1KFTBUS: $J_1KFTBUS, J_1KFTIND: $J_1KFTIND, UPTIM: $UPTIM, DEAR6: $DEAR6, RIC: $RIC, LEGALNAT: $LEGALNAT, _VSO_R_PALHGT: $_VSO_R_PALHGT, _VSO_R_I_NO_LYR: $_VSO_R_I_NO_LYR, _VSO_R_ULD_SIDE: $_VSO_R_ULD_SIDE, _VSO_R_LOAD_PREF: $_VSO_R_LOAD_PREF, DUNS: $DUNS, J_1IEXRN: $J_1IEXRN, J_1IPANNO: $J_1IPANNO, PSPNR: $PSPNR, J_3GSTDMON: $J_3GSTDMON, J_3GSTDTAG: $J_3GSTDTAG, J_3GTAGMON: $J_3GTAGMON, J_3GVMONAT: $J_3GVMONAT, J_3GEMINBE: $J_3GEMINBE, J_3GFMGUE: $J_3GFMGUE, J_3GZUSCHUE: $J_3GZUSCHUE}', 'pk': 'KUNNR', 'fields': ['MANDT', 'MCOD1', 'MCOD2', 'MCOD3', 'ANRED', 'BBBNR', 'BBSNR', 'BUBKZ', 'ORT02', 'SPRAS', 'STKZA', 'VBUND', 'DEAR2', 'UMSAT', 'UMJAH', 'JMZAH', 'JMJAH', 'STKZN', 'UMSA1', 'WERKS', 'DUEFL', 'HZUOR', 'FITYP', 'CASSD', 'J_1KFREPRE', 'J_1KFTBUS', 'J_1KFTIND', 'UPTIM', 'DEAR6', 'RIC', 'LEGALNAT', '_VSO_R_PALHGT', '_VSO_R_I_NO_LYR', '_VSO_R_ULD_SIDE', '_VSO_R_LOAD_PREF', 'DUNS', 'J_1IEXRN', 'J_1IPANNO', 'PSPNR', 'J_3GSTDMON', 'J_3GSTDTAG', 'J_3GTAGMON', 'J_3GVMONAT', 'J_3GEMINBE', 'J_3GFMGUE', 'J_3GZUSCHUE']}}, 'relationship_merge_templates': [{'cypher': 'MATCH (a:Customer {KUNNR: $KUNNR}) MATCH (b:Address {ADRNR: $ADRNR}) MERGE (a)-[:HAS_ADDRESS {ADRNR: $ADRNR}]->(b)', 'from_key': 'KUNNR', 'to_key': 'ADRNR'}, {'cypher': 'MATCH (a:Address {ADRNR: $ADRNR}) MATCH (b:City {ID: $ORT01}) MERGE (a)-[:IN_CITY {ORT01: $ORT01}]->(b)', 'from_key': 'ADRNR', 'to_key': 'ORT01'}, {'cypher': 'MATCH (a:City {ID: $ORT01}) MATCH (b:Country {LAND1: $LAND1}) MERGE (a)-[:PART_OF {LAND1: $LAND1}]->(b)', 'from_key': 'ORT01', 'to_key': 'LAND1'}, {'cypher': 'MATCH (a:Customer {KUNNR: $KUNNR}) MATCH (b:Contact {ID: $concatenatedFields}) MERGE (a)-[:HAS_CONTACT {TELF1: $TELF1, TELFX: $TELFX, TELF2: $TELF2}]->(b)', 'from_key': 'KUNNR', 'to_key': 'concatenatedFields'}, {'cypher': 'MATCH (a:Customer {KUNNR: $KUNNR}) MATCH (b:TaxInfo {ID: $concatenatedFields}) MERGE (a)-[:HAS_TAX_INFO {STCD1: $STCD1, STCD2: $STCD2, STCD3: $STCD3, STCD5: $STCD5, STCEG: $STCEG}]->(b)', 'from_key': 'KUNNR', 'to_key': 'concatenatedFields'}, {'cypher': 'MATCH (a:Customer {KUNNR: $KUNNR}) MATCH (b:Lifecycle {ID: $concatenatedFields}) MERGE (a)-[:HAS_LIFECYCLE {ERDAT: $ERDAT, ERNAM: $ERNAM, LOEVM: $LOEVM, AEDAT: $AEDAT, USNAM: $USNAM}]->(b)', 'from_key': 'KUNNR', 'to_key': 'concatenatedFields'}, {'cypher': 'MATCH (a:Customer {KUNNR: $KUNNR}) MATCH (b:Extra {KUNNR: $KUNNR}) MERGE (a)-[:HAS_EXTRA]->(b)', 'from_key': 'KUNNR', 'to_key': 'KUNNR'}]}
    # 3. 노드/관계 템플릿 파싱
    node_templates = cypher_template["node_merge_templates"]
    relationship_templates = cypher_template["relationship_merge_templates"]
    node_template_df = parse_node_templates_to_df(node_templates)
    relationship_template_df = parse_relationship_templates_to_df(relationship_templates)

    # 4. 벡터화를 위한 전처리 (PK 보정, name 생성 등 포함)
    processed_dfs = preprocess_for_vectorization(df, node_template_df)

    # 5. Neo4j 저장 (노드 + 관계 멀티스레딩 저장)
    result = store_processed_dfs_to_neo4j(
        processed_dfs=processed_dfs,
        node_template_df=node_template_df,
        relationship_template_df=relationship_template_df,
        source_df=df,  # 관계 필터링용 원본 DataFrame
        neo4j_uri=config.NEO4J_URI,
        user=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )
    print(f"결과>> {result}")