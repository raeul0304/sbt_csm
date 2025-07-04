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

        pk_name_in_cypher = spec["pk"] #LLM이 제공하는 기본값


        records.append({
            "node": node,
            "merge_label": merge_label,
            "pk": spec["pk"],
            "set_fields": spec["set_fields"],
            "concat_fields": spec.get("concat_fields", []),
            "name_field": name_field
        })
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
    return pd.DataFrame(records)



# 2. 전처리 및 필드 보정
## concatenatedFields를 hash화해서 primary key 생성 -> 중복된 값이 있을 수 있기 때문
def generate_hashed_uuid(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(lambda x: hashlib.md5(x.encode()).hexdigest())

## concat_fields 조합 별로 join/hash 캐싱
def build_concat_cache(df: pd.DataFrame, node_df: pd.DataFrame) -> dict:
    cache = {}
    for _, row in node_df.iterrows():
        key = tuple(sorted(row["concat_fields"]))
        if key and key not in cache:
            cols = [f for f in key if f in df.columns]
            raw = df[cols].agg("_".join, axis=1)
            hashed = generate_hashed_uuid(raw)
            cache[key] = {"raw": raw, "hash": hashed}
    print(f"concat fields 조합별 cache 빌드 결과 : {cache}")
    return cache


## 각 노드의 name 필드 지정 로직 처리
def resolve_node_name(node: str, df: pd.DataFrame, name_field: str, set_fields: list, concat_fields: list, concat_cache: dict) -> pd.Series:
    if node.lower() == "extra":
        return pd.Series(["extra"] * len(df))
    elif node.lower() == "lifecycle" and "ERNAM" in df.columns:
        return df["ERNAM"]
    elif concat_fields:
        return concat_cache[tuple(sorted(concat_fields))]["raw"]
    elif name_field and name_field in df.columns:
        return df[name_field]
    else:
        name_fields_in_set = [f for f in set_fields if "NAME" in f.upper() and f in df.columns]
        if name_fields_in_set:
            return df[name_fields_in_set[0]]
        else:
            return pd.Series(["UNKNOWN"] * len(df)) #TODO: 여기로 빠지는게 있는지 확인하고, 있으면 상위 노드의 속성값으로 대체하도록 하기


## 중복으로 사용된 field 삭제하기
def remove_duplicate_fields(sub_df: pd.DataFrame, node: str, node_df: pd.DataFrame, set_fields: list) -> pd.DataFrame:
    for field in set_fields:
        used_by = node_df[node_df["set_fields"].apply(lambda s: field in s)]["node"].tolist()
        if len(used_by) > 1 and node != used_by[0] and field in sub_df.columns:
            sub_df.drop(columns=[field], inplace=True)
    return sub_df

## 노드별 df 만들기
def build_node_subdf(df: pd.DataFrame, node_row: pd.Series, concat_cache: dict) -> pd.DataFrame:
    node = node_row["node"]
    pk = node_row["pk"]
    set_fields = node_row["set_fields"]
    concat_fields = node_row["concat_fields"]

    fields_needed = list(set(set_fields + [pk]))
    valid_fields = [f for f in fields_needed if f in df.columns]
    sub_df = df[valid_fields].copy()

    if concat_fields:
        key = tuple(sorted(concat_fields))
        sub_df["concatenatedFields_raw"] = concat_cache[key]["raw"]
        sub_df["concatenatedFields_hash"] = concat_cache[key]["hash"]
        if pk == "concatenatedFields":
            sub_df[pk] = concat_cache[key]["hash"]
    
    if pk in sub_df.columns:
        is_missing = sub_df[pk].isnull() | (sub_df[pk] == "")
        if is_missing.any():
            fallback_base = sub_df[set_fields].astype(str).agg("_".join, axis=1)
            fallback_hash = generate_hashed_uuid(fallback_base)
            sub_df.loc[is_missing, pk] = fallback_hash[is_missing]
    
    print("sub_df >>>")
    print(sub_df.head())
    return sub_df


## 벡터화를 위한 전처리
def preprocess_for_vectorization(df: pd.DataFrame, node_df: pd.DataFrame) -> dict:
    processed_dfs = {}
    df = df.fillna("").copy()
    concat_cache = build_concat_cache(df, node_df)

    for _, row in node_df.iterrows():
        sub_df = build_node_subdf(df, row, concat_cache)
        sub_df["name"] = resolve_node_name(row["node"], df, row["name_field"], row["set_fields"], row["concat_fields"], concat_cache)
        sub_df = remove_duplicate_fields(sub_df, row["node"], node_df, row["set_fields"])
        sub_df = sub_df.fillna("")
        processed_dfs[row["node"]] = sub_df

    print(f"전처리 마친 dataframe >>> {processed_dfs}")
    return processed_dfs


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
    processed_dfs: dict # 전처리된 노드별 dataframe
):
    cypher_template = rel_row["cypher"]
    from_node_label = rel_row["from"]
    to_node_label = rel_row["to"]
    from_key_col = rel_row["from_key"] # from_key_col이 'concatenatedFields_hash'를 참조할 수 있음
    to_key_col = rel_row["to_key"]     # to_key_col이 'concatenatedFields_hash'를 참조할 수 있음
    rel_type = rel_row["rel_type"]

    print(f"  관계 저장 시도: {from_node_label} -> {to_node_label}")
    print(f"  from_key: {from_key_col}, to_key: {to_key_col}")
    print(f"  original cypher: {cypher_template}")

    rel_properties = []
    # 관계 속성 파싱 로직은 그대로 유지 (관계 자체의 속성)
    match = re.search(r'\[.+?:.+?\{(.+?)\}\]', cypher_template)
    if match:
        properties_str = match.group(1)
        rel_properties = [p.split(':')[0].strip() for p in properties_str.split(',')]
        rel_properties = [prop for prop in rel_properties if prop] # 빈 문자열 제거
    print(f"  파싱된 관계 속성: {rel_properties}")

    try:
        # Helper function to get data for a given key column
        # 이 함수는 source_df와 processed_dfs 모두에서 키 컬럼을 찾을 수 있도록 함
        # 찾으면 해당 Series를 반환하고, 아니면 None 반환
        def _get_key_series_from_dfs(key_col_name, node_label_for_processed_df, dfs_dict, source_dataframe):
            # 1. processed_dfs 내의 특정 노드 DataFrame에서 우선 찾음
            if node_label_for_processed_df in dfs_dict and \
               isinstance(dfs_dict[node_label_for_processed_df], pd.DataFrame) and \
               key_col_name in dfs_dict[node_label_for_processed_df].columns:
                print(f"  노드 키 데이터 '{key_col_name}' from processed_dfs: {node_label_for_processed_df}")
                return dfs_dict[node_label_for_processed_df][key_col_name]

            # 2. source_dataframe에서 찾음
            if key_col_name in source_dataframe.columns:
                print(f"  노드 키 데이터 '{key_col_name}' from source_df")
                return source_dataframe[key_col_name]

            # 3. 어디에서도 찾을 수 없는 경우
            print(f"*** 경고: 데이터프레임들에서 키 컬럼 '{key_col_name}'를 찾을 수 없습니다.")
            return None

        # 관계 데이터 생성을 위한 빈 DataFrame 초기화
        data_for_relations = pd.DataFrame()

        # 1. from_key_col 데이터 가져오기
        from_key_series = _get_key_series_from_dfs(from_key_col, from_node_label, processed_dfs, source_df)
        if from_key_series is None:
            print(f"*** {from_node_label} 노드의 키 데이터 ({from_key_col})를 찾을 수 없어 관계 생성을 건너뜁니다.")
            return
        # Series를 DataFrame에 추가할 때 인덱스를 리셋하여 병합 오류 방지
        data_for_relations[from_key_col] = from_key_series.reset_index(drop=True)

        # 2. to_key_col 데이터 가져오기
        to_key_series = _get_key_series_from_dfs(to_key_col, to_node_label, processed_dfs, source_df)
        if to_key_series is None:
            print(f"*** {to_node_label} 노드의 키 데이터 ({to_key_col})를 찾을 수 없어 관계 생성을 건너뜓니다.")
            return
        # Series를 DataFrame에 추가할 때 인덱스를 리셋하여 병합 오류 방지
        data_for_relations[to_key_col] = to_key_series.reset_index(drop=True)

        # 3. 관계 속성 데이터 (source_df에서 가져옴)
        for prop in rel_properties:
            if prop in source_df.columns:
                data_for_relations[prop] = source_df[prop].reset_index(drop=True)
            else:
                print(f"**** 경고: 관계 생성을 위한 필요한 컬럼 '{prop}'이(가) 원본 DataFrame에 없습니다. 해당 데이터는 null로 처리됩니다.")
                # 해당 컬럼이 없는 경우, NaN으로 채운 Series를 추가 (source_df의 길이만큼)
                data_for_relations[prop] = pd.Series([pd.NA] * len(source_df)).reset_index(drop=True)

        # 이제 data_for_relations이 필요한 모든 컬럼을 가진 DataFrame이 됨
        combined_df = data_for_relations

        # 4. 빈 문자열을 NaN으로 변환하여 dropna가 올바르게 작동하도록 함
        combined_df = combined_df.replace('', pd.NA)

        # 5. 노드 키 컬럼에 값이 없는 행 제거 (관계 생성이 불가능하므로)
        required_subset_for_dropna = [from_key_col, to_key_col]
        combined_df = combined_df.dropna(subset=required_subset_for_dropna, how='any')

        # 중복 관계 제거 (from_key, to_key, 관계 속성 기준으로)
        if combined_df.empty:
            print(f"⚠️  유효한 관계 데이터가 없음: {from_node_label} -> {to_node_label} (노드 키 데이터 누락 또는 전처리 후 모두 제거됨)")
            return
        combined_df = combined_df.drop_duplicates(subset=required_subset_for_dropna + rel_properties)

        if len(combined_df) == 0:
            print(f"⚠️  유효한 관계 데이터가 없음: {from_node_label} -> {to_node_label} (노드 키 데이터 누락 또는 전처리 후 모두 제거됨)")
            return

        # 6. 딕셔너리 리스트로 변환 (Neo4j UNWIND 쿼리용)
        rows = combined_df.to_dict(orient="records")

        # 7. UNWIND를 사용한 관계 생성 Cypher 쿼리 생성
        set_clause = ""
        if rel_properties:
            properties_set = []
            for prop in rel_properties:
                # `row.{prop}` 형식으로 Cypher 쿼리 문자열 생성
                properties_set.append(f"r.{prop}: row.{prop}")
            set_clause = f" ON CREATE SET {', '.join(properties_set)} ON MATCH SET {', '.join(properties_set)}"

        # from_key_col과 to_key_col은 동적으로 Cypher 쿼리에 포함됨
        unwind_cypher = f"""
        UNWIND $rows AS row
        MATCH (from:{from_node_label} {{{from_key_col}: row.{from_key_col}}})
        MATCH (to:{to_node_label} {{{to_key_col}: row.{to_key_col}}})
        MERGE (from)-[r:{rel_type}]->(to)
        {set_clause}
        """

        print(f"  실행할 Cypher: {unwind_cypher}")
        print(f"  관계 데이터 개수: {len(rows)}")
        print(f"  샘플 데이터 (첫 3개): {rows[:3] if len(rows) > 3 else rows}")

        # 8. Neo4j 실행
        result = session.run(unwind_cypher, {"rows": rows})

        if result is None:
            error_message = "session.run()이 None을 반환했습니다. 쿼리 실행 실패."
            print(f"**** 관계 저장 실패: {from_node_label} -> {to_node_label}, 에러: {error_message}")
            log_failure_to_csv(
                mode="relationship",
                entity_type=f"{from_node_label}→{to_node_label}",
                template=cypher_template.strip(),
                row={},
                error=error_message
            )
            return

        print(f">>>> 관계 저장 성공: {from_node_label} -> {to_node_label}, {len(rows)}개 관계 생성 시도")

        # 결과 요약 출력
        summary = result.consume()
        print(f"  생성된 관계: {summary.counters.relationships_created}")

    except Exception as e:
        print(f"**** 관계 저장 실패: {from_node_label} -> {to_node_label}, 에러: {str(e)}")
        log_failure_to_csv(
            mode="relationship",
            entity_type=f"{from_node_label}→{to_node_label}",
            template=cypher_template.strip(),
            row={}, # 전체 데이터 프레임 전달이 어려우므로 빈 딕셔너리 전달
            error=str(e)
        )


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
            print(f">>>> 저장되는 node: {node}, pk: {pk}")
            if row["node"] not in processed_dfs:
                continue
            rows = processed_dfs[row["node"]].to_dict(orient="records")
            print(f">>>> 저장되는 rows: {rows}")
            _save_node(session, node, pk, rows) # 직접 호출

        # 2. 관계 저장 (source_df에서 실제 관계 필터링)
        for _, rel_row in relationship_template_df.iterrows():
            print(f">>>> 관계 저장: {rel_row}")
            _save_relationship_vectorized(session, rel_row, source_df, processed_dfs) # 직접 호출

    driver.close()
    return "GraphDB 저장 완료 (노드 + 행기반 관계 + 벡터화 저장)"


def run_graphdb_pipeline(schema: dict) -> str:
    # 1. 원본 데이터 불러오기 (필요 필드만)
    df = extract_data_according_to_schema(schema)

    # 2. Cypher 템플릿 생성 (LLM 기반)
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


if __name__ == "__main__":
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
# 1. 원본 데이터 불러오기 (필요 필드만)
df = extract_data_according_to_schema(schema)
print(f"원본 데이터에서 필요 컬럼 가져오기 성공")

# 2. Cypher 템플릿 생성 (LLM 기반)
cypher_template = run_cypher_template_generation_llm(schema)
print("cypher template 생성하기 성공")

# 3. 노드/관계 템플릿 파싱
node_templates = cypher_template["node_merge_templates"]
relationship_templates = cypher_template["relationship_merge_templates"]
node_template_df = parse_node_templates_to_df(node_templates)
relationship_template_df = parse_relationship_templates_to_df(relationship_templates)
print("노드 및 관계 템플릿 파싱 성공")

# 4. 벡터화를 위한 전처리 (PK 보정, name 생성 등 포함)
processed_dfs = preprocess_for_vectorization(df, node_template_df)
print("벡터화를 위한 전처리 과정 성공")
    
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

print(f"결과>>>> {result}")