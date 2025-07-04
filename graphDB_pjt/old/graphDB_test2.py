import config
import logging
import json
import re
import os
import pandas as pd
import uuid
from neo4j import GraphDatabase
from graphDB_pjt.db_to_graphDB.graphdb_llm import run_cypher_template_generation_llm

# 로깅 설정
logging.basicConfig(level='INFO', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)

# Neo4j 설정
URI = config.NEO4J_URI
AUTH = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

# 실패 로그 경로
FAILED_LOG_PATH = os.path.join(os.getcwd(), "graphdb_failed_rows.csv")


def extract_data_according_to_schema(schema):
    used_columns = schema.get("used_fields", [])
    file_name = schema.get("table_name", "KNA1")
    xlsx_path = os.path.join(os.getcwd(), "tmp", f"{file_name}_cleaned.xlsx")

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"엑셀 파일이 존재하지 않습니다: {xlsx_path}")

    df = pd.read_excel(xlsx_path, dtype=str).fillna("").applymap(str.strip)
    df.columns = [col.strip().upper() for col in df.columns]
    return df[[col for col in used_columns if col in df.columns]].copy()


def build_safe_set_clause(template: str, bindings: dict):
    match = re.search(r"SET\s+n\s+\+=\s+{(.+?)}", template)
    if not match:
        return template, bindings

    field_pairs = [s.strip() for s in match.group(1).split(",")]
    filtered_pairs = []

    for pair in field_pairs:
        key, param = pair.split(":")
        param = param.strip().strip("$")
        if param in bindings:
            filtered_pairs.append(f"{key.strip()}: ${param}")

    new_set = ", ".join(filtered_pairs)
    safe_template = re.sub(r"SET\s+n\s+\+=\s+{.+?}", f"SET n += {{{new_set}}}", template)
    return safe_template, bindings


def log_failure_to_csv(entity_type: str, mode: str, template: str, bindings: dict, error: str):
    import csv
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


def store_data_in_graphdb(df, node_templates, relationship_templates, node_properties, primary_keys):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                bindings = {}

                # 바인딩 생성
                for entity, props in node_properties.items():
                    pk = primary_keys.get(entity)
                    for prop in props:
                        val = row_dict.get(prop.upper(), "").strip()
                        if prop == pk:
                            bindings[prop] = val if val else str(uuid.uuid4())
                        elif val:
                            bindings[prop] = val

                # 노드 저장
                for entity, template in node_templates.items():
                    pk = primary_keys.get(entity)
                    if not pk or pk not in bindings:
                        logger.warning(f"[{entity}] PK({pk}) 누락 → MERGE 스킵")
                        log_failure_to_csv(entity, "node", template, bindings, "Primary key missing")
                        continue

                    try:
                        safe_template, safe_bindings = build_safe_set_clause(template, bindings)
                        session.run(safe_template, safe_bindings)
                        logger.info(f"[노드 저장 성공] {entity}")
                    except Exception as e:
                        log_failure_to_csv(entity, "node", template, bindings, str(e))

                # 관계 저장
                for rel_template in relationship_templates:
                    keys_needed = re.findall(r"\$([a-zA-Z0-9_]+)", rel_template)
                    if all(k in bindings for k in keys_needed):
                        try:
                            session.run(rel_template, bindings)
                            logger.info(f"[관계 저장 성공]")
                        except Exception as e:
                            log_failure_to_csv("N/A", "relationship", rel_template, bindings, str(e))
                    else:
                        log_failure_to_csv("N/A", "relationship", rel_template, bindings, "Missing keys")


def transform_data_to_graphdb(schema):
    df = extract_data_according_to_schema(schema)
    cypher_template = run_cypher_template_generation_llm(schema)
    node_templates = cypher_template.get("node_merge_templates", {})
    relationship_templates = cypher_template.get("relationship_merge_templates", {})
    node_properties = schema.get("node_properties", {})
    primary_keys = schema.get("primary_key", {})
    return store_data_in_graphdb(df, node_templates, relationship_templates, node_properties, primary_keys)


if __name__ == "__main__":
    # 테스트용 예시 스키마 넣어도 됨
    pass
