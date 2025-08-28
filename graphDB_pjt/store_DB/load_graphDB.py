from neo4j import GraphDatabase
import pandas as pd

uri = "neo4j+s://4a1461b3.databases.neo4j.io"
username = "neo4j"
password = "4a3eaaII9Gde1VIFSvECkPtER_Jzsks0NOtK6cBs0Sw"
driver = GraphDatabase.driver(uri, auth=(username, password))


def create_material(tx, matnr=None, idnrk=None, stlnr=None):
    material_code = matnr or idnrk
    if not material_code:
        return

    props = {}
    if matnr: props["MATNR"] = matnr
    if idnrk: props["IDNRK"] = idnrk
    if stlnr: props["STLNR"] = stlnr

    tx.run("""
        MERGE (m:Material {material_code: $material_code})
        SET m += $props
    """, material_code=material_code, props=props)


def create_relationship(tx, parent_code, component_code, props):
    tx.run("""
        MERGE (p:Material {material_code: $parent})
        MERGE (c:Material {material_code: $component})
        MERGE (p)-[r:HAS_COMPONENT]->(c)
        FOREACH (_ IN CASE WHEN size(keys($props)) > 0 THEN [1] ELSE [] END |
            SET r += $props
        )
    """, parent=parent_code, component=component_code, props=props)


def create_eina_graph(tx, idnrk, lifnr, ort01):
    material_code = idnrk
    if not material_code:
        return
    tx.run("""
        MERGE (m:Material {material_code: $material_code})
        SET m.IDNRK = $idnrk
    """, material_code=material_code, idnrk=idnrk)

    tx.run("MERGE (s:Supplier {LIFNR: $lifnr})", lifnr=lifnr)
    tx.run("MERGE (c:City {ORT01: $ort01})", ort01=ort01)

    tx.run("""
        MATCH (m:Material {material_code: $material_code})
        MATCH (s:Supplier {LIFNR: $lifnr})
        MERGE (m)-[:SUPPLIED_BY]->(s)
    """, material_code=material_code, lifnr=lifnr)

    tx.run("""
        MATCH (s:Supplier {LIFNR: $lifnr})
        MATCH (c:City {ORT01: $ort01})
        MERGE (s)-[:LOCATED_IN]->(c)
    """, lifnr=lifnr, ort01=ort01)


def load_mast_nodes(mast_path):
    df = pd.read_excel(mast_path, dtype=str)
    with driver.session() as session:
        for _, row in df.iterrows():
            matnr = str(row["MATNR"]).strip() if pd.notna(row["MATNR"]) else None
            stlnr = str(row["STLNR"]).strip() if pd.notna(row["STLNR"]) else None
            session.execute_write(create_material, matnr=matnr, stlnr=stlnr)


def load_stpo_relationships(stpo_path, mast_path):
    stpo_df = pd.read_excel(stpo_path, dtype=str)
    mast_df = pd.read_excel(mast_path, dtype=str)
    stlnr_to_matnr = dict(zip(mast_df["STLNR"], mast_df["MATNR"]))

    with driver.session() as session:
        for _, row in stpo_df.iterrows():
            stlnr = str(row["STLNR"]).strip() if pd.notna(row["STLNR"]) else None
            idnrk = str(row["IDNRK"]).strip() if pd.notna(row["IDNRK"]) else None
            if not stlnr or not idnrk:
                continue

            parent_code = stlnr_to_matnr.get(stlnr)
            component_code = idnrk

            # 노드 생성 (조건부 속성 저장)
            session.execute_write(create_material, matnr=parent_code)
            session.execute_write(create_material, idnrk=component_code)

            props = {}
            if pd.notna(row.get("MENGE")):
                props["MENGE"] = float(row["MENGE"])
            if pd.notna(row.get("MEINS")):
                props["MEINS"] = row["MEINS"].strip()
            if pd.notna(row.get("STPOZ")):
                props["STPOZ"] = int(row["STPOZ"])

            session.execute_write(create_relationship, parent_code, component_code, props)


def load_eina_data(eina_path):
    df = pd.read_csv(eina_path, dtype=str)
    with driver.session() as session:
        for _, row in df.iterrows():
            idnrk = str(row["IDNRK"]).strip() if pd.notna(row["IDNRK"]) else None
            lifnr = str(row["LIFNR"]).strip() if pd.notna(row["LIFNR"]) else None
            ort01 = str(row["ORT01"]).strip() if pd.notna(row["ORT01"]) else None
            if idnrk and lifnr and ort01:
                session.execute_write(create_eina_graph, idnrk, lifnr, ort01)


# 실행
stpo_path = "/home/pjtl2w01admin/csm/graphDB_pjt/data/test/stpo_test.xlsx"
mast_path = "/home/pjtl2w01admin/csm/graphDB_pjt/data/test/mast_test.xlsx"
eina_path = "/home/pjtl2w01admin/csm/graphDB_pjt/data/test/Mini_EINA_Table.csv"
load_stpo_relationships(stpo_path, mast_path)
load_mast_nodes(mast_path)
load_eina_data(eina_path)