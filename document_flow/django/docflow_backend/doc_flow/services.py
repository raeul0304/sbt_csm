import psycopg
from psycopg import sql
import config
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Mapping, Tuple
from django.db import connection
from .constants import GROUP_LABELS, X_BY_GROUP, Y_GAP, RULES, Group, DEFAULT_GROUP, DOC_TYPE_TO_GROUP


# ----Sales Order Number의 document flow 전체 경로 조회----
def get_document_flow_path(connection, schema, sales_order_number):
   
    if not connection:
        return []
        
    print(f"1단계: VBFA 테이블에서 Document Flow 경로를 조회 중입니다... (Sales Order: {sales_order_number})")
    
    query = sql.SQL( """
    SELECT
        VBELV AS pre_doc_no,
        VBELN AS post_doc_no,
        VBTYP_V AS pre_doc_type,
        VBTYP_N AS post_doc_type
    FROM
        {schema}.VBFA
    WHERE
        (VBELV = LPAD(%s, 10, '0') AND VBTYP_N = 'J')
        OR VBELV IN (
            SELECT VBELN
            FROM {schema}.VBFA
            WHERE VBELV = LPAD(%s, 10, '0') AND VBTYP_N = 'J'
        )
    ORDER BY
        ERDAT,
        ERZET;
    """).format(schema=sql.Identifier(schema))
    
    try:
        cursor = connection.cursor()
        # 쿼리를 실행하고 파라미터를 안전하게 전달
        cursor.execute(query, (sales_order_number, sales_order_number))  # ← 튜플로 한 번에
        columns = [c.name if hasattr(c, "name") else c[0] for c in cursor.description]
        path_results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return path_results
    except Exception as e:
        print(f"쿼리 실행 중 오류 발생: {e}")
        return []


# ----- document별 데이터 조회 -----
def execute_sql(conn, sql, params=None):
    if conn is None:
        raise ValueError("연결된 DB가 없습니다 (conn is None).")

    try:
        with conn.cursor() as cur:
            if params is not None:
                cur.execute(sql, params)
            else:
                cur.execute(sql)

            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        conn.rollback()
        print("SQL 실패:\n", sql)
        print("params:", params)
        raise e


def get_sales_order_details(conn, schema, sales_order_number):
    """Sales Order 상세 데이터를 조회"""
    #print(f" - Sales Order ({sales_order_number}) 조회 중...")
    query = sql.SQL("""
    SELECT
      'Sales Order' AS document_type,
      VBAK.VBELN AS doc_no,
      VBAP.POSNR AS item,
      VBAP.VGBEL AS preced_doc,
      VBAP.VGPOS AS orig_item,
      VBAP.KWMENG AS quantity,
      VBAP.VRKME AS unit,
      VBAP.NETWR AS ref_value,
      VBAK.WAERK AS curr,
      VBAK.ERDAT AS created_on,
      VBAP.MATNR AS material,
      VBAP.ARKTX AS description,
      CASE
        WHEN VBAK.GBSTK='C' THEN 'Completed'
        WHEN VBAK.GBSTK='B' THEN 'Partially Completed'
        WHEN VBAK.GBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'C' AS doc_code
    FROM {schema}.VBAK
    JOIN {schema}.VBAP ON VBAK.VBELN = VBAP.VBELN
    WHERE VBAK.VBELN = LPAD(%s, 10, '0')
    """).format(schema=sql.Identifier(schema))

    return execute_sql(conn, query, (sales_order_number,))


def get_outbound_delivery_details(conn, schema, doc_no, pre_doc_no=None):
    """Outbound Delivery (J) 상세 데이터를 조회"""
    #print(f" - Outbound Delivery ({doc_no}) 조회 중...")
    query = sql.SQL(""" 
    SELECT
      'Outbound Delivery' AS document_type,
      LIKP.VBELN AS doc_no,
      LIPS.POSNR AS item,
      LIPS.VGBEL AS preced_doc,
      LIPS.VGPOS AS orig_item,
      LIPS.LFIMG AS quantity,
      LIPS.VRKME AS unit,
      LIKP.ERDAT AS created_on,
      LIPS.MATNR AS material,
      LIPS.ARKTX AS description,
      CASE
        WHEN LIKP.GBSTK='C' THEN 'Completed'
        WHEN LIKP.GBSTK='B' THEN 'Partially Completed'
        WHEN LIKP.GBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'J' AS doc_code
    FROM {schema}.LIKP
    JOIN {schema}.LIPS ON LIKP.VBELN = LIPS.VBELN
    WHERE LIKP.VBELN = %s
    """).format(schema=sql.Identifier(schema))

    return execute_sql(conn, query, (doc_no,))


def get_picking_request_details(conn, schema, doc_no, pre_doc_no):
    """Picking Request (Q) 상세 데이터를 조회"""
    #print(f" - Picking Request ({doc_no}) 조회 중...")
    date_obj = datetime.strptime(doc_no, '%Y%m%d')
    formatted_doc_no = date_obj.strftime('%Y-%m-%d')
    #print(f"Formatted Picking Request No: {formatted_doc_no}")
    query = sql.SQL(""" 
    SELECT
      'Picking Request' AS document_type,
      LIKP.ERDAT AS doc_no,
      LIPS.POSNR AS item,
      LIKP.VBELN AS preced_doc,
      LIPS.VGPOS AS orig_item,
      LIPS.LFIMG AS quantity,
      LIPS.VRKME AS unit,
      LIKP.ERDAT AS created_on,
      LIPS.MATNR AS material,
      LIPS.ARKTX AS description,
      CASE
        WHEN LIKP.GBSTK='C' THEN 'Completed'
        WHEN LIKP.GBSTK='B' THEN 'Partially Completed'
        WHEN LIKP.GBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'Q' AS doc_code
    FROM {schema}.LIKP
    JOIN {schema}.LIPS ON LIKP.VBELN = LIPS.VBELN
    WHERE LIKP.ERDAT = %s AND LIKP.VBELN = %s
    """).format(schema=sql.Identifier(schema))

    result = execute_sql(conn, query, (formatted_doc_no, pre_doc_no,))
    result['doc_no'] = result['doc_no'].astype(str).str.replace('-', '', regex=False)
    return result


def get_goods_issue_details(conn, schema, doc_no, pre_doc_no=None):
    """GD Goods Issue (R) 상세 데이터를 조회"""
    #print(f" - GD Goods Issue ({doc_no}) 조회 중...")
    query = sql.SQL("""
    SELECT
      'GD Goods Issue' AS document_type,
      MSEG.MBLNR AS doc_no,
      MSEG.ZEILE AS item,
      MSEG.VBELN_IM AS preced_doc,
      MSEG.VBELP_IM AS orig_item,
      MSEG.MENGE AS quantity,
      MSEG.MEINS AS unit,
      MSEG.DMBTR AS ref_value,
      MSEG.WAERS AS curr,
      MSEG.CPUDT_MKPF AS created_on,
      MSEG.MATNR AS material,
      LIPS.ARKTX AS description,
      CASE
        WHEN LIKP.WBSTK='C' THEN 'Completed'
        WHEN LIKP.WBSTK='B' THEN 'Partially Completed'
        WHEN LIKP.WBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'R' AS doc_code
    FROM {schema}.MSEG
    LEFT JOIN {schema}.LIPS ON MSEG.VBELN_IM = LIPS.VBELN AND MSEG.VBELP_IM = LIPS.POSNR
    LEFT JOIN {schema}.LIKP ON LIPS.VBELN = LIKP.VBELN
    WHERE MSEG.MBLNR = %s
    """).format(schema=sql.Identifier(schema))
    return execute_sql(conn, query, (doc_no,))


def get_re_goods_delivery_details(conn, schema, doc_no, pre_doc_no=None):
    """RE goods delivery (h) 상세 데이터를 조회"""
    #print(f" - RE goods delivery ({doc_no}) 조회 중...")
    query = sql.SQL("""
    SELECT
      'RE Goods Delivery' AS document_type,
      MSEG.MBLNR AS doc_no,
      MSEG.ZEILE AS item,
      MSEG.VBELN_IM AS preced_doc,
      MSEG.VBELP_IM AS orig_item,
      MSEG.MENGE AS quantity,
      MSEG.MEINS AS unit,
      MSEG.DMBTR AS ref_value,
      MSEG.WAERS AS curr,
      MSEG.CPUDT_MKPF AS created_on,
      MSEG.MATNR AS material,
      LIPS.ARKTX AS description,
      CASE
        WHEN LIKP.GBSTK='C' THEN 'Completed'
        WHEN LIKP.GBSTK='B' THEN 'Partially Completed'
        WHEN LIKP.GBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'h' AS doc_code
    FROM {schema}.MSEG
    LEFT JOIN {schema}.LIPS ON MSEG.VBELN_IM = LIPS.VBELN AND MSEG.VBELP_IM = LIPS.POSNR
    LEFT JOIN {schema}.LIKP ON LIPS.VBELN = LIKP.VBELN
    WHERE MSEG.MBLNR = %s
    """).format(schema=sql.Identifier(schema))
    return execute_sql(conn, query, (doc_no,))


def get_invoice_details(conn, schema, doc_no, pre_doc_no=None):
    """Invoice (M) 상세 데이터를 조회"""
    #print(f" - Invoice ({doc_no}) 조회 중...")
    query = sql.SQL("""
    SELECT
        'Invoice' AS document_type,
        VBRK.VBELN AS doc_no,
        VBRP.POSNR AS item,
        VBRP.VGBEL AS preced_doc,
        VBRP.VGPOS AS orig_item,
        VBRP.FKIMG AS quantity,
        VBRP.VRKME AS unit,
        VBRP.NETWR AS ref_value,
        VBRK.WAERK AS curr,
        VBRK.ERDAT AS created_on,
        VBRP.MATNR AS material,
        VBRP.ARKTX AS description,
        CASE
            WHEN VBRK.GBSTK='C' THEN 'Completed'
            WHEN VBRK.GBSTK='B' THEN 'Partially Completed'
            WHEN VBRK.GBSTK='A' THEN 'Not Completed'
            ELSE 'Unknown'
        END AS status,
        'M' as doc_code
    FROM {schema}.VBRK
    JOIN {schema}.VBRP ON VBRK.VBELN = VBRP.VBELN
    WHERE VBRK.VBELN = %s
    """).format(schema=sql.Identifier(schema))

    return execute_sql(conn, query, (doc_no,))


def get_cancel_invoice_details(conn, schema, doc_no, pre_doc_no=None):
    """Cancel Invoice (N) 상세 데이터를 조회"""
    #print(f" - Cancel Invoice ({doc_no}) 조회 중...")
    query = sql.SQL("""
    SELECT
      'Cancel Invoice' AS document_type,
      VBRK.VBELN AS doc_no,         
      VBRP.POSNR AS item,          
      VBRK.STBLG AS preced_doc,    
      VBRP.VGPOS AS orig_item,    
      VBRP.FKIMG AS quantity,   
      VBRP.VRKME AS unit,     
      VBRP.NETWR AS ref_value,     
      VBRK.WAERK AS curr,         
      VBRK.ERDAT AS created_on,   
      VBRP.MATNR AS material,       
      VBRP.ARKTX AS description,  
      CASE
        WHEN VBRK.GBSTK='C' THEN 'Completed'
        WHEN VBRK.GBSTK='B' THEN 'Partially Completed'
        WHEN VBRK.GBSTK='A' THEN 'Not Completed'
        ELSE 'Unknown'
      END AS status,
      'N' AS doc_code          
    FROM {schema}.VBRK
    JOIN {schema}.VBRP ON VBRP.VBELN = VBRK.VBELN
    WHERE VBRK.VBELN = %s
    """).format(schema=sql.Identifier(schema))

    return execute_sql(conn, query, (doc_no,))
  

def get_journal_entry_details(conn, schema, doc_no):
    """Cancel Invoice - Journal Entry 상세 데이터를 조회"""
    #print(f" - Journal Entry ({doc_no}) 조회 중...")
    query = sql.SQL("""
    SELECT
      'Journal Entry' AS document_type,
      BSEG.BELNR AS doc_no,
      BSEG.VBELN as preced_doc,
      BSEG.WRBTR AS ref_value,
      BKPF.WAERS AS curr,
      BKPF.CPUDT AS created_on,
      CASE 
        WHEN BSEG.AUGBL IS NULL OR TRIM(BSEG.AUGBL) = '' 
        THEN 'Not Cleared' 
        ELSE 'Cleared' 
      END AS status,
      'E' AS doc_code
    FROM {schema}.BSEG
    JOIN {schema}.BKPF ON BKPF.BELNR = BSEG.BELNR
    WHERE BSEG.VBELN = %s AND BSEG.KOART = 'D'
    """).format(schema=sql.Identifier(schema))

    return execute_sql(conn, query, (doc_no,))




# ----- document별 데이터 조회 실행 -----
def get_document_details(conn, schema, sales_order_number, document_path):
    """
    Document Flow 경로 목록을 기반으로 각 문서의 상세 데이터를 조회하는 함수.
    """
    final_data = []
    delivery_doc_no = None

    # Sales Order 상세 데이터 조회 (경로와 관계없이 항상 조회)
    so_df = get_sales_order_details(conn, schema, sales_order_number)
    if not so_df.empty:
        #print(f"sales order 결과: {pd.DataFrame(so_df)}")
        final_data.append(so_df)

    print("2단계: 경로에 따라 후속 문서 상세 데이터 조회 중...")

    # 1) document_path가 DataFrame이면 row dict 리스트로 변환
    if isinstance(document_path, pd.DataFrame):
        rows = document_path.to_dict(orient='records')
    else:
        rows = document_path

    # 2) 각 row에서 post 문서만 추출하여 조회
    for row in rows:
        doc_no   = row.get('post_doc_no')
        doc_type = row.get('post_doc_type')
        result = pd.DataFrame() # 결과 초기화
        
        if doc_type == 'J':
            result = get_outbound_delivery_details(conn, schema, doc_no)
            final_data.append(result)  
            #print(f"outbound delivery 결과: {pd.DataFrame(result)}")
            delivery_doc_no = result['doc_no'].iloc[0]
            #print(f"Delivery Doc No: {delivery_doc_no}")
        
        elif doc_type == 'Q':
            result = get_picking_request_details(conn, schema, doc_no, pre_doc_no=delivery_doc_no)
            final_data.append(result)  
            #print(f"picking request 결과: {result}")
            
        
        elif doc_type == 'R':
            result = get_goods_issue_details(conn, schema, doc_no, pre_doc_no=delivery_doc_no)
            final_data.append(result)  
            #print(f"goods issue 결과: {result}")

        
        elif doc_type == 'h':
            result = get_re_goods_delivery_details(conn, schema, doc_no, pre_doc_no=delivery_doc_no)
            final_data.append(result)  
            #print(f"RE goods delivery 결과: {result}")


        elif doc_type == 'M':
            result = get_invoice_details(conn, schema, doc_no, pre_doc_no=delivery_doc_no)
            #print(f"invoice 결과: {result}")
            final_data.append(result)  
            invoice_doc_no = result['doc_no'].iloc[0]

            journal_result = get_journal_entry_details(conn, schema, invoice_doc_no)
            if journal_result is not None:
                #print(f"journal entry 결과: {journal_result}")
                final_data.append(journal_result)
        
        elif doc_type == 'N':
            result = get_cancel_invoice_details(conn, schema, doc_no, pre_doc_no=delivery_doc_no)
            #print(f"cancel invoice 결과: {result}")
            final_data.append(result)  
            cancel_invoice_doc_no = result['doc_no'].iloc[0]

            journal_result = get_journal_entry_details(conn, schema, cancel_invoice_doc_no)
            if journal_result is not None:
                #print(f"journal entry 결과: {journal_result}")
                final_data.append(journal_result)
     

    if final_data:
        return pd.concat(final_data, ignore_index=True)
    return pd.DataFrame()


# ----- document flow 최종 결과 json 변환 -----
# 그룹 구성
def get_group(
    doc_type: str,
    doc_type_to_group: Mapping[str, Group] = DOC_TYPE_TO_GROUP,
    default_group: Group = DEFAULT_GROUP,
) -> Group:
    key = doc_type.strip().lower()
    return doc_type_to_group.get(key, default_group)


# edge 구성
def build_flow_edges(df, rules: Mapping[str, Tuple[str, ...]] = RULES) -> List[Dict[str, Any]]:
    last_seen_idx : Dict[str, int] = {}
    edges: List[Dict[str, Any]] = []

    for i, row in df.reset_index(drop=True).iterrows():
        cur_code = str(row["doc_code"]).strip()

        if cur_code in rules:
            candidates = rules[cur_code]

            best_src = None
            best_idx = -1

            for cand_code in candidates:
                if cand_code in last_seen_idx and last_seen_idx[cand_code] < i:
                    if last_seen_idx[cand_code] > best_idx:
                        best_idx = last_seen_idx[cand_code]
                        best_src = cand_code
            
            if best_src:
                eid = f"{best_src}-{cur_code}-{i}"
                edges.append({
                    "id": eid,
                    "source": f"{best_src}-{best_idx}",
                    "target": f"{cur_code}-{i}",
                    "type": "lrstep",
                    "markerEnd": {"type": "ArrowClosed"}
                })
        
        last_seen_idx[cur_code] = i
    
    return edges


# position 계산
def compute_positions(
        df, edges,
        x_by_group: Mapping[Group, int] = X_BY_GROUP,
        y_gap: int = Y_GAP
    ) -> Dict[str, Dict[str, int]]:
    meta: Dict[str, Dict[str, str]] = {}
    
    for i, row in df.iterrows():
        doc_code = str(row["doc_code"]).strip()
        dtype = str(row["document_type"]).strip()
        nid = f"{doc_code}-{i}"
        group = get_group(dtype)
        meta[nid] = {"doc_type": dtype, "doc_code": doc_code, "group": group}
    
    
    latest_y = 0
    position: Dict[str, Dict[str, int]] = {}  #결과 포지션

    #sales order
    for i, row in enumerate(df.itertuples(index=False), start=0):
        if getattr(row, "document_type") in Group.SALES_ORDER:
            so_id = f"{str(getattr(row, 'doc_code')).strip()}-{i}"
            position[so_id] = {"x": x_by_group[Group.SALES_ORDER], "y": 0}
            break

    #엣지 순서에 따른 포지션 계산
    for e in edges:
        src, tgt = e["source"], e["target"]
        src_group, tgt_group = meta[src]["group"], meta[tgt]["group"]

        #source 배치 x시 처리
        if src not in position:
            position[src] = {"x": x_by_group[src_group], "y": latest_y}
        
        #target 배치
        tx = x_by_group[tgt_group]
        if src_group == tgt_group:
            # 다른 레인이면 최근 y + 150
            latest_y += y_gap
            ty = latest_y
        else:
            # 다른 레인이면 source와 같은 y
            ty = position[src]["y"]
        
        position[tgt] = {"x": tx, "y": ty}
    
    #최종 meta 구성
    for nid, p in position.items():
        meta[nid]["position"] = p
    
    return meta


#노드 구성
def build_nodes(positions):
    nodes = []
    for nid, info in positions.items():
        nodes.append({
            "id": nid,
            "type" : "doc_flow",
            "position": info["position"],
            "data": {"label": info["doc_type"]}
        })
    
    return nodes


def generate_final_json_docflow(df):
    edges = build_flow_edges(df)
    positions = compute_positions(df, edges)
    nodes = build_nodes(positions)
    return {"nodes": nodes, "edges": edges}


# ----- 전체 실행 -----
def run_documentflow_pipeline(schema, sales_no):
    conn = connection
    if conn:
        doc_path = get_document_flow_path(conn, schema, sales_no)
        
        if not doc_path:
            sales_doc = get_sales_order_details(conn, schema, sales_no)
            if sales_doc.empty:
                print(f"\nDocument Flow가 없습니다. (Sales Order: {sales_no})")
                return None
            print("\n최종결과:")
            print(pd.DataFrame(sales_doc))

        else:
            print("\n성공적으로 Document Flow 경로를 조회했습니다.")
            print(pd.DataFrame(doc_path))
            
            final_docflow = get_document_details(conn, schema, sales_no, doc_path)
            print("최종 doc flow: ")
            print(pd.DataFrame(final_docflow))

            result = generate_final_json_docflow(final_docflow)

            return result
                
        conn.close()
    else:
        print("프로그램을 종료합니다.")









