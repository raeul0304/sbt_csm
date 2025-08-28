import pandas as pd
from datetime import time
import re

def read_excel_file(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

# 컬럼명 - 특수문자 제거
def make_field_cypher_safe(field_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", field_name.strip())

def apply_column_name_cleaning(df):
    original_columns = df.columns.tolist()
    cleaned_columns = [make_field_cypher_safe(col.upper()) for col in original_columns]

    column_changes = {
        orig: new for orig, new in zip(original_columns, cleaned_columns) if orig != new
    }

    df_renamed = df.copy()
    df_renamed.columns = cleaned_columns

    df_renamed_only = df_renamed[list(column_changes.values())].copy() if column_changes else pd.DataFrame()
    return df_renamed, df_renamed_only, column_changes



def detect_changes(df_before: pd.DataFrame, df_after: pd.DataFrame):
    common_cols = df_before.columns.intersection(df_after.columns)
    df_before_trimmed = df_before[common_cols]
    df_after_trimmed = df_after[common_cols]

    modified_mask = (df_before_trimmed != df_after_trimmed) & df_before_trimmed.notna()
    modified_rows = df_after.loc[modified_mask.any(axis=1)].copy()
    modified_cols = modified_mask.any(axis=0)
    modified_columns = modified_cols[modified_cols].index.tolist()
    modified_rows_with_columns = modified_rows[modified_columns]

    return modified_rows_with_columns



# 날짜/시간 컬럼 문자열로 변환
def date_time_data_to_string(df):
    df_before = df.copy()
    df_after = df.copy()

    for col in df_after.columns:
        if df_after[col].map(lambda x: isinstance(x, (pd.Timestamp, time))).any():
            df_after[col] = df_after[col].astype(str)

    modified_subset = detect_changes(df_before, df_after)
    return df_after, modified_subset


def remove_empty_columns(df):
    df_before = df.copy()
    df_after = df.copy()

    # 기호 처리
    missing_values = ["..", "N/A", "#N/A", "없음", "무", "NONE"]
    df_after.replace(missing_values, "", inplace=True)

    # 빈 컬럼 제거
    df_after = df_after.dropna(axis=1, how='all')
    df_after = df_after.loc[:, ~(df_after == "").all()]

    # 0으로만 채워진 수치형 컬럼 제거
    zero_columns = [
        col for col in df_after.columns
        if pd.api.types.is_numeric_dtype(df_after[col]) and (df_after[col] == 0).all()
    ]
    df_after = df_after.drop(columns=zero_columns)

    modified_subset = detect_changes(df_before, df_after)
    return df_after, modified_subset





###데이터 전처리 모음 (kna1 버전)
def preprocess_data(df):
    #cypher 구문 생성시 에러 방지용 전처리 - ". 이나 / 모두 _ 로 변경"
    original_columns = df.columns.tolist()
    cleaned_columns = [make_field_cypher_safe(col.upper()) for col in original_columns]
    df.columns = cleaned_columns
    
    # 모든 값이 NaN인 컬럼 제거
    df_cleaned = df.dropna(axis=1, how='all')
    df_cleaned = df_cleaned.loc[:, ~(df_cleaned == "").all()]

    # 날짜/시간 컬럼 문자열로 변환
    for col in df_cleaned.columns:
        if df_cleaned[col].map(lambda x: isinstance(x, (pd.Timestamp, time))).any():
            df_cleaned[col] = df_cleaned[col].astype(str)

    #모든 값이 0인 수치 컬럼 제거
    zero_columns = [
        col for col in df_cleaned.columns
        if pd.api.types.is_numeric_dtype(df_cleaned[col]) and (df_cleaned[col] == 0).all()
    ]
    df_cleaned = df_cleaned.drop(columns=zero_columns)

    # 기호 처리
    missing_values = ["..", "N/A", "#N/A", "없음", "무", "NONE"]
    df_cleaned.replace(missing_values, "", inplace=True)
    
    return df_cleaned



def list_preprocessed_columns(cleaned_df):
    columns = cleaned_df.columns.tolist()
    print(f"Preprocessed columns: {columns}")
    return columns



if __name__ == "__main__":
    file_path = '/home/pjtl2w01admin/csm/graphDB_pjt/data/whole_data/Table_KNA1.XLSX'
    df = read_excel_file(file_path)
    df_cleaned, modified_subset, column_changes = apply_column_name_cleaning(df)
    print("Column changes:", column_changes, "Modified subset:", modified_subset, "DataFrame shape:", df_cleaned.head(5))
    df_cleaned, modified_subset = date_time_data_to_string(df_cleaned)
    print("Modified subset after date/time conversion:", modified_subset, "DataFrame shape after conversion:", df_cleaned.head(5))
    df_cleaned, modified_subset = remove_empty_columns(df_cleaned)
    print("Modified subset after removing empty columns:", modified_subset, "DataFrame shape after removal:", df_cleaned.head(5))
    #df_cleaned = preprocess_data(df)
    #preprocessed_columns = list_preprocessed_columns(df_cleaned)