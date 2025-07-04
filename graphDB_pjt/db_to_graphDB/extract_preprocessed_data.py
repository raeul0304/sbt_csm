import pandas as pd
from datetime import time
import re

def read_excel_file(file_path):
    return pd.read_excel(file_path, engine='openpyxl')


def make_field_cypher_safe(field_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", field_name.strip())


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
    file_path = 'C:\\Users\\USER\\OneDrive\\Desktop\\SBT GLOBAL\\프로젝트\\GraphDB 프로젝트\\sap data\\Table_KNA1.XLSX'
    df = read_excel_file(file_path)
    df_cleaned = preprocess_data(df)
    preprocessed_columns = list_preprocessed_columns(df_cleaned)