from datetime import datetime
import pandas as pd

def extract_year_and_quarters():
    current_year = str((datetime.now().year))[2:] + "년"
    current_month = datetime.now().month
    current_quarter = ""

    if current_month in [1, 2, 3]:
        current_quarter = "1Q"
    elif current_month in [4, 5, 6]:
        current_quarter = "2Q"
    elif current_month in [7, 8, 9]:
        current_quarter = "3Q"
    else:
        current_quarter = "4Q"
    
    current_year_and_quarter = current_year + " " + current_quarter
    return current_year_and_quarter
    #print(current_year_and_quarter)

def extract_3_quarters(current_year_and_quarter, df):
    row_no = df[df['기간'] == current_year_and_quarter].index[0]
    # 현재 분기 제외, 이전 3개 분기 데이터 추출
    selected_rows = df.iloc[row_no-3:row_no]
    
    selected_rows = selected_rows.sort_index(ascending=False).reset_index(drop=True)
    #print(selected_rows)
    
    return selected_rows

def generate_url_and_dataframe(result):
    url = "https://www.komis.or.kr/Komis/MnrlIndc/PricePred"
    return url, result

# if __name__ == "__main__":
#     df = pd.read_csv("coal_price_prediction.csv")
#     current_year_and_quarter = extract_year_and_quarters()
#     result= extract_3_quarters(current_year_and_quarter, df)
#     generate_url_and_dataframe(result)
    
    
