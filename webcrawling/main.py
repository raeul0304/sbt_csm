import pandas as pd
from mineral_data_crawling import mineral_data_crawling
from extract_quarters import extract_year_and_quarters, extract_3_quarters, generate_url_and_dataframe

def main():
    mineral_data_crawling()
    crawled_data = pd.read_csv("coal_price_prediction.csv")
    current_year_and_quarter = extract_year_and_quarters()
    result= extract_3_quarters(current_year_and_quarter, crawled_data)
    url, dataframe = generate_url_and_dataframe(result)
    print(url)
    print(dataframe)

if __name__ == "__main__":
    main()
