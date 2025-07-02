import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import traceback


def mineral_data_crawling():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    # options.add_argument('--headless')  # 화면 없이 실행하려면 이 옵션을 활성화

    try:
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.komis.or.kr/Komis/MnrlIndc/PricePred")
        
        wait = WebDriverWait(driver, 6)
        time.sleep(2)

        # '유연탄' 버튼 클릭
        try:
            #XPath로 직접 찾기
            coal_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@name='mnrkndUnqRadioCd' and @value='MNRL0032']")))
            print("유연탄 버튼 찾음 (방법 1)")
        except TimeoutException:
            #JavaScript로 요소 찾기
            print("방법 1 실패, 방법 2 시도 중...")
            coal_button = driver.execute_script("return document.querySelector('input[name=\"mnrkndUnqRadioCd\"][value=\"MNRL0032\"]')")
            driver.execute_script("arguments[0].click();", coal_button)
            print("유연탄 버튼 찾음 (방법 2)")
        
        time.sleep(1)
        
        # '중기(3년 분기예측)' 선택
        try:
            select_year_box = wait.until(EC.element_to_be_clickable((By.ID, "srchPrdSeCd")))
            select_year_box.click()
            time.sleep(1)
            
            option_3yrs_box = wait.until(EC.element_to_be_clickable((By.XPATH, "//option[@value='S']")))
            option_3yrs_box.click()
            print("중기(3년) 옵션 선택 완료")
            time.sleep(1)
        except TimeoutException:
            print("중기(3년) 옵션 선택 실패, JavaScript로 시도...")
            driver.execute_script("document.getElementById('srchPrdSeCd').value = 'S';")
            driver.execute_script("document.getElementById('srchPrdSeCd').dispatchEvent(new Event('change'));")
        
        # 적용 버튼 클릭 
        try:
            apply_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '적용')]")))
            apply_button.click()
            print("적용 버튼 클릭 완료")
        except TimeoutException:
            print("적용 버튼 찾기 실패, 다른 방법 시도...")
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if "적용" in button.text:
                    driver.execute_script("arguments[0].click();", button)
                    print(f"적용 버튼 찾음: {button.text}")
                    break
        
        time.sleep(1)
        
        # 테이블 데이터 크롤링
        try:
            print("테이블 데이터 수집 시작...")
            table = wait.until(EC.presence_of_element_located((By.ID, "resultMnrkTable")))
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            data = []
            for i, row in enumerate(rows[1:]):  # 첫 번째 행은 헤더이므로 제외
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 2:
                    period = cols[0].text.strip()
                    price = cols[1].text.strip()
                    data.append([period, price])
                    print(f"행 {i+1}: {period} - {price}")
            
            # DataFrame으로 변환 후 저장
            df = pd.DataFrame(data, columns=["기간", "가격(USD/ton)"])
            df.to_csv("coal_price_prediction.csv", index=False, encoding="utf-8-sig")
            
            # 데이터 확인
            print("CSV 저장 완료: coal_price_prediction.csv")
            print(df)
            
        except TimeoutException:
            print("테이블을 찾을 수 없습니다. 페이지 소스 확인...")
            page_source = driver.page_source
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            print("페이지 소스를 'page_source.html'에 저장했습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
        print(traceback.format_exc())

    finally:
        # 드라이버 종료
        try:
            if 'driver' in locals():
                driver.quit()
                print("드라이버가 성공적으로 종료되었습니다.")
        except Exception as e:
            print(f"드라이버 종료 중 오류 발생: {e}")