# base 파일은 ACE_Streamlit_v1.py

import pandas as pd
import numpy as np
import streamlit as st
# import plost 
import plotly.express as px
from chatbot_module import render_chatbot, initialize_session_state


pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.options.display.float_format = '{:,.2f}'.format

st.set_page_config(
    page_title="INFORACTIVE-손익예측",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

col = st.columns((1.5, 5, 1.5), gap='large')

# data import
# streamlit 사용할 때 cache 적용할 것
def importFile():
    df_sales = pd.read_csv('data/salesAll.csv', dtype=str)
    df_purchase = pd.read_csv('data/purchaseAll.csv', dtype=str)
    df_expenses = pd.read_csv('data/ExpensesAll.csv', dtype=str)
    df_produce = pd.read_csv('data/produceAll.csv', dtype=str)
    df_rawCost = pd.read_csv('data/rawCostAll.csv', dtype=str)
    df_activity = pd.read_csv('data/activityAll.csv', dtype=str)
    df_subul = pd.read_csv('data/subulAll.csv', dtype=str)
    df_subul_zero = pd.read_csv('data/df_subul_zero.csv')
    return dict(
        df_sales=df_sales,
        df_purchase=df_purchase,
        df_expenses=df_expenses,
        df_produce=df_produce,
        df_rawCost=df_rawCost,
        df_activity=df_activity,
        df_subul=df_subul,
        df_subul_zero=df_subul_zero
    )

dataset=importFile()

df_sales=dataset['df_sales']
df_purchase=dataset['df_purchase']
df_expenses=dataset['df_expenses']
df_produce=dataset['df_produce']
df_rawCost=dataset['df_rawCost']
df_activity=dataset['df_activity']
df_subul=dataset['df_subul']
df_subul_zero=dataset['df_subul_zero']

### streamlit control
m00 = pd.pandas.read_excel('m00-index.xlsx')
m00.rename(columns=lambda x: x.strip(), inplace=True) 
m00 = m00.map(lambda x: x.strip() if isinstance(x, str) else x)

with st.sidebar:
    st.title('INFORACTIVE-손익예측')
    
    expander1 = st.expander("🔢 Parameter Controls", expanded=True)
    
    
    # 세션 상태에서 option 값 가져와 selectbox 초기값으로 설정
    fert_options = m00.loc[(m00["kind"]=="FERT")].iloc[:, 0].tolist()
    roh_options = m00.loc[(m00["kind"]=="ROH")].iloc[:, 0].tolist()
    fert_default_idx = fert_options.index(st.session_state.get('option', 'FERT101')) if st.session_state.get('option', 'FERT101') in fert_options else 0
    roh_default_idx = roh_options.index(st.session_state.get('option2', 'ROH0001')) if st.session_state.get('option2', 'ROH0001') in roh_options else 0
    
    option = expander1.selectbox(
        '제품 선택', 
        fert_options,
        index = fert_default_idx
    )
    # UI에서 값이 변경되면 세션 상태도 업데이트
    if option != st.session_state.get('option'):
        st.session_state.option = option
    
    # 입고 수량 변화% - 세션 상태 값 사용하여 슬라이더 초기화
    m10change = expander1.slider(
        '입고 수량 변화%', 
        -12.0, 12.0, 
        float(st.session_state.get('m10change', 0.0)), 
        1.0
    )/100

    m20change = expander1.slider(
        '판매 수량 변화%',
        -12.0, 12.0,
        float(st.session_state.get('m20change', 0.0))
    )

    m110change = expander1.slider(
        'USD 환율 변화%',
        -12.0, 12.0,
        float(st.session_state.get('m110change', 0.0))
    )
    
    expander1.divider()

    option2 = expander1.selectbox(
        '원재료 선택', 
        roh_options,
        index = roh_default_idx
    )
    # UI에서 값이 변경되면 세션 상태도 업데이트
    if option2 != st.session_state.get('option2'):
        st.session_state.option = option2
    
    m50change2 = expander1.slider(
        '원재료 구매단가 변화%',
        -12.0, 12.0,
        float(st.session_state.get('m50change2', 0.0))
    )
    
    
    color_theme_list = px.colors.named_colorscales()
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list,index=4)

    # option = expander1.selectbox('제품 선택', m00.loc[(m00["kind"]=="FERT")])
    # m10change = expander1.slider('입고 수량 변화%', -12.0, 12.0, 0.0, 1.0)/100
    #m20change = expander1.slider('판매 수량 변화%', -12.0, 12.0, 0.0, 1.0)/100
    #m110change = expander1.slider('USD 환율 변화%', -12.0, 12.0, 0.0, 1.0)/100
    #option2 = expander1.selectbox('원재료 선택', m00.loc[(m00["kind"]=="ROH")])
    #m50change2 = expander1.slider('원재료 구매단가 변화%', -12.0, 12.0, 0.0,1.0)/100

### 시뮬레이션 파라미터 지정 (엑셀처럼 한개씩만) - 나중에 앱이 완성 되면 여러개 동시에 하도록 해야 함, 어차피 프로그램은 여러개 업데이트할 수 있도록 설계. - 즉 %를 아예 컬럼으로 넣었음 ㅎㅎ - 완전히 느리지만, 확실한 오리지널 방법 사용한 툴이다.

#1. 환율
s1통화종류="USD"
s1환율per=m110change #percent

#2. 구매단가
s2구매단가템=option2
s2구매단가per=m50change2

#3. 판매수량
s3판매수량템=option
s3판매수량per=m20change

#4. 판매단가
s4판매단가템=option
s4판매단가per=0

#5. 생산수량
s5생산수량템=option
s5생산수량per=m10change

#6. 비용계획
s6비용오리오브젝트="102001"
s6비용오리원가요소="53110010"
s6비용KEY=s6비용오리오브젝트 + s6비용오리원가요소
s6비용per=0


### Sales data - final (s1, s3, s4)

df_sales.rename(columns=lambda x: x.strip(), inplace=True) 
df_sales = df_sales.map(lambda x: x.strip() if isinstance(x, str) else x)

df_sales=df_sales.iloc[:,0:8]

df_sales[["수량","금액"]] = df_sales[["수량",'금액']].apply(pd.to_numeric) #to numeric

new_cols=['환율%','수량%','단가%', '총변화%', '최종수량','최종금액']

df_sales.loc[:,new_cols]=0.0

df_sales.loc[df_sales['현지통화'] == s1통화종류, '환율%'] = s1환율per/100
df_sales.loc[df_sales['자재'] == s3판매수량템, '수량%'] = s3판매수량per/100
df_sales.loc[df_sales['자재'] == s4판매단가템, '단가%'] = s4판매단가per/100
df_sales['총변화%']=(1+df_sales['환율%'])*(1+df_sales['수량%'])*(1+df_sales['단가%'])-1
df_sales['최종수량']=df_sales['수량']*(df_sales['수량%']+1)
df_sales['최종금액']=df_sales['금액']*(df_sales['총변화%']+1)
df_sales_f=df_sales.copy()



### Purchase data - final (s1, s2)
df_purchase.rename(columns=lambda x: x.strip(), inplace=True) 
df_purchase = df_purchase.map(lambda x: x.strip() if isinstance(x, str) else x)

df_purchase=df_purchase.iloc[:,0:11]

df_purchase[["수량","금액", "현지통화금액","원화단가"]] = df_purchase[["수량","금액", "현지통화금액","원화단가"]].apply(pd.to_numeric) #to numeric

new_cols=['환율%','단가%', '총변화%', '최종단가','최종금액']

df_purchase.loc[:,new_cols]=0.0

df_purchase.loc[df_purchase['현지통화'] == s1통화종류, '환율%'] = s1환율per/100
df_purchase.loc[df_purchase['자재'] == s2구매단가템, '단가%'] = s2구매단가per/100
df_purchase['총변화%']=(1+df_purchase['환율%'])*(1+df_purchase['단가%'])-1
df_purchase['최종단가']=df_purchase['원화단가']*(df_purchase['총변화%']+1)
df_purchase['최종금액']=df_purchase['금액']*(df_purchase['총변화%']+1)
df_purchase_f=df_purchase.copy()


### expenses data - final (s6)

df_expenses.rename(columns=lambda x: x.strip(), inplace=True) 

df_expenses = df_expenses.map(lambda x: x.strip() if isinstance(x, str) else x)

df_expenses["금액"] = pd.to_numeric(df_expenses["금액"])
df_expenses=df_expenses.iloc[:,0:15]


# 빈곳들이 많아서 concatenate 말고 fillna로 빈곳 채운 후 sum 사용해서 string 더해줌
df_expenses_s1=df_expenses[['오리진오브젝트', '오리진원가요소']].fillna('').sum(axis=1)
df_expenses_s2=df_expenses[['오브젝트', '액티비티유형']].fillna('').sum(axis=1) # 요게 오른쪽으로 스트링 더하기임
df_expenses['KEY1']=df_expenses_s1
df_expenses['KEY2']=df_expenses_s2

df_expenses['금액변화%']=0.0
df_expenses.loc[df_expenses['KEY1'] == s6비용KEY, '금액변화%']=s6비용per/100

df_expenses['최종금액']=-(1+df_expenses['금액변화%'])*df_expenses['금액']
df_expenses['DIFF']=df_expenses['최종금액']+df_expenses['금액']
df_expenses_f=df_expenses.copy()



### produce Data - final (s5)

df_produce.rename(columns=lambda x: x.strip(), inplace=True) 
df_produce = df_produce.map(lambda x: x.strip() if isinstance(x, str) else x)

df_produce["MENGE"] = df_produce["MENGE"].apply(pd.to_numeric)

df_produce=df_produce.iloc[:,0:6]



df_produce["수량%"]=0.0
df_produce.loc[df_produce['MATNR'] == s5생산수량템, '수량%'] = s5생산수량per/100

df_produce["최종수량"]=df_produce["MENGE"]*(1+df_produce["수량%"])
df_produce["DIFF"]=df_produce["최종수량"]-df_produce["MENGE"]
df_produce_f=df_produce.copy()


### 원가계산서 1단계 (s5) - 수량(최종수량)까지
df_rawCost.rename(columns=lambda x: x.strip(), inplace=True) 
df_rawCost = df_rawCost.map(lambda x: x.strip() if isinstance(x, str) else x)
df_rawCost[["MENGE","HWGES"]] = df_rawCost[["MENGE",'HWGES']].apply(pd.to_numeric)
df_rawCost=df_rawCost.iloc[:,0:19]
df_rawCost_s1=df_rawCost[['IDNRK', 'PSWRK','KOSTL','LSTAR']].fillna('').sum(axis=1)
df_rawCost["KEY"]=df_rawCost_s1
df_rawCost['수량%']=0.0
df_rawCost.loc[df_rawCost['MATNR'] == s5생산수량템, '수량%']=s5생산수량per/100
df_rawCost["최종수량"]=df_rawCost["MENGE"]*(1+df_rawCost["수량%"])
df_rawCost_01=df_rawCost.copy()


# df_rawCost_01[['TYPPS','KEY']]['TYPPS']

### 액티비티 단가 1단계 - 수량 업데이트 (#3 in step)
df_activity.rename(columns=lambda x: x.strip(), inplace=True)
df_activity = df_activity.map(lambda x: x.strip() if isinstance(x, str) else x)
df_activity["단가"] = df_activity["단가"].astype(float)
df_activity=df_activity.iloc[:,0:3]
df_activity_s1=df_activity[['오브젝트유형', '액티비티유형']].fillna('').sum(axis=1)
df_activity["KEY1"]=df_activity_s1

temp_02=[]
for ii in df_activity['KEY1']:
    aa=df_rawCost_01.loc[df_rawCost_01['KEY'] == ii, '최종수량'].sum()
    temp_02.append(aa)

df_activity['수량']=temp_02

df_activity_01=df_activity.copy()

### 액티비티 final - 금액, 단가, 소온 업데이트 (#4 in step)

temp_03=[]
for ii in df_activity['KEY1']:
    aa=df_expenses_f.loc[df_expenses_f['KEY2'] == ii, '최종금액'].sum()
    temp_03.append(aa)

df_activity_01['금액']=temp_03

# 단가
df_activity_01["최종단가"]=0.0
df_activity_01.loc[df_activity_01['수량'] != 0.0, '최종단가']=df_activity_01['금액']/df_activity_01['수량']

# DIFF
df_activity_01['DIFF'] = df_activity_01['단가']-df_activity_01['최종단가']

df_activity_01['수량없는 금액']=0.00
df_activity_01.loc[df_activity_01['수량'] == 0,'수량없는 금액']=df_activity_01['금액']

df_activity_f=df_activity_01.copy()

### 원가계산서 2단계 - 가공비 업데이트 
df_rawCost_01['최종금액']=0.0

temp_04=[]
for index, row in df_rawCost_01.iterrows():
    if row['TYPPS']=="E":
        aa=df_activity_f.loc[df_activity_f['KEY1'] == row['KEY'], '최종단가'].sum()*100*row['최종수량']
    else:
        aa=0.0    
    temp_04.append(aa)

df_rawCost_01['최종금액']=temp_04

df_rawCost_02=df_rawCost_01.copy()


### Rollup  1단계 (수불부 누적재고 까지)
# 수불부 1단계 - M 금액을 0로 두고 나머지는 완성하기,  사실 이거 0으로 안둬도 된다, 그냥 맘대로 되게 두자, 어차피 수렴한다.
# --> 1단계를 함수로 정의하자  roll_up_01()

df_subul.rename(columns=lambda x: x.strip(), inplace=True) 
df_subul = df_subul.map(lambda x: x.strip() if isinstance(x, str) else x)

df_subul=df_subul.iloc[:,0:32]
numColumns=df_subul.columns[6:32]
df_subul[numColumns] = df_subul[numColumns].apply(pd.to_numeric).fillna(0)


def roll_up_01():
    # 여기 나중에 기초재고수량 데이타 가져오는 것 온다
    pur01, pur02, pro01, pro02=[], [], [], []

    for ii in df_subul['자재']:
        # 구매입고
        aa=df_purchase_f.loc[df_purchase_f['자재'] == ii, '수량'].sum()
        bb=df_purchase_f.loc[df_purchase_f['자재'] == ii, '최종금액'].sum()
        pur01.append(aa)
        pur02.append(bb)

        #생산입고 - 여기서 계산은 하지만 나중에 금액은 일단 0으로 둔다.
        cc=df_produce_f.loc[df_produce_f['MATNR'] == ii, '최종수량'].sum()
        dd=df_rawCost_02.loc[df_rawCost_02['MATNR'] == ii, '최종금액'].sum()
        pro01.append(cc)
        pro02.append(dd)


    df_subul['구매입고 수량']=pur01
    df_subul['구매입고 금액']=pur02
    df_subul['생산입고 수량']=pro01
    df_subul['생산입고 금액']=pro02
    
    inQTYcolumns = ['기초재고 수량', '구매입고 수량', '생산입고 수량', '외주입고 수량', '이전입고 수량', '기타입고 수량']
    inCOSTcolumns = ['기초재고 금액', '구매입고 금액', '생산입고 금액', '외주입고 금액', '이전입고 금액', '기타입고 금액']

    df_subul['누적재고 수량']=df_subul[inQTYcolumns].sum(axis=1)
    df_subul['누적재고 금액']=df_subul[inCOSTcolumns].sum(axis=1)

roll_up_01()

### Rollup 02 - roll up 2단계
# 원각계산서 M  업데이트

def roll_up_02():
    temp_04=[]
    for index, row in df_rawCost_02.iterrows():
        if row['TYPPS']=="M":
            aa1=df_subul.loc[df_subul['자재'] == row['IDNRK'], '누적재고 금액'].sum()
            aa2=df_subul.loc[df_subul['자재'] == row['IDNRK'], '누적재고 수량'].sum() #이거 나중에 미리 단가로 만들어 놔라
            aa=aa1/aa2*row['최종수량']
        else:
            aa=row['최종금액']
        temp_04.append(aa)

    df_rawCost_02['최종금액']=temp_04

roll_up_02()
roll_up_01()
roll_up_02()
roll_up_01()
roll_up_02()
roll_up_01()

### finalizing subuls
temp_04=[]
for ii in df_subul['자재']:
    # 판매출고 수량
    aa=df_sales_f.loc[df_sales_f['자재'] == ii, '최종수량'].sum()
    temp_04.append(aa)



temp_05=[]
for ii in df_subul['자재']:
    # 공정출고 수량
    aa=df_rawCost_02.loc[df_rawCost_02['IDNRK'] == ii, '최종수량'].sum()
    temp_05.append(aa)

#판매출고 수량, 금액, 공정출고 수량, 금액,기말재고 수량, 금액
df_subul['판매출고 수량']=temp_04
df_subul['판매출고 금액']=df_subul['누적재고 금액']/df_subul['누적재고 수량']*df_subul['판매출고 수량']

df_subul['공정출고 수량']=temp_05
df_subul['공정출고 금액']=df_subul['누적재고 금액']/df_subul['누적재고 수량']*df_subul['공정출고 수량']

df_subul['기말재고 수량']=df_subul['누적재고 수량']-df_subul['판매출고 수량']-df_subul['공정출고 수량']-df_subul['외주출고 수량']-df_subul['이전출고 수량']-df_subul['기타출고 수량']
df_subul['기말재고 금액']=df_subul['누적재고 금액']-df_subul['판매출고 금액']-df_subul['공정출고 금액']-df_subul['외주출고 금액']-df_subul['이전출고 금액']-df_subul['기타출고 금액']


### 손익계산서 요약본
result= pd.DataFrame([{'전':27360000000.0 , '후': 0.0, '차이':0.0}, {'전': 20309653406.2695  , '후': 0.0,'차이':0.0}, {'전': 7050346593.73051, '후': 0.0,'차이':0.0 }], index=['매출액', '매출원가', '매출이익'])
result.at['매출액', '후'] = df_sales_f['최종금액'].sum()
result.at['매출원가', '후'] = df_subul['판매출고 금액'].sum()
result.at['매출이익', '후']= result.at['매출액', '후']-result.at['매출원가', '후']
result.at['매출액','차이']= result.at['매출액', '후']-result.at['매출액', '전']
result.at['매출원가','차이']= result.at['매출원가', '후']-result.at['매출원가', '전']
result.at['매출이익','차이']= result.at['매출이익', '후']-result.at['매출이익', '전']
#pd.options.display.float_format = '{:,.0f}'.format
result.index.name = '손익계산서-요약'

### 제조원가 명세서 요약본
result2= pd.DataFrame([{'전':18399680000.0 , '후': 0.0, '차이':0.0}, {'전':7634640.0 , '후': 0.0,'차이':0.0},
                       {'전':5247348536.0 , '후': 0.0, '차이':0.0}, {'전':23654663176.0 , '후': 0.0,'차이':0.0}, 
                       {'전':2755275162.73051 , '후': 0.0, '차이':0.0}, {'전':589734607.0 , '후': 0.0,'차이':0.0}, 
                       {'전':0 , '후': 0.0, '차이':0.0}, {'전':20309653406.2695 , '후': 0.0,'차이':0.0},  
                       {'전': 20309653406.2695, '후': 0.0,'차이':0.0 }], 
                       index=['원재료비', '부재료비', '가공비', '당기제조비용', '재공품', '액티비티미배부','액티비티단수차', '당기제품제조원가', '수불부'])
result2.at['원재료비', '후'] = df_subul.loc[df_subul['평가클래스'] == '3000', '공정출고 금액'].sum()
result2.at['부재료비', '후'] = df_subul.loc[df_subul['평가클래스'] == '3010', '공정출고 금액'].sum()
result2.at['가공비', '후'] = df_expenses_f['최종금액'].sum()*100
result2.at['당기제조비용', '후']= result2.at['원재료비', '후']+result2.at['부재료비', '후']+result2.at['가공비', '후']
wip01= df_subul.loc[df_subul['평가클래스']=='7900', '생산입고 금액'].sum()
wip02= df_subul.loc[df_subul['평가클래스']=='7900', '공정출고 금액'].sum()
result2.at['재공품', '후'] = wip01-wip02
result2.at['액티비티미배부','후']= df_activity_f['수량없는 금액'].sum()*100
result2.at['액티비티단수차','후']= result2.at['가공비', '후']-df_rawCost_02.loc[df_rawCost_02['TYPPS']=='E','최종금액' ].sum()-result2.at['액티비티미배부','후']
result2.at['당기제품제조원가','후']= result2.at['당기제조비용', '후']-result2.at['재공품', '후']-result2.at['액티비티미배부','후']-result2.at['액티비티단수차','후']
result2.at['수불부','후']= df_subul.loc[df_subul['평가클래스'] == '7920', '생산입고 금액'].sum()

result2.at['원재료비', '차이'] = result2.at['원재료비', '후']-result2.at['원재료비', '전']
result2.at['부재료비', '차이'] = result2.at['부재료비', '후']-result2.at['부재료비', '전']
result2.at['가공비', '차이'] = result2.at['가공비', '후']-result2.at['가공비', '전']
result2.at['당기제조비용', '차이'] = result2.at['당기제조비용', '후']-result2.at['당기제조비용', '전']
result2.at['재공품', '차이'] = result2.at['재공품', '후']-result2.at['재공품', '전']
result2.at['액티비티미배부', '차이'] = result2.at['액티비티미배부', '후']-result2.at['액티비티미배부', '전']
result2.at['액티비티단수차', '차이'] = result2.at['액티비티단수차', '후']-result2.at['액티비티단수차', '전']
result2.at['당기제품제조원가', '차이'] = result2.at['당기제품제조원가', '후']-result2.at['당기제품제조원가', '전']
result2.at['수불부', '차이'] = result2.at['수불부', '후']-result2.at['수불부', '전']
result2.index.name="제조원가명세서-요약"

### 명세서 풀버전
df_detail = pd.DataFrame(index=range(29), columns=["L1","L2","Class", "금액", "내역"])
df_detail=df_detail.fillna("")
df_detail.iloc[[0,5,10,17,18,21,24,26,28],0]=["원재료비", "부재료","경비","당기제조비용","재공품","타계정","당기제품제조원가","수불부","차이"]
df_detail.iloc[0:24,1]=["", "기초","입고","타계정출고","기말","","기초","입고","타계정","기말","","노무비","유틸리티",
                        "감가상각비","수선비","폐수","소모품비","","","기초반제품","기말반제품","","액티비티미배부","액티비티단수차"]
df_detail.iloc[0:27,2]=["","3000","3000","3000","3000","","3010","3010","3010","3010","",
                        "1000","2000","3000","4000","5000","6000","","","7900","7900","","","","","","7920"]
df_detail.iloc[0:29,4]=["기초+입고-타계정출고-기말","","","","","기초+입고-타계정출고-기말","","","","","비용계획","","","","","","",
                        "원재료비+부재료비+경비","","","","액티비티미배부+액티비티단수차","","","당기제조비용+재공품-타계정","",
                        "수불부 7920 생산입고","","당기제품제조원가-수불부"]

df_detail['금액'].apply(pd.to_numeric)
df_detail.iat[1,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[1,2], "기초재고 금액"].sum()
df_detail.iat[2,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[2,2], "구매입고 금액"].sum()
df_detail.iat[3,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[3,2], "기타출고 금액"].sum()
df_detail.iat[4,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[4,2], "기말재고 금액"].sum()
df_detail.iat[0,3]=df_detail.iat[1,3]+df_detail.iat[2,3]-df_detail.iat[3,3]-df_detail.iat[4,3]

df_detail.iat[6,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[6,2], "기초재고 금액"].sum()
df_detail.iat[7,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[7,2], "구매입고 금액"].sum()
df_detail.iat[8,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[8,2], "기타출고 금액"].sum()
df_detail.iat[9,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[9,2], "기말재고 금액"].sum()
df_detail.iat[5,3]=df_detail.iat[6,3]+df_detail.iat[7,3]-df_detail.iat[8,3]-df_detail.iat[9,3]

df_detail.iat[11,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[11,2], "최종금액"].sum()*100
df_detail.iat[12,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[12,2], "최종금액"].sum()*100
df_detail.iat[13,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[13,2], "최종금액"].sum()*100
df_detail.iat[14,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[14,2], "최종금액"].sum()*100
df_detail.iat[15,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[15,2], "최종금액"].sum()*100
df_detail.iat[16,3]=df_expenses_f.loc[df_expenses_f['액티비티유형']==df_detail.iat[16,2], "최종금액"].sum()*100
df_detail.iat[10,3]=df_detail.iloc[11:17,3].sum()

df_detail.iat[17,3]=df_detail.iat[0,3]+df_detail.iat[5,3]+df_detail.iat[10,3]

df_detail.iat[19,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[19,2], "기초재고 금액"].sum()
df_detail.iat[20,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[20,2], "기말재고 금액"].sum()
df_detail.iat[18,3]=df_detail.iat[19,3]-df_detail.iat[20,3]

df_detail.iat[22,3]=df_activity_01['수량없는 금액'].sum()*100
df_detail.iat[23,3]=(df_detail.iat[10,3]-df_detail.iat[22,3]-df_rawCost_02.loc[df_rawCost_02['TYPPS']=='E','최종금액' ].sum())
df_detail.iat[21,3]=df_detail.iat[22,3]+df_detail.iat[23,3]

df_detail.iat[24,3]=df_detail.iat[17,3]+df_detail.iat[18,3]-df_detail.iat[21,3]

df_detail.iat[26,3]=df_subul.loc[df_subul['평가클래스']==df_detail.iat[26,2], "생산입고 금액"].sum()

df_detail.iat[28,3]=df_detail.iat[24,3]-df_detail.iat[26,3]

#df['cost'] = df['cost'].map('${:,.2f}'.format)
df_detail['금액'] = df_detail['금액'].apply(pd.to_numeric)
#pd.options.display.float_format = '{:,.0f}'.format

df_detail['금액'] = df_detail['금액'].apply(lambda x: round(x, 2))


df_subul_zero=df_subul_zero.iloc[:, 1:]
### df_subul_zero
df_subul_num=df_subul.iloc[:, 6:]
df_subul_diff=df_subul_num - df_subul_zero

### subul difference heatmap
# mask = -0.01< df.my_channel < 0.01
# column_name = 'my_channel'
# df.loc[mask, column_name] = 0

csv2=df_subul_diff.round(1).replace([np.inf, -np.inf], np.nan)
csv3=csv2
csv4=csv3[[ '구매입고 금액', '생산입고 금액' , '누적재고 금액',  '판매출고 금액',  '공정출고 금액', '기말재고 금액']]
index1=pd.Index(['ROH0001', 'ROH0002', 'ROH0003', 'ROH2001', 'ROH2002', 'ROH2003', 'HALB001', 'HALB002', 'HALB003', 'HALB004', 'HALB005', 'FERT101',
 'FERT102', 'FERT103', 'FERT104', 'FERT105', 'FERT106', 'FERT201', 'FERT202', 'FERT203'])

subul_diff=csv4.set_index(index1)


### streamlit

fig=px.imshow(subul_diff,color_continuous_scale=selected_color_theme,text_auto='.f',   
              aspect='auto', width=1200, height=600)
fig.update_layout(
    yaxis = dict( tickfont = dict(size=14)),
    xaxis = dict( tickfont = dict(size=14)),
    hoverlabel = dict(font=dict(size=14)),
)
fig.update_xaxes(showgrid=True, visible=True)
fig.update_yaxes(showgrid=True)

fig.update_coloraxes(colorbar_tickfont = dict(size=20) )
fig.update_traces(textfont_size=14, xgap=1, ygap=1)


Revenue=result.at['매출액','후']
chgRevenue=result.at['매출액','차이']/Revenue
COGS=result.at['매출원가','후']
chgCOGS=result.at['매출원가','차이']/COGS
Profit=result.at['매출이익','후']
chgProfit=result.at['매출이익','차이']/Profit
WIP=result2.at['재공품', '후']
chgWIP=result2.at['재공품', '차이']/WIP

def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'


col = st.columns((1.5, 4, 2), gap='large')

with col[0]:
    st.markdown('#### METRICS')
    st.metric("REVENUE", format_number(Revenue), f'{round(chgRevenue*100,2) } %')
    st.divider()
    st.metric("COGS", format_number(COGS), f'{round(chgCOGS*100,2) } %')
    st.divider()
    st.metric("Gross Profit", format_number(Profit), f'{round(chgProfit*100,2) } %')
    st.divider()
    st.metric("WIP", format_number(WIP), f'{round(chgWIP*100,2) } %')
    st.divider()
    st.button("Export Data")

df_rawCost_02_m1=df_rawCost_02.copy()
df_rawCost_02_m1['수량변화']=df_rawCost_02_m1['최종수량']-df_rawCost_02_m1['MENGE']
df_rawCost_02_m1['금액변화']=df_rawCost_02_m1['최종금액']-df_rawCost_02_m1['HWGES']
df_rawCost_02_m1[['수량변화','금액변화']].apply(lambda x: round(x, 0))

df_rawCost_02_m2=df_rawCost_02_m1[['MATNR','KEY', '최종금액','수량변화', '금액변화']]

with col[1]:
    st.markdown('#### CHANGE DETAILS')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["수불부", "원가계산서", "손익요약", "제조원가명세서-요약", "제조원가명세서"])
    with tab1:
    
        st.write(fig)
    with tab2:
        st.dataframe(df_rawCost_02_m2, width=1200, height=600,
                     column_config={
                        "수량변화": st.column_config.ProgressColumn(
                            "수량변화",
                            format="%.0f",
                            min_value=0,
                            max_value=max(df_rawCost_02_m2.수량변화)+100000,),
                        "금액변화": st.column_config.ProgressColumn(
                            "금액변화",
                            format="%.0f",
                            min_value=0,
                            max_value=max(df_rawCost_02_m2.금액변화)+100000,)}
        )
        
    with tab3:
        st.dataframe(result, width=1200,height=600)

    
    with tab4:
         st.dataframe(result2, width=1200, height=600)

    with tab5:
        st.dataframe(df_detail, width=1200, height=600)

    
    

    
    
m00=pd.read_csv('m00_csv.csv')
m00=m00[['mat','kind']]
m93show=pd.read_csv('m93show_csv.csv')
m93show=m93show.iloc[:,1:]
mm=pd.read_csv('mm_csv.csv')
mm=mm.set_index('mat')

result_m1=result.reset_index()
result_m1=pd.melt(result_m1, id_vars=['손익계산서-요약'],var_name='variable', value_name='value')


# summary bar chart
# fig_m1 = px.bar(result_m1, x="손익계산서-요약", color="variable",
#              y='value',
#              barmode='group',
#              height=300,
#             )

# fig_m1.update_layout(
#     yaxis = dict( tickfont = dict(size=14)),
#     xaxis = dict( tickfont = dict(size=14)),
#     hoverlabel = dict(font=dict(size=14)),
#     uniformtext_minsize=12,
# )

                
initialize_session_state()

    
with col[2]:
    # st.markdown('#### AI AGENT')
    # st.write(fig_m1)

    render_chatbot()

    st.markdown('#### FERT to ROH BOM')

    default_option3_idx = 3
    if 'option3' in st.session_state:
        try:
            current_option3 = st.session_state.option3
            if current_option3 in fert_options:
                default_option3_idx = fert_options.index(current_option3)
        except (ValueError, IndexError):
            print("option3 관련 에러 발생 - 값, 인덱스 관련")
    
    option3 = st.selectbox('제품 선택', fert_options, label_visibility="hidden", index=default_option3_idx)

    #선택된 값을 세션 상태에 저장
    if st.session_state.get('option3') != option3:
        st.session_state.option3 = option3

    
    mpie=((mm.loc[option3, :]).to_frame()).reset_index(drop=False)
    
    fig1 = px.pie(mpie, values=option3,names='index') #names='country',
    # #mpie["haha"]=0.0pie(mpie, values='pop', names='country', title='FERT to ROH BOM')
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

 


