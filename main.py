# pip install streamlit fbprophet yfinance plotly
# streamlit run main.py
# python 3 -m venv virtual
# source virtual/bin/activate

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
import pandas as pd
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


START = "2020-01-01"  #改合适的日期，分疫情前后数据预测，2020年几月为开始；再出一个没有疫情影响的预测
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Predict Foreign Exchange Rate from 6893 Big Data Team 41')

currency = ('USDCNY=X', 'USDJPY=X', 'USDGBP=X', 'DOGE-USD')
selected_currency = st.selectbox('Select exchange rate from yahoo for prediction', currency)

n_weeks = st.slider('weeks of prediction:', 1, 105) #最好改成predict多少天 改下面一行就行
period = n_weeks * 7

#ticker就是y_finance的货币名
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Load data..')
data = load_data(selected_currency)
data_load_state.text('show data')

#配合前面直接能得到data。怎么找到和本地算法相同的features 怎么从雅虎取到合适的值。
#st.subheader('Raw data')
st.subheader('Existing data with features for prediction')
st.write(data.tail())



def plot_raw_data():
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="currency_open"))
     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="currency_close"))
     fig.layout.update(title_text='Time Series data, please choose the range you would like to vizualize', xaxis_rangeslider_visible=True)
     st.plotly_chart(fig)


plot_raw_data()

#设计控制每天forecast的频率


# Predict forecast with Prophet.
df_train = data[['Date','Close']]     #挑选需要训练的数据(没有找到测试数据？)
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})   #找fbprophet网站学习。画出了不同的attribute来分析

#节假日影响因素
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

#说是用了sklearn模型训练
m = Prophet(holidays=holidays)
m.add_country_holidays(country_name='US')
m.fit(df_train)
future = m.make_future_dataframe(periods=period)  #预测的future数据集
forecast = m.predict(future)


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_weeks} weeks')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

##calculate mse etc
df_cv = cross_validation(m, initial='360 days', period='90 days', horizon = '180 days')
#df_cv = cross_validation(m, initial=2020-1-1, period= period, horizon = 2021-1-1)
st.subheader("model evaluation")
st.write(df_cv.tail())
#df_p = performance_metrics(df_cv)
#fig4 = plot_cross_validation_metric(df_cv, metric='mse')
#st.write(fig4)
fig6 = plot_cross_validation_metric(df_cv, metric='rmse')
st.write(fig6)
fig7 = plot_cross_validation_metric(df_cv, metric='mae')
st.write(fig7)
fig8 = plot_cross_validation_metric(df_cv, metric='mape')
st.write(fig8)
fig9 = plot_cross_validation_metric(df_cv, metric='coverage')
st.write(fig9)
#st.plotly_chart(fig4)
#st.write(df_p.head(10))

#修改figure放在论文里体现得出数据的效果

#试试heroku部署
#先git init. 用gui部署到git，再 heroku create


##对照显示covid的预测
division = 'country'  #regional data is available for some countries
region = 'United States'
#region = 'China'
prediction = 'ConfirmedCases' #ConfirmedDeaths is also available for forecasting.

#get the latest data from OxCGRT
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
full_df = pd.read_csv(DATA_URL,
                usecols=['Date','CountryName','RegionName','Jurisdiction',
                           'ConfirmedCases','ConfirmedDeaths'],
                parse_dates=['Date'],
                encoding="ISO-8859-1",
                dtype={"RegionName": str,
                        "CountryName":str})

#Filter the region we want to predict
if division == 'country':
    df = full_df[(full_df['Jurisdiction'] == 'NAT_TOTAL') & (full_df['CountryName'] == region)][:-1]
elif division == 'state':
    df = full_df[(full_df['Jurisdiction'] == 'STATE_TOTAL') & (full_df['RegionName'] == region)][:-1]

#Since we are not using exogenous variables, we just keep the dates and endogenous data
df = df[['Date',prediction]].rename(columns = {'Date':'ds', prediction:'y'})

#展现从url拉到的data
st.subheader('US covid data from url')
st.write(df.tail())



# set how many days to forecast

# instantiate and fit the model
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=period)
forecast2 = m.predict(future)

# create the prediction dataframe 'forecast_length' days past the fit data


to_plot = forecast2[forecast2.ds > '2020-12-01'].merge(df, how='left')


#展现新冠预测数据集
st.subheader('US covid cases prediction')
st.write(forecast2.tail())

st.subheader("visualized US covid cases prediction")
fig3 = plot_plotly(m, forecast2)
st.plotly_chart(fig3)







#第二个国家的covid
##对照显示covid的预测


division = 'country'  #regional data is available for some countries
#region = 'United States'
#region = 'China'

if selected_currency == "USDCNY=X":
    region = 'China'
elif selected_currency == "USDJPY=X":
    region = 'Japan'
elif selected_currency == "USDGBP=X":
    region = 'United Kingdom'
else:
    region = 'United States'


prediction = 'ConfirmedCases' #ConfirmedDeaths is also available for forecasting.

#get the latest data from OxCGRT
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
full_df = pd.read_csv(DATA_URL,
                usecols=['Date','CountryName','RegionName','Jurisdiction',
                           'ConfirmedCases','ConfirmedDeaths'],
                parse_dates=['Date'],
                encoding="ISO-8859-1",
                dtype={"RegionName": str,
                        "CountryName":str})

#Filter the region we want to predict
if division == 'country':
    df = full_df[(full_df['Jurisdiction'] == 'NAT_TOTAL') & (full_df['CountryName'] == region)][:-1]
elif division == 'state':
    df = full_df[(full_df['Jurisdiction'] == 'STATE_TOTAL') & (full_df['RegionName'] == region)][:-1]

#Since we are not using exogenous variables, we just keep the dates and endogenous data
df = df[['Date',prediction]].rename(columns = {'Date':'ds', prediction:'y'})

#展现从url拉到的data(可以加)
#st.subheader('the chosen foreign country covid data from url')
#st.write(df.tail())



# set how many days to forecast

# instantiate and fit the model
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=period)
forecast3 = m.predict(future)

# create the prediction dataframe 'forecast_length' days past the fit data


to_plot = forecast3[forecast3.ds > '2020-12-01'].merge(df, how='left')


#展现新冠预测数据集
st.subheader('the chosen foreign country covid cases prediction')
st.write(forecast3.tail())

st.subheader("visualized the chosen foreign country covid cases prediction")
fig5 = plot_plotly(m, forecast3)
st.plotly_chart(fig5)


#算mse(目前有问题--直接官网)
#fb_pre = np.array(forecast['yhat'].iloc[2034:2905])#2034到2905是前面30%的测试集所对应的数据范围
#MSE = true_data - fb_pre
#MSE = MSE*MSE
#MSE_loss = sum(MSE)/len(MSE)


#df_cv = cross_validation(m, initial='360 days', period='90 days', horizon = '180 days')
#st.write(df_cv.tail())
#df_p = performance_metrics(df_cv)
#st.write(df_p.head(10))