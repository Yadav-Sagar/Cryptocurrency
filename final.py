import streamlit as st
import pandas as pd
import datetime as dt
from itertools import cycle
import numpy as np
import math
import requests
import time
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARIMA  
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import base64
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# For model building we will use these library

#import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM




hist=pd.read_csv('hist.csv')
st.title('Welcome')
st.write("""
# Here you will find the results of Prediction of Bitcoin Prices Usings Different Models
# """)


col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    file_ = open("giffy.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

with col3:
    st.write("")
st.write("\n")
st.write("\n")
st.markdown("<h1 style='text-align: center;'>Dataframe</h1>", unsafe_allow_html=True)
#st.title('Dataframe')
st.write('The data for the dataframe is collected from the historical price section of the CryptoCompare website using API call ')
st.dataframe(hist)



#2017

def V2017():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
    y_2017 = hist.loc[(hist['Date'] >= '2017-01-01')
                     & (hist['Date'] < '2017-12-31')]

    y_2017 = y_2017.drop(y_2017[['volumefrom','volumeto']],axis=1)
    monthvise= y_2017.groupby(y_2017['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    #Monthwise Comparision between the Stock open and close price
    fig=go.Figure()
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45) 
                    
    #fig.show()

    y_2017.groupby(y_2017['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2017.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2017.groupby(y_2017['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    #Monthwise high and low of the stock
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig1.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig1.update_layout(barmode='group')
                    




    #Stock Analysis Chart
    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2017, x=y_2017.Date, y=[y_2017['open'], y_2017['close'], 
                                            y_2017['high'], y_2017['low']],
                labels={'Date': 'Date','value':'Stock value'})
    #fig3.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)

    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2017</h1>", unsafe_allow_html=True)
    #st.title("Analysis Of Year 2017")
    st.header("Data")
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.dataframe(y_2017)
    with col3:
        st.write("")
    
    st.header('Month-wise comparison between stock open and close price')
    
    st.dataframe(monthvise)
    
    
    st.plotly_chart(fig)
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig1)
    st.markdown("<h2 style='text-align: center;'>Stock Anaysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Anaysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2017 📈📉")




def V2018():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
    y_2018 = hist.loc[(hist['Date'] >= '2018-01-01')
                        & (hist['Date'] < '2018-12-31')]

    y_2018=y_2018.drop(y_2018[['volumefrom','volumeto']],axis=1)



    monthvise= y_2018.groupby(y_2018['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)




    y_2018.groupby(y_2018['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2018.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2018.groupby(y_2018['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig2.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig2.update_layout(barmode='group')


    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2018, x=y_2018.Date, y=[y_2018['open'], y_2018['close'], 
                                            y_2018['high'], y_2018['low']],
                labels={'Date': 'Date','value':'Stock value'})
    #fig3.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)

    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2018</h1>", unsafe_allow_html=True)
    #st.title("Analysis of Year 2018")
    #st.markdown("<h2 style='text-align: center;'>Data</h2>", unsafe_allow_html=True)
    st.header('Data')
    st.dataframe(y_2018)
    #st.markdown("<h2 style='text-align: center;'>Month-wise comparison between stock open and close price</h2>", unsafe_allow_html=True)
    st.header('Month-wise comparison between stock open and close price')
    st.dataframe(monthvise)
    st.plotly_chart(fig)
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig2)
    st.markdown("<h2 style='text-align: center;'>Stock Analysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Analysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2018 📈📉")



#2019
def V2019():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')



    y_2019 = hist.loc[(hist['Date'] >= '2019-01-01')
                        & (hist['Date'] < '2019-12-31')]

    y_2019=y_2019.drop(y_2019[['volumefrom','volumeto']],axis=1)




    monthvise= y_2019.groupby(y_2019['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)


                



    y_2019.groupby(y_2019['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2019.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2019.groupby(y_2019['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig2.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig2.update_layout(barmode='group')


    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2019, x=y_2019.Date, y=[y_2019['open'], y_2019['close'], 
                                            y_2019['high'], y_2019['low']],
                labels={'Date': 'Date','value':'Stock value'})
    #fig3.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)

    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2019</h1>", unsafe_allow_html=True)
    #st.title("Analysis of Year 2019")
    st.header('Data')
    st.dataframe(y_2019,900,900)
    #st.markdown("<h2 style='text-align: center;'>Month-wise comparison between stock open and close price</h2>", unsafe_allow_html=True)
    st.header('Month-wise comparison between stock open and close price')
    st.dataframe(monthvise)
    st.plotly_chart(fig)
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig2)
    st.markdown("<h2 style='text-align: center;'>Stock Analysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Analysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2019 📈📉")

    


#2020
def V2020():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
    y_2020 = hist.loc[(hist['Date'] >= '2020-01-01')
                        & (hist['Date'] < '2020-12-31')]

    y_2020=y_2020.drop(y_2020[['volumefrom','volumeto']],axis=1)

    monthvise= y_2020.groupby(y_2020['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45) 
                

    y_2020.groupby(y_2020['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2020.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2020.groupby(y_2020['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig2.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig2.update_layout(barmode='group') 
                    

    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2020, x=y_2020.Date, y=[y_2020['open'], y_2020['close'], 
                                            y_2020['high'], y_2020['low']],
                labels={'Date': 'Date','value':'Stock value'})
    #fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)

    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2020</h1>", unsafe_allow_html=True)
    #st.title("Analysis of Year 2020")
    st.header('Data')
    st.dataframe(y_2020,900,900)

    st.header('Month-wise comparision between Stock open and close price')
    st.dataframe(monthvise)
    st.plotly_chart(fig)     
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig2)
    st.markdown("<h2 style='text-align: center;'>Stock Analysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Analysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2020 📈📉")


#2021
def V2021():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
    y_2021 = hist.loc[(hist['Date'] >= '2021-01-01')
                        & (hist['Date'] < '2021-12-31')]

    y_2021=y_2021.drop(y_2021[['volumefrom','volumeto']],axis=1)

    monthvise= y_2021.groupby(y_2021['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    y_2021.groupby(y_2021['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2021.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2021.groupby(y_2021['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig2.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig2.update_layout(barmode='group')

    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2021, x=y_2021.Date, y=[y_2021['open'], y_2021['close'], 
                                            y_2021['high'], y_2021['low']],
                labels={'Date': 'Date','value':'Stock value'})
    #fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)
    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2021</h1>", unsafe_allow_html=True)
    #st.title("Analysis of Year 2021")
    st.header('Data')
    st.dataframe(y_2021,900,900)

    st.header('Month-wise comparision between Stock open and close price')
    st.dataframe(monthvise)
    st.plotly_chart(fig)     
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig2)
    st.markdown("<h2 style='text-align: center;'>Stock Analysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Analysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2021 📈📉")

def V2022():
    hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
    y_2022 = hist.loc[(hist['Date'] >= '2022-01-01')
                        & (hist['Date'] < '2022-12-31')]

    y_2022=y_2022.drop(y_2022[['volumefrom','volumeto']],axis=1)

    monthvise= y_2022.groupby(y_2022['Date'].dt.strftime('%B'))[['open','close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    y_2022.groupby(y_2022['Date'].dt.strftime('%B'))['low'].min()
    monthvise_high = y_2022.groupby(hist['Date'].dt.strftime('%B'))['high'].max()
    monthvise_high = monthvise_high.reindex(new_order, axis=0)

    monthvise_low = y_2022.groupby(y_2022['Date'].dt.strftime('%B'))['low'].min()
    monthvise_low = monthvise_low.reindex(new_order, axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig2.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    fig2.update_layout(barmode='group')


    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

    fig3 = px.line(y_2022, x=y_2022.Date, y=[y_2022['open'], y_2022['close'], 
                                            y_2022['high'], y_2022['low']],
                labels={'Date': 'Date','value':'Stock value'})
    fig3.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
    fig3.for_each_trace(lambda t:  t.update(name = next(names)))
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)

    st.markdown("<h1 style='text-align: center;'>Analysis of Year 2022</h1>", unsafe_allow_html=True)
    #st.title("Analysis of Year 2022")
    st.header('Data')
    st.dataframe(y_2022,900,900)

    st.header('Month-wise comparision between Stock open and close price')
    st.dataframe(monthvise)
    st.plotly_chart(fig)     
    st.markdown("<h2 style='text-align: center;'>Month-wise High and Low stock price</h2>", unsafe_allow_html=True)
    #st.header('Month-wise High and Low stock price')
    st.plotly_chart(fig2)
    st.markdown("<h2 style='text-align: center;'>Stock Analysis Chart</h2>", unsafe_allow_html=True)
    #st.header('Stock Analysis Chart')
    st.plotly_chart(fig3)
    st.sidebar.markdown("# Visualization of Year 2022 📈📉")







#st.sidebar.markdown("# Visualization of Year 2017 ❄️")
page_names_to_funcs = {
    "2017": V2017,
    '2018': V2018,
    '2019': V2019,
    '2020': V2020,
    '2021': V2021,
    '2022': V2022
    #"Page 3": page3,
    }

selected_page = st.sidebar.selectbox("Select Year for Visualization", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

df=hist[['close']]
prediction_days=30
df['prediction']=df[['close']].shift(-prediction_days)
X=np.array(df.drop(['prediction'],1))
X=X[:len(df)-prediction_days]
y=np.array(df['prediction'])
#Get all thee values except precdiction days
y=y[:-prediction_days]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
prediction_days_array=np.array(df.drop(['prediction'],1))[-prediction_days:]





# Implementation Of models

# SVM Model

def svm_model():
    
    
    from sklearn.svm import SVR
    svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.00001)
    svr_rbf.fit(X_train,y_train)
    
    svr_rbf_confidence=svr_rbf.score(X_test,y_test)
    svr_rbf_test=svr_rbf.score(X_train,y_train)
    svm_prediction=svr_rbf.predict(prediction_days_array)
    prediction_df=prediction_days_array.reshape(30,)
    df_prediction=pd.DataFrame({'Actual Values': prediction_df, 'Predicted Values': svm_prediction}, columns=['Actual Values', 'Predicted Values'])
    st.title("SVM Model")
    #st.dataframe(df_prediction)
    st.write("SVM Model Accuracy on test data : %f" %svr_rbf_confidence)
    
    st.write('SVM Model Accuracy on train data :  %f' %svr_rbf_test)
    
    
    st.dataframe(df_prediction)
    fig = px.line(df_prediction, y=[df_prediction['Actual Values'],df_prediction['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig)
    st.sidebar.markdown("# SVM Model")






def linear_regression():
    #df=hist[['close']]
    #prediction_days=100
    #df['prediction']=df[['close']].shift(-prediction_days)
    #X=np.array(df.drop(['prediction'],1))
    #X=X[:len(df)-prediction_days]
    #y=np.array(df['prediction'])
#Get all thee values except precdiction days
    #y=y[:-prediction_days]
    
    model=LinearRegression()
    model.fit(X_train,y_train)
    future_set = hist.shift(periods=30).tail(30)
    Linear_prediction = model.predict(prediction_days_array)
    pred_train_linear= model.predict(X_train)
    pred_test_linear= model.predict(X_test)
    train_sqrt=np.sqrt(mean_squared_error(y_train,pred_train_linear))
    train_r2=r2_score(y_train, pred_train_linear)
    test_sqrt=np.sqrt(mean_squared_error(y_test,pred_test_linear))
    test_r2=r2_score(y_test, pred_test_linear)
    train_accuracy=model.score(X_train,y_train)
    test_accuracy=model.score(X_test,y_test)
    prediction_df=prediction_days_array.reshape(30,)
    df_prediction=pd.DataFrame({'Actual Values': prediction_df, 'Predicted Values': Linear_prediction}, columns=['Actual Values', 'Predicted Values'])
    st.header("Linear Regression Model")
    st.write('Accuracy of model on train data: %f' %train_accuracy)
    st.write("Accuracy of model on test data: %f" %test_accuracy)
    st.header("Last 30 Days actual values and predicted values")
    st.dataframe(df_prediction)
    fig = px.line(df_prediction, y=[df_prediction['Actual Values'],df_prediction['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig)
    st.header('Errors')
    st.write("Mean Squared error of train data : %f" %train_sqrt)
    st.write("Coefficient of determination of train data: %f" %train_r2)
    st.write("Mean Squared error of test data: %f" %test_sqrt)
    st.write("Coefficient of determination of test data: %f" %test_r2)
    st.header("After applying Lasso Regression")
    st.write('Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point,')
    

    lasso_model=linear_model.Lasso(alpha =100,max_iter=100,tol=0.1)
    lasso_model.fit(X_train,y_train)
    lasso_test_accuracy=lasso_model.score(X_test,y_test)
    lasso_train_accuracy = lasso_model.score(X_train,y_train)
    pred_test_lasso= lasso_model.predict(prediction_days_array)
    df_prediction=pd.DataFrame({'Actual Values': prediction_df, 'Predicted Values': pred_test_lasso}, columns=['Actual Values', 'Predicted Values'])
    st.write("Accuracy of model on train data : %f" %lasso_train_accuracy)
    st.write("Accuracy of model on test data %f " %lasso_test_accuracy)
    st.dataframe(df_prediction)
    fig = px.line(df_prediction, y=[df_prediction['Actual Values'],df_prediction['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig)


    #Ridge Model
    st.header("After applying Ridge Regression")
    rr = Ridge(alpha=250)
    rr.fit(X_train, y_train)
    pred_train_ridge=rr.predict(X_train)
    pred_test_ridge=rr.predict(X_test)
    r_train_accuracy=rr.score(X_train,y_train)
    r_test_accuracy=rr.score(X_test,y_test)
    pred_test=rr.predict(prediction_days_array)
    df_prediction=pd.DataFrame({'Actual Values': prediction_df, 'Predicted Values': pred_test}, columns=['Actual Values', 'Predicted Values'])
    st.write("Accuracy of model on train data : %f" %r_train_accuracy)
    st.write("Accuracy of model on test data %f " %r_test_accuracy)
    st.dataframe(df_prediction)
    fig = px.line(df_prediction, y=[df_prediction['Actual Values'],df_prediction['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig)

    #Elestci Net
    st.header("After applying Elestic net Regression")
    model_enet = ElasticNet(alpha=100, l1_ratio=0.3)
    model_enet.fit(X_train, y_train)
    pred_train_enet=model_enet.predict(X_train)
    pred_test_enet=model_enet.predict(X_test)
    en_pred_test=model_enet.predict(prediction_days_array)
    en_train_accuracy=model_enet.score(X_train,y_train)
    en_test_accuracy=model_enet.score(X_test,y_test)
    df_prediction=pd.DataFrame({'Actual Values': prediction_df, 'Predicted Values': en_pred_test}, columns=['Actual Values', 'Predicted Values'])
    st.write("Accuracy of model on train data : %f" %r_train_accuracy)
    st.write("Accuracy of model on test data %f " %r_test_accuracy)
    st.dataframe(df_prediction)
    fig = px.line(df_prediction, y=[df_prediction['Actual Values'],df_prediction['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig)

    st.sidebar.markdown("# Linear Regression Model")





    




#ARIMA Model
def arima():
    st.title('ARIMA Model')
    st.write("""
    ARIMA is an acronym for “autoregressive integrated moving average.
    There are two prominent methods of time series prediction: univariate and multivariate. 
          \n Univariate uses only the previous values in the time series to predict future values. 
         \n Multivariate also uses external variables in addition to the series of values to create the forecast.\n
    """)
    data = hist['close']
    Date1 = hist['Date']
    train1 = hist[['Date','close']]
    # Setting the Date as Index
    train2 = train1.set_index('Date')
    train2.sort_index(inplace=True)
    #fig ,ax = plt.subplots()
    #ax.plot(train2)
    #plt.xlabel('Date', fontsize=15)
    #plt.ylabel('Price in USD', fontsize=15)
    #plt.title("Closing price distribution of bitcoin", fontsize=15)
    #plt.show()
    #st.pyplot(fig)

    def test_stationarity(x):


        #Determing rolling statistics
        rolmean = x.rolling(window=22,center=False).mean()

        rolstd = x.rolling(window=12,center=False).std()
        #Perform Dickey Fuller test    
        result=adfuller(x)
        print('ADF Stastistic: %f'%result[0])
        print('p-value: %f'%result[1])
        pvalue=result[1]
        for key,value in result[4].items():
            if result[0]>value:
                print("The graph is non stationery")
                break
            else:
                print("The graph is stationery")
                break;
        
            
    ts = train2['close']      
    test_stationarity(ts)
##    Log Transforming the Series
    ts_log = np.log(ts)
    test_stationarity(ts_log)
## Removing trend and seasonality with differencing
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    test_stationarity(ts_log_diff)

    size = int(len(ts_log)-100)
# Divide into train and test
    train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
    history = [x for x in train_arima]
    predictions = list()
    originals = list()
    error_list = list()
    for t in range(len(test_arima)):
        model = ARIMA(history, order=(2, 1, 0))
        model_fit = model.fit(disp=-1)
        
        output = model_fit.forecast()
        
        pred_value = output[0]
        
            
        original_value = test_arima[t]
        history.append(original_value)
        
        pred_value = np.exp(pred_value)
        
        
        original_value = np.exp(original_value)
        
        # Calculating the error
        error = ((abs(pred_value - original_value)) / original_value) * 100
        error_list.append(error)
        predictions.append(float(pred_value))
        originals.append(float(original_value))
        mean_error=float(sum(error_list)/float(len(error_list)))
# Mean absolute error
    
    lst=list(range(1,101))
    temp_df=pd.DataFrame(list(zip(lst,predictions,originals)),columns=['Days','Predicted Values','Actual Values'])
    fig = px.line(temp_df, x="Days", y=[temp_df['Actual Values'],temp_df['Predicted Values']],labels={'value':'Stock price','index': 'Days'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    
    st.write('**Mean absolute error of predicted values : %f**'%(mean_error))
    st.header('Difference betweent the Original and predicted Values')
    st.dataframe(temp_df)
    st.header('Expected Vs Predicted Views Forecasting')
    st.plotly_chart(fig)

    st.sidebar.markdown("# ARIMA Time Series Model")
    


#LSTM Model
def lstm_model():
    closedf=hist[['Date','close']]
    closedf=closedf[closedf['Date']>'2021-06-22']
    close_stock=closedf.copy()
    fig = px.line(closedf, x=closedf.Date, y=closedf.close,labels={'date':'Date','close':'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
    fig.update_layout(title_text='Considered period to predict Bitcoin close price', 
                    plot_bgcolor='white', font_size=15, font_color='black')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)
    #fig.show()
    del closedf['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf))
    train_size=int(len(closedf)*0.60)
    test_size=len(closedf)-train_size
    train_data,test_data=closedf[0:train_size,:],closedf[train_size:len(closedf),:1]
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    model=Sequential()

    model.add(LSTM(10,input_shape=(None,1),activation="relu"))

    model.add(Dense(1))

    model.compile(loss="mean_squared_error",optimizer="adam")
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)
#Plotting Loss vs Validation data
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    fig ,ax = plt.subplots()
    ax.plot(epochs, loss, 'r', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    
    plt.legend(loc=0)
    plt.figure()
    st.header('Training and validation loss')
    st.plotly_chart(fig)
    #plt.show()
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    print("Test predicted data: ", testPredictPlot.shape)

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'date': close_stock['Date'],
                        'original_close': close_stock['close'],
                        'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    plotdf.to_csv('plotdf.csv',index =None)

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                            plotdf['test_predicted_close']],
                labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(
                    plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    #errors value
    train_data_rmse=math.sqrt(mean_squared_error(original_ytrain,train_predict))
    train_data_mse = mean_squared_error(original_ytrain,train_predict)
    train_data_mae = mean_absolute_error(original_ytrain,train_predict)
    evs=explained_variance_score(original_ytrain, train_predict)
    test_evs=explained_variance_score(original_ytest, test_predict)




    st.header('Comparision between original close price vs predicted close price')
    #st.dataframe(plotdf)
    st.plotly_chart(fig)
    st.header('Evaluation metrices RMSE, MSE and MAE')
    st.write('Test data RMSE: %f' %train_data_rmse)
    st.write("Train data MSE: %f" %train_data_mse)
    st.write("Train data MAE: %f" %train_data_mae)
    st.write("Train data explained variance regression score: %f" %evs)
    st.write("Test data explained variance regression score: %f" %test_evs) 

    st.sidebar.markdown("# LSTM Model")




        





model_names_to_funcs = {
    
    "ARIMA Model": arima,
    "Linear Regression":linear_regression,
    'LSTM Model': lstm_model,
    "SVM Model": svm_model
    
    }

selected_model = st.sidebar.selectbox("Select Model", model_names_to_funcs.keys())
model_names_to_funcs[selected_model]()

