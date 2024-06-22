pip install streamlit-option-menu

import streamlit as st 
from streamlit_option_menu import option_menu 
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
import pickle
from sksurv.linear_model import CoxPHSurvivalAnalysis
from PIL import Image

with open('mace_non_ascvd_model.pkl', 'rb') as f:
    non_hdl_mace, ldl_mace, bmi_mace = pickle.load(f)

image = Image.open('./images/mace_app_banner.png')
st.image(image, use_column_width ="always" )


# Define time point prediction
time_points = np.arange(0, 61)

# Prediction function 
def prediction_survival(input, model):
    pred_prob = model.predict_survival_function(input)
    for i, surv_func in enumerate(pred_prob):
        pred = surv_func(time_points)
        # Change to failure probability
        pred = [1 - x for x in pred]
        pred_5yr =  (1 - surv_func(60))*100
    return pred, pred_5yr

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('5-year Major Cardiovascular Event Prediction System for Thai people',
                          
                          ['Model using Non-HDL Cholestoral Level',
                           'Model using LDL Cholestoral Level',
                           'Model using Body Mass Index'],
                          icons=['activity','heart', 'balloon'],
                          default_index=0)
    
    st.markdown(
        """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
# Model using non-HDL cholestoral level
if (selected == 'Model using Non-HDL Cholestoral Level'):
    
    st.header("Model using Non-HDL Cholestoral Level")
    
    
    def transform_input(AGE,SEX,SMOKE,AF,TCOL,HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        NON_HDL = TCOL - HDL
        
        if SEX == 'ชาย':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่':
            AF = 1
        else:
            AF = 0
            

        return [AGE,SEX,SMOKE,AF,NON_HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',20, 120)
        SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
        SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
        AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])
        st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
        EGFR = st.slider('Estimated glomerular filtration rate (%)',0, 100)
        TCOL = st.slider('Total cholesterol (mg/dL)',0, 1000)
        HDL = st.slider('HDL cholesterol (mg/dL)',0, 1000)
        st.write("ประวัติการใช้ยา")
        ANY_HTN_MED = st.selectbox('รับประทานยาลดความดันโลหิต', ['ไม่ใช่', 'ใช่'])
        ANY_ORAL_DM = st.selectbox('รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        
        if st.form_submit_button('Predict'):
            
            input_tranfrom = transform_input(AGE,SEX,SMOKE,AF,TCOL,HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)

            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","Non_HDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, non_hdl_mace)
            mace_5yr_prop = format(pred_5yr,".2f")
            text = "5-year MACE probability: " + str(mace_5yr_prop) + " %"
            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name='3P-MACE probability (%)',
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title='3P-MACE probability (%)',
                xaxis_title="Months",
                title='5-year Major Cardiovascular Event Prediction',
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = ['Baseline','1 year', '2 year', '3 year', '4 year', '5 year']),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")


# Model using LDL cholestoral level
if (selected == 'Model using LDL Cholestoral Level'):
    
    st.header("Model using LDL Cholestoral Level")
    
    
    def transform_input_ldl(AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        LDL = LDL
        
        if SEX == 'ชาย':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่':
            AF = 1
        else:
            AF = 0
            

        return [AGE,SEX,SMOKE,AF,LDL ,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',20, 120)
        SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
        SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
        AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])
        st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
        EGFR = st.slider('Estimated glomerular filtration rate (%)',0, 100)
        LDL = st.slider('LDL cholesterol (mg/dL)',0, 1000)
        st.write("ประวัติการใช้ยา")
        ANY_HTN_MED = st.selectbox('รับประทานยาลดความดันโลหิต', ['ไม่ใช่', 'ใช่'])
        ANY_ORAL_DM = st.selectbox('รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        
        if st.form_submit_button('Predict'):
            
            input_tranfrom = transform_input_ldl(AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)

            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","LDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, ldl_mace)
            mace_5yr_prop = format(pred_5yr,".2f")
            text = "5-year MACE probability: " + str(mace_5yr_prop) + " %"
            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name='3P-MACE probability (%)',
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title='3P-MACE probability (%)',
                xaxis_title="Months",
                title='5-year Major Cardiovascular Event Prediction',
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = ['Baseline','1 year', '2 year', '3 year', '4 year', '5 year']),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")

# Model using BMI
if (selected == 'Model using Body Mass Index'):
    
    st.header("Model using Body Mass Index")
    
    
    def transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        BMI = BMI

        if SEX == 'ชาย':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่':
            AF = 1
        else:
            AF = 0
            

        return [AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',20, 120)
        SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
        SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
        AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])
        BMI =  st.slider('ดัชนีมวลกาย (kg/m2)',0, 80)
        st.write("ประวัติการใช้ยา")
        ANY_HTN_MED = st.selectbox('รับประทานยาลดความดันโลหิต', ['ไม่ใช่', 'ใช่'])
        ANY_ORAL_DM = st.selectbox('รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        
        if st.form_submit_button('Predict'):
            
            input_tranfrom = transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)

            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","BMI","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, bmi_mace)
            mace_5yr_prop = format(pred_5yr,".2f")
            text = "5-year MACE probability: " + str(mace_5yr_prop) + " %"
            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name='3P-MACE probability (%)',
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title='3P-MACE probability (%)',
                xaxis_title="Months",
                title='5-year Major Cardiovascular Event Prediction',
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = ['Baseline','1 year', '2 year', '3 year', '4 year', '5 year']),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")

css="""
<style>
    [data-testid="stForm"] {
        background: #;
    }
</style>
"""

st.write(css, unsafe_allow_html=True)