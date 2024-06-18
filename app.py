import streamlit as st 
from streamlit_option_menu import option_menu 
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
import pickle
from sksurv.linear_model import CoxPHSurvivalAnalysis

with open('mace_non_ascvd_model.pkl', 'rb') as f:
    non_hdl_mace, ldl_mace, bmi_mace = pickle.load(f)

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
                          
                          ['Model using non-HDL cholestoral level',
                           'Model using LDL cholestoral level',
                           'Model using body mass index'],
                          icons=['activity','heart', "exercise"],
                          default_index=0)
    
# Non-established ASCVD patients Page
if (selected == 'Model using non-HDL cholestoral level'):
    
    st.title("5-year Major Cardiovascular Event Prediction Model using non-HDL cholestoral level")
    
    
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
        AGE =  st.slider('อายุ (ปี)',45, 120)
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

            

# Established ASCVD patients Page
if (selected == 'Established ASCVD patients'):
    
    st.title("5-year Major Cardiovascular Event Prediction Model for Thai people with established ASCVD")
    baseline_5yr = 0.888
    
    
    def transform_input(AGE,BMI,SBP,AF,HF50,FHX,EGFR,TCOL,HDL,DIURETICS,INSULIN,ASA,P2Y12):
        NON_HDL = TCOL - HDL
        if AF == 'ใช่':
            AF = 1
        else:
            AF = 0

        if HF50 == 'ใช่':
            HF50 = 1
        else:
            HF50 = 0

        if FHX == 'ใช่':
            FHX = 1
        else:
            FHX = 0
            
        if DIURETICS == 'ใช่':
            DIURETICS = 1
        else:
            DIURETICS = 0

        
        if INSULIN == 'ใช่':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if ASA == 'ใช่':
            ASA = 1
        else:
            ASA = 0
            
        if P2Y12 == 'ใช่':
            P2Y12 = 1
        else:
            P2Y12 = 0

        if ASA == 'ใช่' and P2Y12 == 'ใช่':
            DUAL_ANTIPLATELETS = 1
        else:
            DUAL_ANTIPLATELETS = 0            
          

        return AGE,BMI,SBP, AF,HF50,FHX,EGFR,TCOL,HDL,NON_HDL,DIURETICS,INSULIN,DUAL_ANTIPLATELETS
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',45, 120)
        BMI =  st.slider('ดัชนีมวลกาย (kg/m2)',0, 60)
        SBP = st.slider('ความดันโลหิตขณะหัวใจบีบตัว (mmHg)',0, 300)
        AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])
        HF50 = st.selectbox('ได้รับการวินิจฉัยภาวะหัวใจล้มเหลวชนิดที่การบีบตัวของหัวใจเป็นผิดปกติ EF<50%', ['ไม่ใช่', 'ใช่'])
        FHX = st.selectbox('มีประวัติญาติสายตรงป่วยเป็นโรคหลอดเลือดสมองหรือหัวใจ', ['ไม่ใช่', 'ใช่'])    
        st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
        EGFR = st.slider('Estimated glomerular filtration rate (%)',0, 100)
        TCOL = st.slider('Total cholesterol (mg/dL)',0, 1000)
        HDL = st.slider('HDL cholesterol (mg/dL)',0, 1000)
        st.write("ประวัติการใช้ยา")
        DIURETICS = st.selectbox('รับประทานยาขับปัสสาวะ', ['ไม่ใช่', 'ใช่'])
        INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        ASA = st.selectbox('รับประทานยาต้านการทำงานของเกล็ดเลือดในกลุ่ม Aspirin', ['ไม่ใช่', 'ใช่'])
        P2Y12 = st.selectbox('รับประทานยาต้านการทำงานของเกล็ดเลือดในกลุ่ม P2Y12', ['ไม่ใช่', 'ใช่'])

        
        if st.form_submit_button('Predict'):
            
            AGE,BMI,SBP, AF,HF50,FHX,EGFR,TCOL,HDL,NON_HDL,DIURETICS,INSULIN,DUAL_ANTIPLATELETS = transform_input(AGE,BMI,SBP,AF,HF50,FHX,EGFR,TCOL,HDL,DIURETICS,INSULIN,ASA,P2Y12)
            pi_5 = (0.021*AGE) - (0.048*BMI) + (0.004*SBP) + (0.237*FHX) - (0.158*EGFR) + (0.002*NON_HDL) + (0.794*AF) + (0.404*HF50) + (0.419*INSULIN) + (0.171*DUAL_ANTIPLATELETS) + (0.445*DIURETICS)
            mace_5yr = 1 - baseline_5yr**np.exp(pi_5)
            mace_5yr_prop = format(mace_5yr*100,".2f")
            text = "5-year MACE probability: " + str(mace_5yr_prop) + " %"
            st.success(text, icon="💔")
