import streamlit as st 
from streamlit_option_menu import option_menu 
import pandas as pd 
import numpy as np 


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('5-year Major Cardiovascular Event Prediction Model for Thai people Prediction System',
                          
                          ['Non-established ASCVD patients',
                           'Established ASCVD patients'],
                          icons=['activity','heart'],
                          default_index=0)
    
# Non-established ASCVD patients Page
if (selected == 'Non-established ASCVD patients'):
    
    st.title("5-year Major Cardiovascular Event Prediction Model for Thai people without established ASCVD")
    baseline_5yr = 0.993
    
    
    def transform_input(AGE,SEX,SMOKE,EGFR,TCOL,HDL,ANY_HTN_MED,ANY_ORAL_DM,INSULIN,ASA,P2Y12,OAC):
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
    
        if ASA == 'ใช่':
            ASA = 1
        else:
            ASA = 0
            
        if P2Y12 == 'ใช่':
            P2Y12 = 1
        else:
            P2Y12 = 0
    
        if OAC == 'ใช่':
            OAC = 1
        else:
            OAC = 0
        return AGE,SEX,SMOKE,EGFR,TCOL,HDL,NON_HDL,ANY_HTN_MED,ANY_ORAL_DM,INSULIN,ASA,P2Y12,OAC
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',45, 120)
        SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
        SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
        st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
        EGFR = st.slider('Estimated glomerular filtration rate (%)',0, 100)
        TCOL = st.slider('Total cholesterol (mg/dL)',0, 1000)
        HDL = st.slider('HDL cholesterol (mg/dL)',0, 1000)
        st.write("ประวัติการใช้ยา")
        ANY_HTN_MED = st.selectbox('รับประทานยาลดความดันโลหิต', ['ไม่ใช่', 'ใช่'])
        ANY_ORAL_DM = st.selectbox('รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
        ASA = st.selectbox('รับประทานยาต้านการทำงานของเกล็ดเลือดในกลุ่ม Aspirin', ['ไม่ใช่', 'ใช่'])
        P2Y12 = st.selectbox('รับประทานยาต้านการทำงานของเกล็ดเลือดในกลุ่ม P2Y12', ['ไม่ใช่', 'ใช่'])
        OAC = st.selectbox('รับประทานยาละลายลิ่มเลือด (Anticogulants) เช่น Warfarin', ['ไม่ใช่', 'ใช่'])
        
        if st.form_submit_button('Predict'):
            
            AGE,SEX,SMOKE,EGFR,TCOL,HDL,NON_HDL,ANY_HTN_MED,ANY_ORAL_DM,INSULIN,ASA,P2Y12,OAC = transform_input(AGE,SEX,SMOKE,EGFR,TCOL,HDL,ANY_HTN_MED,ANY_ORAL_DM,INSULIN,ASA,P2Y12,OAC)
            pi_5 = (0.040*AGE) - (0.047*SEX) + (0.003*NON_HDL) + (0.323*SMOKE) - (0.014*SMOKE*5) - (0.022*EGFR) + (0.0002*EGFR*5) + (1.257*ANY_HTN_MED) - (0.376*ANY_ORAL_DM) + (0.351*INSULIN) + (0.558*ASA) + (0.541*P2Y12) + (1.405*OAC)
            mace_5yr = 1 - baseline_5yr**np.exp(pi_5)
            mace_5yr_prop = format(mace_5yr*100,".2f")
            text = "5-year MACE probability: " + str(mace_5yr_prop) + " %"
            st.success(text, icon="💔")

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
