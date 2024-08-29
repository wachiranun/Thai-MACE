import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
import pickle
from sksurv.linear_model import CoxPHSurvivalAnalysis
from PIL import Image

##Header##
with open(r"C:\Users\niraw\OneDrive\Desktop\Thai-MACE\mace_non_ascvd_model.pkl", 'rb') as f:
    non_hdl_mace, ldl_mace, bmi_mace = pickle.load(f)
image = Image.open(r"C:\Users\niraw\OneDrive\Desktop\Thai-MACE\images\mace_app_banner.png")
logo = r"C:\Users\niraw\OneDrive\Desktop\Thai-MACE\images\slidebar_button.png"
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


# Initialize session state for language and metric selection
if "language" not in st.session_state:
    st.session_state.language = "th"

if "metric_system" not in st.session_state:
    st.session_state.metric_system = "standard"  # Default to standard metric

# Function to set the language in session state
def set_language(lang):
    st.session_state.language = lang

# Function to set the metric system in session state
def set_metric_system(metric):
    st.session_state.metric_system = metric

# Sidebar with language selection buttons
with st.sidebar:
    st.header("Select Language:")
    if st.button("EN"):
        set_language('en')
    if st.button("TH"):
        set_language('th')
    
    st.header("Select Metric System:")
    if st.button("Standard Metric"):
        set_metric_system('standard')
    if st.button("American Metric"):
        set_metric_system('american')

    if st.session_state.language == "th":
        selected = option_menu('ระบบทำนายความเสี่ยงของเหตุการณ์หัวใจและหลอดเลือดใน 5 ปีสำหรับคนไทย',
                ['แบบจำลองที่ใช้ระดับคอเลสเตอรอล Non-HDL',
                'แบบจำลองที่ใช้ระดับคอเลสเตอรอล LDL',
                'แบบจำลองที่ใช้ดัชนีมวลกาย'],
                icons=['activity','heart', 'balloon'],
                default_index=0)
    if st.session_state.language == "en":
        selected = option_menu('5-year Major Cardiovascular Event Prediction System for Thai people',
                ['Model using Non-HDL Cholestoral Level',
                'Model using LDL Cholestoral Level',
                'Model using Body Mass Index'],
                icons=['activity','heart', 'balloon'],
                default_index=0)

def thai_page():
    if selected == "แบบจำลองที่ใช้ระดับคอเลสเตอรอล Non-HDL":
        
        st.header("แบบจำลองที่ใช้ระดับคอเลสเตอรอล Non-HDL")
        
        def transform_input(AGE, SEX, SMOKE, AF, TCOL, HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN):
            NON_HDL = TCOL - HDL
            
            SEX = 1 if SEX == 'ชาย' else 0
            SMOKE = 1 if SMOKE == 'สูบ' else 0
            AF = 1 if AF == 'ใช่' else 0
            ANY_HTN_MED = 1 if ANY_HTN_MED == 'ใช่' else 0
            ANY_ORAL_DM = 1 if ANY_ORAL_DM == 'ใช่' else 0
            INSULIN = 1 if INSULIN == 'ใช่' else 0

            # Convert values to American metric if selected
            if st.session_state.metric_system == "american":
                TCOL = TCOL * 0.0259  # Conversion factor from mg/dL to mmol/L
                HDL = HDL * 0.0259
                EGFR = EGFR / 1.73  # Adjust for body surface area
                NON_HDL = NON_HDL * 0.0259
            
            return [AGE, SEX, SMOKE, AF, NON_HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN]
        
        with st.form('my_form'):
            st.write("ข้อมูลพื้นฐาน")
            AGE = st.slider('อายุ (ปี)', 20, 120)
            SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
            SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
            AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])

            st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
            if st.session_state.metric_system == "standard":
                TCOL = st.slider('คอเลสเตอรอลรวม (mg/dL)', 0, 1000, step=1)
                HDL = st.slider('คอเลสเตอรอล HDL (mg/dL)', 0, 1000, step=1)
                EGFR = st.slider('อัตราเลือดที่ผ่านตัวกรองของไต (%)', 0, 100, step=1)
            elif st.session_state.metric_system == "american":
                TCOL = st.slider('คอเลสเตอรอลรวม (mmol/L)', 0.0, 25.9, step=0.1)
                HDL = st.slider('คอเลสเตอรอล HDL (mmol/L)', 0.0, 25.9, step=0.1)
                EGFR = st.slider('อัตราเลือดที่ผ่านตัวกรองของไต (%) (mL/min/1.73 m²)', 0, 173, step=1)

            st.write("ประวัติการใช้ยา")
            ANY_HTN_MED = st.selectbox('รับประทานยาลดความดันโลหิต', ['ไม่ใช่', 'ใช่'])
            ANY_ORAL_DM = st.selectbox('รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
            INSULIN = st.selectbox('ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด', ['ไม่ใช่', 'ใช่'])
            
            if st.form_submit_button('Predict'):
                
                input_transformed = transform_input(AGE, SEX, SMOKE, AF, TCOL, HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN)

                new_prediction = pd.DataFrame.from_dict({1: input_transformed}, columns=["AGE", "SEX", "SMOKE", "AF", "Non_HDL", "EGFR", "ANY_HTN_MED", "ANY_ORAL_DM", "INSULIN"], orient='index')

                pred, pred_5yr = prediction_survival(new_prediction, non_hdl_mace)
                mace_5yr_prop = format(pred_5yr, ".2f")
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
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 12, 24, 36, 48, 60],
                        ticktext=['Baseline', '1 year', '2 year', '3 year', '4 year', '5 year']),
                    font=dict(
                        size=15,
                        color="RebeccaPurple"),
                )

                event = st.plotly_chart(fig, on_select="rerun")

    if selected == "แบบจำลองที่ใช้ระดับคอเลสเตอรอล LDL":
        st.header("แบบจำลองที่ใช้ระดับคอเลสเตอรอล LDL")

        def transform_input_ldl(AGE, SEX, SMOKE, AF, LDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN):
            LDL = LDL
            SEX = 1 if SEX == 'ชาย' else 0
            SMOKE = 1 if SMOKE == 'สูบ' else 0
            ANY_HTN_MED = 1 if ANY_HTN_MED == 'ใช่' else 0
            ANY_ORAL_DM = 1 if ANY_ORAL_DM == 'ใช่' else 0
            INSULIN = 1 if INSULIN == 'ใช่' else 0
            AF = 1 if AF == 'ใช่' else 0

            # Apply conversions based on metric system
            if st.session_state.metric_system == "american":
                LDL = LDL * 0.0259  # Convert mg/dL to mmol/L
                EGFR = EGFR / 1.73  # Adjust for body surface area

            return [AGE, SEX, SMOKE, AF, LDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN]

        with st.form('my_form'):
            st.write("ข้อมูลพื้นฐาน")
            AGE = st.slider('อายุ (ปี)', 20, 120)
            SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])
            SMOKE = st.selectbox('สูบบุหรี่', ['ไม่สูบ', 'สูบ'])
            AF = st.selectbox('ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', ['ไม่ใช่', 'ใช่'])

            st.write("ผลการตรวจทางห้องปฏิบัติการณ์")
            if st.session_state.metric_system == "standard":
                LDL = st.slider('คอเลสเตอรอล LDL (mg/dL)', 0, 1000, step=1)
                EGFR = st.slider('อัตราเลือดที่ผ่านตัวกรองของไต (%)', 0, 100, step=1)
            elif st.session_state.metric_system == "american":
                LDL = st.slider('คอเลสเตอรอล HDL (mmol/L)', 0.0, 25.9, step=0.1)
                EGFR = st.slider('อัตราเลือดที่ผ่านตัวกรองของไต (%) (mL/min/1.73 m²)', 0, 173, step=1)

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

    if (selected == "แบบจำลองที่ใช้ดัชนีมวลกาย"):
        
        st.header("แบบจำลองที่ใช้ดัชนีมวลกาย")
        
        
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
            BMI =  st.slider('ดัชนีมวลกาย (kg/m2)',0.0, 80.0, step = 0.1)
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

def english_page():
    if selected == 'Model using Non-HDL Cholestoral Level':
        
        st.header('Model using Non-HDL Cholestoral Level')
        
        def transform_input(AGE, SEX, SMOKE, AF, TCOL, HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN):
            NON_HDL = TCOL - HDL
            
            SEX = 1 if SEX == 'MALE' else 0
            SMOKE = 1 if SMOKE == 'SMOKER' else 0
            AF = 1 if AF == 'YES' else 0
            ANY_HTN_MED = 1 if ANY_HTN_MED == 'YES' else 0
            ANY_ORAL_DM = 1 if ANY_ORAL_DM == 'YES' else 0
            INSULIN = 1 if INSULIN == 'YES' else 0

            # Convert values to American metric if selected
            if st.session_state.metric_system == "american":
                TCOL = TCOL * 0.0259  # Conversion factor from mg/dL to mmol/L
                HDL = HDL * 0.0259
                EGFR = EGFR / 1.73  # Adjust for body surface area
                NON_HDL = NON_HDL * 0.0259
            
            return [AGE, SEX, SMOKE, AF, NON_HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN]
        
        with st.form('my_form'):
            st.write("INFORMATION")
            AGE = st.slider('AGE (YEAR)', 20, 120)
            SEX = st.selectbox('GENDER', ['MALE', 'FEMALE'])    
            SMOKE = st.selectbox('SMOKING', ['NON-SMOKER', 'SMOKER'])
            AF = st.selectbox('HAVING ATRIAL FIBRILATION', ['NO', 'YES'])

            st.write("RESULTS FROM LAB")
            if st.session_state.metric_system == "standard":
                TCOL = st.slider('TOTAL CHORESTEROL (mg/dL)', 0, 1000, step=1)
                HDL = st.slider('HDL CHEORESTEROL LEVEL (mg/dL)', 0, 1000, step=1)
                EGFR = st.slider('ESTIMATED GLOMERULAR FILTRATION RATE (%)', 0, 100, step=1)
            elif st.session_state.metric_system == "american":
                TCOL = st.slider('TOTAL CHORESTEROL (mmol/L)', 0.0, 25.9, step=0.1)
                HDL = st.slider('HDL CHEORESTEROL LEVEL (mmol/L)', 0.0, 25.9, step=0.1)
                EGFR = st.slider('ESTIMATED GLOMERULAR FILTRATION RATE (%) (mL/min/1.73 m²)', 0, 173, step=1)

            st.write("MEDICAL USAGES HISTORY")
            ANY_HTN_MED = st.selectbox('TAKING MEDICINE FOR REDUCING BP', ['NO', 'YES'])
            ANY_ORAL_DM = st.selectbox('TAKING MEDICINE FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])
            INSULIN = st.selectbox('TAKING INSULIN FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])
            
            if st.form_submit_button('Predict'):
                
                input_transformed = transform_input(AGE, SEX, SMOKE, AF, TCOL, HDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN)

                new_prediction = pd.DataFrame.from_dict({1: input_transformed}, columns=["AGE", "SEX", "SMOKE", "AF", "Non_HDL", "EGFR", "ANY_HTN_MED", "ANY_ORAL_DM", "INSULIN"], orient='index')

                pred, pred_5yr = prediction_survival(new_prediction, non_hdl_mace)
                mace_5yr_prop = format(pred_5yr, ".2f")
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
                    xaxis=dict(
                        tickmode='array',
                        tickvals=[0, 12, 24, 36, 48, 60],
                        ticktext=['Baseline', '1 year', '2 year', '3 year', '4 year', '5 year']),
                    font=dict(
                        size=15,
                        color="RebeccaPurple"),
                )

                event = st.plotly_chart(fig, on_select="rerun")

    if selected == 'Model using LDL Cholestoral Level':
        st.header('Model using LDL Cholestoral Level')

        def transform_input_ldl(AGE, SEX, SMOKE, AF, LDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN):
            LDL = LDL
            SEX = 1 if SEX == 'MALE' else 0
            SMOKE = 1 if SMOKE == 'SMOKER' else 0
            ANY_HTN_MED = 1 if ANY_HTN_MED == 'YES' else 0
            ANY_ORAL_DM = 1 if ANY_ORAL_DM == 'YES' else 0
            INSULIN = 1 if INSULIN == 'YES' else 0
            AF = 1 if AF == 'YES' else 0

            # Apply conversions based on metric system
            if st.session_state.metric_system == "american":
                LDL = LDL * 0.0259  # Convert mg/dL to mmol/L
                EGFR = EGFR / 1.73  # Adjust for body surface area

            return [AGE, SEX, SMOKE, AF, LDL, EGFR, ANY_HTN_MED, ANY_ORAL_DM, INSULIN]

        with st.form('my_form'):
            st.write("INFORMATION")
            AGE = st.slider('AGE (YEAR)', 20, 120)
            SEX = st.selectbox('GENDER', ['MALE', 'FEMALE'])
            SMOKE = st.selectbox('SMOKING', ['NON-SMOKER', 'SMOKER'])
            AF = st.selectbox('HAVING ATRIAL FIBRILATION', ['NO', 'YES'])

            st.write("RESULTS FROM LAB")
            if st.session_state.metric_system == "standard":
                LDL = st.slider('LDL CHEORESTEROL LEVEL (mg/dL)', 0, 1000, step=1)
                EGFR = st.slider('ESTIMATED GLOMERULAR FILTRATION RATE (%)', 0, 100, step=1)
            elif st.session_state.metric_system == "american":
                LDL = st.slider('LDL CHEORESTEROL LEVEL (mmol/L)', 0.0, 25.9, step=0.1)
                EGFR = st.slider('ESTIMATED GLOMERULAR FILTRATION RATE (%) (mL/min/1.73 m²)', 0, 173, step=1)

            st.write("MEDICAL USAGES HISTORY")
            ANY_HTN_MED = st.selectbox('TAKING MEDICINE FOR REDUCING BP', ['NO', 'YES'])
            ANY_ORAL_DM = st.selectbox('TAKING MEDICINE FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])
            INSULIN = st.selectbox('TAKING INSULIN FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])

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

    if (selected == 'Model using Body Mass Index'):
        
        st.header('Model using Body Mass Index')
        
        
        def transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
            BMI = BMI
            SEX = 1 if SEX == 'MALE' else 0
            SMOKE = 1 if SMOKE == 'SMOKER' else 0
            ANY_HTN_MED = 1 if ANY_HTN_MED == 'YES' else 0
            ANY_ORAL_DM = 1 if ANY_ORAL_DM == 'YES' else 0
            INSULIN = 1 if INSULIN == 'YES' else 0
            AF = 1 if AF == 'YES' else 0
                

            return [AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]
        
            
        with st.form('my_form'):
            st.write("INFORMATION")
            AGE = st.slider('AGE (YEAR)', 20, 120)
            SEX = st.selectbox('GENDER', ['MALE', 'FEMALE'])
            SMOKE = st.selectbox('SMOKING', ['NON-SMOKER', 'SMOKER'])
            AF = st.selectbox('HAVING ATRIAL FIBRILATION', ['NO', 'YES'])
            BMI =  st.slider('BMI INDEX (kg/m2)',0.0, 80.0, step = 0.1)
            st.write("MEDICAL USAGES HISTORY")
            ANY_HTN_MED = st.selectbox('TAKING MEDICINE FOR REDUCING BP', ['NO', 'YES'])
            ANY_ORAL_DM = st.selectbox('TAKING MEDICINE FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])
            INSULIN = st.selectbox('TAKING INSULIN FOR CONTROLING BLOOD SUGAR', ['NO', 'YES'])
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


# Main app logic
if st.session_state.language == "th":
    if st.session_state.metric_system == "standard":
        thai_page()
    elif st.session_state.metric_system == "american":
        thai_page()
elif st.session_state.language == "en":
    if st.session_state.metric_system == "standard":
        english_page()
    elif st.session_state.metric_system == "american":
        english_page()


    