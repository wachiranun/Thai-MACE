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

logo = './images/slidebar_button.png'

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


# The station labels in language 'th' and 'en'
stations = {"th": ['เลือกโมเดลที่ใช้ในการทำนายการเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือด',
                   ['โมเดลทำนายการเกิดโรคโดยใช้ค่าไขมันคอเลสเตอรอลชนิดดี (non-HDL Cholestoral Level)','โมเดลทำนายการเกิดโรคโดยใช้ค่าไขมันคอเลสเตอรอลชนิดไม่ดี (LDL Cholestoral Level)',
                    'โมเดลทำนายการเกิดโรคโดยใช้ดัชนีมวลกาย (Body Mass Index)'],
                   "เลือกหน่วยวัดผลการตรวจทางห้องปฏิบัติการ",
                   ("หน่วยวัด US", "หน่วยวัด SI"),
                   "เลือกค่าการทำงานของไต ค่าการทำงานของไต (eGFR) หรือ ค่าครีตินีน (Creatinine) เพื่อใช้ในการทำนายผล",
                   ("ค่าการทำงานของไต (eGFR)", "ครีตินีน (Creatinine)"),
                   "เลือกใช้ค่าดัชนีมวลกาย (BMI) หรือ ค่าน้ำหนักและส่วนสูง เพื่อใช้ในการทำนายผล",
                   ("ดัชนีมวลกาย (BMI)", "ค่าน้ำหนักและส่วนสูง"),
                   "ข้อมูลพื้นฐาน",
                   'อายุ (ปี)', 
                   'เพศ',
                   ['ชาย','หญิง'],
                   'ประวัติการสูบบุหรี่',
                   ['ไม่สูบ', 'เลิกสูบแล้ว', 'สูบ'],
                   'ได้รับการวินิจฉัยโรคหัวใจห้องบนเต้นผิดจังหวะ (Atrial Fibrilation)', 
                   ['ไม่ใช่', 'ใช่'],
                   'ดัชนีมวลกาย BMI, (kg/m2)',
                   'น้ำหนัก (kg)',
                   'ส่วนสูง (cm)',
                   "ค่าดัชนีมวลกาย (BMI, kg/m2) ที่ใช้ในการทำนาย:",
                   'ผลการตรวจทางห้องปฏิบัติการ',
                   'ค่าการทำงานของไต Estimated Glomerular Filtration Rate, (mL/min/1.73m2)',
                   'ค่าครีตินีน Creatinine',
                    'ไขมันคอเลสเตอรอล Total cholesterol',
                    'ไขมันคอเลสเตอรอลชนิดดี HDL cholesterol',
                    'ไขมันคอเลสเตอรอลชนิดเลว LDL cholesterol',
                    'ค่าไขมันคอเลสเตอรอลชนิดเลว (LDL, mg/dl) ที่ใช้ในการทำนาย:',
                    'ค่าไขมันคอเลสเตอรอลที่ไม่ใช่ชนิดดี (non-HDL, mg/dl) ที่ใช้ในการทำนาย:',
                    'ค่าการทำงานของไต (eGFR, mL/min/1.73m2) ที่ใช้ในการทำนาย:',
                    'ประวัติการใช้ยา',
                    'รับประทานยาลดความดันโลหิต',
                    'รับประทานยาเพื่อควบคุมระดับน้ำตาลในเลือด',
                    'ใช้ยาฉีดอินซูลินเพื่อควบคุมระดับน้ำตาลในเลือด',
                    'ทำนายโอกาสเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปี',
                    'โอกาสเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปี: ',
                    'โอกาสเกิดเหตุการณ์ใน 5 ปี (%)',
                    'เดือน',
                    'ผลการทำนายโอกาสเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปี',
                    ['ปัจจุบัน','1 ปี', '2 ปี', '3 ปี', '4 ปี', '5 ปี'],
                    """
                    **การแบ่งกลุ่มผู้ป่วยตามความเสี่ยงในการเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปี \n
                    :green[กลุ่มเสี่ยงต่ำ (น้อยกว่า 2.8%)] \n
                    กลุ่มเสี่ยงน้อย (2.8% ถึง 5.3%) \n
                    :orange[กลุ่มเสี่ยงปานกลาง (5.4% ถึง 9.9%)] \n
                    :red[กลุ่มเสี่ยงสูง (มากกว่าเท่ากับ 10%)] \n
                    แนะนำให้ใช้โมเดลเพื่อทำนายการเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปี \n
                    _สำหรับกลุ่มผู้ป่วยอายุ 45-98 ปี ที่ยังไม่มีประวัติป่วยด้วยโรคหลอดเลือดหัวใจและสมอง_ \n
                    _และมีปัจจัยเสี่ยงของการเกิดโรคหลอดเลือดหัวใจและสมองอย่างน้อย 3 ข้อ_ ประกอบได้ด้วยปัจจัยเสี่ยงดังนี้ \n
                    1. เพศชายอายุ >55 ปี หรือเพศหญิงอายุ >65 ปี\n 
                    2. เป็นโรคเบาหวาน หรือมีภาวะระดับน้ำตาลในเลือดขณะอดอาหารผิดปกติ\n
                    3. เป็นโรคความดันโลหิตสูง \n
                    4. เป็นโรคไขมันในเลือดสูง \n
                    5. โรคไตเรื้อรังที่ตรวจพบโปรตีนรั่วในปัสสาวะ (proteinuria +1) หรือมีค่าการทำงานของไต (eGFR) ต่ำกว่า 60 mL/min/1.73m2\n
                    6. มีประวัติคนในครอบครัวเป็นโรคหลอดเลือดหัวใจและสมองก่อนวัยอันควร \n
                    """,
                    "กลุ่มเสี่ยงต่ำ (น้อยกว่า 2.8%)",
                    "กลุ่มเสี่ยงน้อย (2.8% ถึง 5.3%)",
                    "กลุ่มเสี่ยงปานกลาง (5.4% ถึง 9.9%)",
                    "กลุ่มเสี่ยงสูง (มากกว่าเท่ากับ 10%)",
                    "แอปพลิเคชั่นนี้เหมาะสำหรับผู้ป่วยอายุ 45-98 ปี ที่ยังไม่มีประวัติป่วยด้วยโรคหลอดเลือดหัวใจและสมอง และมีปัจจัยเสี่ยงของการเกิดโรคหลอดเลือดหัวใจและสมอง",
                    "เปรียบเทียบโอกาสเสี่ยงในการเกิดโรคด้วยข้อมูลในอดีต",
                    "การประเมินความเสี่ยงในการเกิดโรคด้วยข้อมูลในอดีต",
                    "**(ความเสี่ยงในปัจจุบัน)**",
                    "**(ความเสี่ยงในอดีต)**",
                    """
                    **หมายเหตุ**\n
                    1. ผลลัพธ์ที่ได้เป็นการประเมินความเสี่ยงต่อการเกิดเหตุการณ์ชนิดรุนแรงจากสาเหตุหัวใจและหลอดเลือดใน 5 ปีข้างหน้า \n
                    2. ผลการประเมินและคำแนะนำที่ได้รับจากโปรแกรมนี้ไม่สามารถใช้แทนการตัดสินใจของแพทย์ได้ \n
                    3. การตรวจรักษาเพิ่มเติมหรือการให้ยารักษาขึ้นอยู่กับดุลยพินิจของแพทย์และการปรึกษากันระหว่างแพทย์และตัวท่าน \n
                    4. ผลการประเมินนี้ห้ามนำไปใช้อ้างอิงในการค้า เช่น การทำประกันชีวิต \n
                    """], 
            "en":['Model Selection',       
                 ['Model using Non-HDL Cholestoral Level','Model using LDL Cholestoral Level','Model using Body Mass Index'],
                  'Unit of laboratory measurements',
                  ('US unit', 'SI unit'),
                  'Select kidney function tests: eGFR or creatinine for prediction',
                  ("eGFR", "Creatinine"),
                  "Select: Body Mass Index (BMI) or Body Weight/Height for prediction",
                  ("Body Mass Index (BMI)", "Body Weight/Height"),
                  'Demographics',
                  'Age (year)', 
                  'Sex',
                  ['Male', 'Female'],
                  'Smoker?',
                  ['Never', 'Former', 'Current'],
                  'History of Atrial Fibrilation?', 
                  ['No', 'Yes'],
                  'Body Mass Index (kg/m2)',
                  'Body weight (kg)',
                  'Height (cm)',
                  'Body Mass Index (kg/m2) using for prediction:',
                  'Laboratory Tests',
                  'Estimated Glomerular Filtration Rate (mL/min/1.73m2)',
                  'Creatinine',
                  'Total Cholesterol',
                  'HDL Cholesterol',
                  'LDL Cholesterol',
                  'LDL Cholesterol (mg/dl) using for prediction:',
                  'Non-HDL Cholesterol (mg/dl) using for prediction:',
                  'eGFR (mL/min/1.73m2) using for prediction:',
                  'Medication History',
                  'On Hypertension Treatment?',
                  'On Oral Diabetes Treatment?',
                  'On Insulin Treatment?',
                  'Predict 5-year Major Cardiovascular Events',
                  "5-year MACE probability: ",
                  '3P-MACE probability (%)',
                  'months',
                  "Predicted 5-year Major Cardiovascular Events",
                  ['Baseline','1 year', '2 year', '3 year', '4 year', '5 year'],
                  """
                  **5-year MACE probability is categorized as: \n
                    :green[Low-risk (< 2.8%)] \n
                    Borderline risk (2.8% to 5.3%) \n
                    :orange[Intermediate risk (5.4% to 9.9%)] \n
                    :red[High risk (≥ 10%)] \n
                    Recommended to use the models to predict 5-year MACE probability \n
                    _for patients with non-estrablised ASCVD age 45–98 with multiple risk factors (MRFs) for ASCVD_\n
                    MRFs were characterized by the presence of a minimum of 3 risk factors for atherosclerosis \n
                    1. Males >55 years or Females >65 years\n 
                    2. Diabetes mellitus (DM) or Impaired fasting glucose\n
                    3. Hypertension \n
                    4. Dyslipidemia \n
                    5. Chronic kidney disease (proteinuria+1 or estimated glomerular filtration rate less than 60 mL/min/1.73m2)\n
                    6. Family history of premature atherosclerotic cardiovascular diseases\n
                    """,
                    "Low-risk (< 2.8%)",
                    "Borderline risk (2.8% to 5.3%)",
                    "Intermediate risk (5.4% to 9.9%)",
                    "High risk (≥ 10%)",
                    "App should be used for patients (those without ASCVD) with multiple risk factors (MRFs) for ASCVD",
                    "Comparing risk estimation using data from a previous visit",
                    "Values at Previous Visit",
                    "**(Current risk)**",
                    "**(Previous risk)**",
                    """
                    **Disclaimer** \n
                    1. The results provided are a prediction of the risk of major cardiovascular events occurring within the next 5 years \n
                    2. The evaluation and recommendations provided by this app cannot replace a doctor's judgment. \n
                    3. Further examinations or treatments, including medication, depend on the decision of your physician and discussion between you and your physician. \n
                    4. The use of the app's assessment results is prohibited for commercial purposes, such as health insurance applications. \n
                    """]}

index_stations = ['label_model_select', 'choice_model_select',
                  'label_unit_op','choice_unit_op','label_kid_op','choice_kid_op',
                  'label_bmi_op', 'choice_bmi_op',
                  'label_demo','label_age','label_sex','choice_sex','label_smoker','choice_smoker','label_af','choice_yesno',
                  'label_bmi', 'label_bw', 'label_ht', 'label_bmi_pred',
                  'label_lab', 'label_egfr', 'label_cr', 'label_tc', 'label_hdl', 'label_ldl', 'label_ldl_pred','label_nonhdl_pred', 'label_egfr_pred',
                  'label_med', 'label_htn_med', 'label_dm_med', 'label_insulin_med',
                  'label_pred','pred_text','yaxis_title', "xaxis_title", "title", "xtick",
                  "text_caption","low_risk","borderline_risk","intermediate_risk","high_risk","precaution",
                  "prev_data_use", "prev_data_head", "current_risk", "previous_risk", "disclaimer"]

df_stations = pd.DataFrame(data=stations, index = index_stations)

# sidebar for navigation
st.logo(logo, icon_image=logo)
with st.sidebar:

    selected_lang = st.selectbox(
                    "เลือกภาษา (Lauguage Selection)",
                    ("ภาษาไทย", "English"))

    if selected_lang == 'ภาษาไทย':
        label = df_stations['th']
    elif selected_lang == 'English':
        label = df_stations['en']

    selected = option_menu(label.label_model_select,            
                    label.choice_model_select,
                    icons=['activity','heart', 'balloon'],
                    default_index=0)

# Disabled prev_data expander as a default
st.session_state.active_expander = True

# Pre-caution before using
st.warning(label.precaution, icon="⚠️")

# Model using non-HDL cholestoral level
if (selected == "Model using Non-HDL Cholestoral Level" or selected =='โมเดลทำนายการเกิดโรคโดยใช้ค่าไขมันคอเลสเตอรอลชนิดดี (non-HDL Cholestoral Level)'):
    
    st.header(selected)
    
    def transform_input(AGE,SEX,SMOKE,AF,TCOL,HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        NON_HDL = TCOL - HDL
        
        if SEX == 'ชาย' or SEX == 'Male':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ' or SMOKE == 'Current':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่' or ANY_HTN_MED == 'Yes':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่' or ANY_ORAL_DM == 'Yes':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่' or INSULIN == 'Yes':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่' or AF == 'Yes':
            AF = 1
        else:
            AF = 0
            
        return [AGE,SEX,SMOKE,AF,NON_HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]
    
    unit_option = st.selectbox(
                    label.label_unit_op,
                    label.choice_unit_op)
    kidney_option = st.selectbox(
                    label.label_kid_op,
                    label.choice_kid_op)
    
    with st.form('my_form'):
        st.write(label.label_demo)
        AGE =  st.slider(label.label_age, 30 , 100, 45)
        SEX = st.selectbox(label.label_sex, label.choice_sex)    
        SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
        AF = st.selectbox(label.label_af, label.choice_yesno)
        st.write(label.label_lab)
        if kidney_option == "ค่าการทำงานของไต (eGFR)"  or kidney_option == 'eGFR':
            EGFR = st.slider(label.label_egfr, 0.00, 200.00, 50.00)
                
        elif kidney_option == "ครีตินีน (Creatinine)" or kidney_option == "Creatinine":
            if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                unit = 'mg/dl'
                CR = st.slider(f'{label.label_cr}, ({unit})', 0.00, 15.00, 0.60)
            elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                unit = 'mmol/L'
                CR_si = st.slider(f'{label.label_cr}, ({unit})', 0.000, 0.832, 0.033)
                CR = CR_si/0.0555
            
            if (SEX == "หญิง" or SEX == 'Female') and CR <= 0.7:
                COEF = -0.329
                EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "หญิง" or SEX == 'Female') and CR > 0.7:
                COEF = -1.209
                EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "ชาย" or SEX == 'Male') and CR <= 0.9:
                COEF = -0.411
                EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "ชาย" or SEX == 'Male') and CR > 0.9:
                COEF = -1.209
                EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
    

        if unit_option == "หน่วยวัด US" or unit_option == "US unit":
            unit = 'mg/dl'
            TCOL = st.slider(f'{label.label_tc}, ({unit})',0, 600, 150)
            HDL = st.slider(f'{label.label_hdl}, ({unit})',0, 200, 50)
        elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
            unit = 'mmol/L'
            TCOL_si = st.slider(f'{label.label_tc}, ({unit})',0.000, 33.30, 8.000)
            HDL_si = st.slider(f'{label.label_hdl}, ({unit})',0.000, 11.10, 3.300)
            TCOL = TCOL_si/0.0555
            HDL = HDL_si/0.0555

        st.write(label.label_med)
        ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
        ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
        INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)
        
        if st.form_submit_button(label.label_pred):
            input_tranfrom = transform_input(AGE,SEX,SMOKE,AF,TCOL,HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
            st.write(label.label_egfr_pred,  format(EGFR,".2f"))
            st.write(label.label_nonhdl_pred, format(input_tranfrom[4],".2f"))
            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","Non_HDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, non_hdl_mace)
            mace_5yr_prop = format(pred_5yr,".2f")

            if pred_5yr < 2.80:
                label_risk = label.low_risk
            if pred_5yr >= 2.80 and pred_5yr < 5.40:
                label_risk = label.borderline_risk
            if pred_5yr >= 5.40 and pred_5yr < 10.00:
                label_risk = label.intermediate_risk
            if pred_5yr >= 10.00:
                label_risk = label.high_risk

            text = f"""
            {label.pred_text}{str(mace_5yr_prop)}% \n
            {label_risk}
            """

            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name=label.pred_text,
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title=label.yaxis_title,
                xaxis_title=label.xaxis_title,
                title=label.title,
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = label.xtick),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")

            st.error(label.disclaimer, icon="🚨")
            
            active_expander = False

            st.session_state.mace_5yr_prop = mace_5yr_prop
            st.session_state.label_risk = label_risk
            st.session_state.active_expander = active_expander

    with st.popover(label.prev_data_use, disabled=st.session_state.active_expander):
        st.write(label.prev_data_head)
        with st.form('my_form_compare'):
            st.write(label.label_demo)
            AGE =  st.slider(label.label_age, 30 , 100, 45)
            SEX = st.selectbox(label.label_sex, label.choice_sex)    
            SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
            AF = st.selectbox(label.label_af, label.choice_yesno)
            st.write(label.label_lab)
            if kidney_option == "ค่าการทำงานของไต (eGFR)"  or kidney_option == 'eGFR':
                EGFR = st.slider(label.label_egfr, 0.00, 200.00, 50.00)
                    
            elif kidney_option == "ครีตินีน (Creatinine)" or kidney_option == "Creatinine":
                if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                    unit = 'mg/dl'
                    CR = st.slider(f'{label.label_cr}, ({unit})', 0.00, 15.00, 0.60)
                elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                    unit = 'mmol/L'
                    CR_si = st.slider(f'{label.label_cr}, ({unit})', 0.000, 0.832, 0.033)
                    CR = CR_si/0.0555
                
                if (SEX == "หญิง" or SEX == 'Female') and CR <= 0.7:
                    COEF = -0.329
                    EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "หญิง" or SEX == 'Female') and CR > 0.7:
                    COEF = -1.209
                    EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "ชาย" or SEX == 'Male') and CR <= 0.9:
                    COEF = -0.411
                    EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "ชาย" or SEX == 'Male') and CR > 0.9:
                    COEF = -1.209
                    EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
        

            if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                unit = 'mg/dl'
                TCOL = st.slider(f'{label.label_tc}, ({unit})',0, 600, 150)
                HDL = st.slider(f'{label.label_hdl}, ({unit})',0, 200, 50)
            elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                unit = 'mmol/L'
                TCOL_si = st.slider(f'{label.label_tc}, ({unit})',0.000, 33.30, 8.000)
                HDL_si = st.slider(f'{label.label_hdl}, ({unit})',0.000, 11.10, 3.300)
                TCOL = TCOL_si/0.0555
                HDL = HDL_si/0.0555

            st.write(label.label_med)
            ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
            ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
            INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)
            
            if st.form_submit_button(label.prev_data_use):
                
                input_tranfrom2 = transform_input(AGE,SEX,SMOKE,AF,TCOL,HDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
                st.write(label.label_egfr_pred,  format(EGFR,".2f"))
                st.write(label.label_nonhdl_pred, format(input_tranfrom2[4],".2f"))
                new_prediction2 = pd.DataFrame.from_dict({1:input_tranfrom2}, columns=["AGE","SEX","SMOKE","AF","Non_HDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

                pred2, pred_5yr2 = prediction_survival(new_prediction2, non_hdl_mace)
                mace_5yr_prop2 = format(pred_5yr2,".2f")

                if pred_5yr2 < 2.80:
                    label_risk2 = label.low_risk
                if pred_5yr2 >= 2.80 and pred_5yr2 < 5.40:
                    label_risk2 = label.borderline_risk
                if pred_5yr2 >= 5.40 and pred_5yr2 < 10.00:
                    label_risk2 = label.intermediate_risk
                if pred_5yr2 >= 10.00:
                    label_risk2 = label.high_risk

                text2 = f""" {label.previous_risk}
                {label.pred_text} \n
                {str(mace_5yr_prop2)}%   {label_risk2} 
                """
                
                text3 = f""" {label.current_risk}
                {label.pred_text} \n
                {str(st.session_state.mace_5yr_prop)}%   {st.session_state.label_risk}
                """

                st.success(text3)
                st.warning(text2)
                st.error(label.disclaimer, icon="🚨")

        

# Model using LDL cholestoral level
elif (selected == "Model using LDL Cholestoral Level" or selected == "โมเดลทำนายการเกิดโรคโดยใช้ค่าไขมันคอเลสเตอรอลชนิดไม่ดี (LDL Cholestoral Level)"):
    
    st.header(selected)
    
    def transform_input_ldl(AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        LDL = LDL
        
        if SEX == 'ชาย' or SEX == 'Male':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ' or SMOKE == 'Current':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่' or ANY_HTN_MED == 'Yes':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่' or ANY_ORAL_DM == 'Yes':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่' or INSULIN == 'Yes':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่' or AF == 'Yes':
            AF = 1
        else:
            AF = 0
            

        return [AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]

    unit_option = st.selectbox(
                    label.label_unit_op,
                    label.choice_unit_op)
    kidney_option = st.selectbox(
                    label.label_kid_op,
                    label.choice_kid_op)
    
    with st.form('my_form_2'):
        st.write(label.label_demo)
        AGE =  st.slider(label.label_age, 30 , 100, 45)
        SEX = st.selectbox(label.label_sex, label.choice_sex)    
        SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
        AF = st.selectbox(label.label_af, label.choice_yesno)
        st.write(label.label_lab)
        if kidney_option == "ค่าการทำงานของไต (eGFR)"  or kidney_option == 'eGFR':
            EGFR = st.slider(label.label_egfr, 0.00, 200.00, 50.00)
                
        elif kidney_option == "ครีตินีน (Creatinine)" or kidney_option == "Creatinine":
            if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                unit = 'mg/dl'
                CR = st.slider(f'{label.label_cr}, ({unit})', 0.00, 15.00, 0.60)
            elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                unit = 'mmol/L'
                CR_si = st.slider(f'{label.label_cr}, ({unit})', 0.000, 0.832, 0.033)
                CR = CR_si/0.0555
            
            if (SEX == "หญิง" or SEX == 'Female') and CR <= 0.7:
                COEF = -0.329
                EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "หญิง" or SEX == 'Female') and CR > 0.7:
                COEF = -1.209
                EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "ชาย" or SEX == 'Male') and CR <= 0.9:
                COEF = -0.411
                EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
            elif (SEX == "ชาย" or SEX == 'Male') and CR > 0.9:
                COEF = -1.209
                EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
    

        if unit_option == "หน่วยวัด US" or unit_option == "US unit":
            unit = 'mg/dl'
            LDL = st.slider(f'{label.label_ldl}, ({unit})',0, 600, 150)
        elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
            unit = 'mmol/L'
            LDL_si = st.slider(f'{label.label_ldl}, ({unit})',0.000, 33.30, 8.324)
            LDL = LDL_si/0.0555


        st.write(label.label_med)
        ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
        ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
        INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)
        
        if st.form_submit_button(label.label_pred):
            
            input_tranfrom = transform_input_ldl(AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
            st.write(label.label_egfr_pred, format(EGFR, ".2f"))
            st.write(label.label_ldl_pred, format(input_tranfrom[4],".2f"))

            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","LDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, ldl_mace)
            mace_5yr_prop = format(pred_5yr,".2f")
            
            if pred_5yr < 2.80:
                label_risk = label.low_risk
            if pred_5yr >= 2.80 and pred_5yr < 5.40:
                label_risk = label.borderline_risk
            if pred_5yr >= 5.40 and pred_5yr < 10.00:
                label_risk = label.intermediate_risk
            if pred_5yr >= 10.00:
                label_risk = label.high_risk

            text = f"""
            {label.pred_text}{str(mace_5yr_prop)}% \n
            {label_risk}
            """

            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name=label.pred_text,
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title=label.yaxis_title,
                xaxis_title=label.xaxis_title,
                title=label.title,
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = label.xtick),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")

            st.error(label.disclaimer, icon="🚨")

            active_expander = False

            st.session_state.mace_5yr_prop = mace_5yr_prop
            st.session_state.label_risk = label_risk
            st.session_state.active_expander = active_expander

    with st.popover(label.prev_data_use, disabled=st.session_state.active_expander):
        st.write(label.prev_data_head)
        with st.form('my_form_compare'):
            st.write(label.label_demo)
            AGE =  st.slider(label.label_age, 30 , 100, 45)
            SEX = st.selectbox(label.label_sex, label.choice_sex)    
            SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
            AF = st.selectbox(label.label_af, label.choice_yesno)
            st.write(label.label_lab)
            if kidney_option == "ค่าการทำงานของไต (eGFR)"  or kidney_option == 'eGFR':
                EGFR = st.slider(label.label_egfr, 0.00, 200.00, 50.00)
                    
            elif kidney_option == "ครีตินีน (Creatinine)" or kidney_option == "Creatinine":
                if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                    unit = 'mg/dl'
                    CR = st.slider(f'{label.label_cr}, ({unit})', 0.00, 15.00, 0.60)
                elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                    unit = 'mmol/L'
                    CR_si = st.slider(f'{label.label_cr}, ({unit})', 0.000, 0.832, 0.033)
                    CR = CR_si/0.0555
                
                if (SEX == "หญิง" or SEX == 'Female') and CR <= 0.7:
                    COEF = -0.329
                    EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "หญิง" or SEX == 'Female') and CR > 0.7:
                    COEF = -1.209
                    EGFR = 144 * ((CR/0.7)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "ชาย" or SEX == 'Male') and CR <= 0.9:
                    COEF = -0.411
                    EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
                elif (SEX == "ชาย" or SEX == 'Male') and CR > 0.9:
                    COEF = -1.209
                    EGFR = 144 * ((CR/0.9)**(COEF)) * (0.993 ** AGE)
        

            if unit_option == "หน่วยวัด US" or unit_option == "US unit":
                unit = 'mg/dl'
                LDL = st.slider(f'{label.label_ldl}, ({unit})',0, 600, 150)
            elif unit_option == "หน่วยวัด SI" or unit_option == "SI unit":
                unit = 'mmol/L'
                LDL_si = st.slider(f'{label.label_ldl}, ({unit})',0.000, 33.30, 8.324)
                LDL = LDL_si/0.0555


            st.write(label.label_med)
            ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
            ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
            INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)
            
            if st.form_submit_button(label.prev_data_use):
                
                input_tranfrom2 = transform_input_ldl(AGE,SEX,SMOKE,AF,LDL,EGFR,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
                st.write(label.label_egfr_pred, format(EGFR, ".2f"))
                st.write(label.label_ldl_pred, format(input_tranfrom2[4],".2f"))

                new_prediction2 = pd.DataFrame.from_dict({1:input_tranfrom2}, columns=["AGE","SEX","SMOKE","AF","LDL","EGFR","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

                pred2, pred_5yr2 = prediction_survival(new_prediction2, ldl_mace)
                mace_5yr_prop2 = format(pred_5yr2,".2f")

                if pred_5yr2 < 2.80:
                    label_risk2 = label.low_risk
                if pred_5yr2 >= 2.80 and pred_5yr2 < 5.40:
                    label_risk2 = label.borderline_risk
                if pred_5yr2 >= 5.40 and pred_5yr2 < 10.00:
                    label_risk2 = label.intermediate_risk
                if pred_5yr2 >= 10.00:
                    label_risk2 = label.high_risk

                text2 = f""" {label.previous_risk}
                {label.pred_text} \n
                {str(mace_5yr_prop2)}%   {label_risk2} 
                """
                
                text3 = f""" {label.current_risk}
                {label.pred_text} \n
                {str(st.session_state.mace_5yr_prop)}%   {st.session_state.label_risk}
                """
                
                st.success(text3)
                st.warning(text2)
                st.error(label.disclaimer, icon="🚨")

# Model using BMI
elif (selected == "Model using Body Mass Index" or selected =="โมเดลทำนายการเกิดโรคโดยใช้ดัชนีมวลกาย (Body Mass Index)"):
    
    st.header(selected)
    
    def transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN):
        BMI = BMI

        if SEX == 'ชาย' or SEX == 'Male':
            SEX = 1
        else:
            SEX = 0
    
        if SMOKE == 'สูบ' or SMOKE == 'Current':
            SMOKE = 1
        else:
            SMOKE = 0
            
        if ANY_HTN_MED == 'ใช่' or ANY_HTN_MED == 'Yes':
            ANY_HTN_MED = 1
        else:
            ANY_HTN_MED = 0
    
        if ANY_ORAL_DM == 'ใช่' or ANY_ORAL_DM == 'Yes':
            ANY_ORAL_DM = 1
        else:
            ANY_ORAL_DM = 0
        
        if INSULIN == 'ใช่' or INSULIN == 'Yes':
            INSULIN = 1
        else:
            INSULIN = 0
    
        if AF == 'ใช่' or AF == 'Yes':
            AF = 1
        else:
            AF = 0
            
        return [AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN]

    bmi_option = st.selectbox(
                    label.label_bmi_op,
                    label.choice_bmi_op)

    with st.form('my_form_3'):
        st.write(label.label_demo)
        AGE =  st.slider(label.label_age, 30 , 100, 45)
        SEX = st.selectbox(label.label_sex, label.choice_sex)    
        SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
        AF = st.selectbox(label.label_af, label.choice_yesno)

        if bmi_option == "ดัชนีมวลกาย (BMI)" or bmi_option == "Body Mass Index (BMI)":
            BMI =  st.slider(label.label_bmi,0.00, 75.00, 25.00)
        elif bmi_option == "ค่าน้ำหนักและส่วนสูง" or bmi_option == "Body Weight/Height":
            BW =  st.slider(label.label_bw,0.00, 200.00, 50.00)
            HT = st.slider(label.label_ht,0.00, 220.00, 150.00)
            BMI = BW/((HT/100)**2)
        
        st.write(label.label_med)
        ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
        ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
        INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)
        
        if st.form_submit_button(label.label_pred):
            
            input_tranfrom = transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
            st.write(label.label_bmi_pred, format(input_tranfrom[4],".2f"))

            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","SMOKE","AF","BMI","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

            pred, pred_5yr = prediction_survival(new_prediction, bmi_mace)
            mace_5yr_prop = format(pred_5yr,".2f")
            
            if pred_5yr < 2.80:
                label_risk = label.low_risk
            if pred_5yr >= 2.80 and pred_5yr < 5.40:
                label_risk = label.borderline_risk
            if pred_5yr >= 5.40 and pred_5yr < 10.00:
                label_risk = label.intermediate_risk
            if pred_5yr >= 10.00:
                label_risk = label.high_risk

            text = f"""
            {label.pred_text}{str(mace_5yr_prop)}% \n
            {label_risk}
            """
            st.success(text, icon="💔")
            
            fig = go.Figure([
                go.Scatter(
                    name=label.pred_text,
                    x=time_points,
                    y=pred,
                    mode='lines',
                    line=dict(color='#D04848'),
                    fill='tonexty'
                )
            ])
            fig.update_layout(
                yaxis_title=label.yaxis_title,
                xaxis_title=label.xaxis_title,
                title=label.title,
                hovermode="x",
                yaxis_tickformat = '.2%',
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 12, 24, 36, 48, 60],
                    ticktext = label.xtick),
                font=dict(
                size=15,
                color="RebeccaPurple"),
            )

            event = st.plotly_chart(fig, on_select="rerun")

            st.error(label.disclaimer, icon="🚨")

            active_expander = False

            st.session_state.mace_5yr_prop = mace_5yr_prop
            st.session_state.label_risk = label_risk
            st.session_state.active_expander = active_expander

    with st.popover(label.prev_data_use, disabled=st.session_state.active_expander):
        st.write(label.prev_data_head)
        with st.form('my_form_compare'):
            st.write(label.label_demo)
            AGE =  st.slider(label.label_age, 30 , 100, 45)
            SEX = st.selectbox(label.label_sex, label.choice_sex)    
            SMOKE = st.selectbox(label.label_smoker, label.choice_smoker)
            AF = st.selectbox(label.label_af, label.choice_yesno)
            st.write(label.label_lab)

            if bmi_option == "ดัชนีมวลกาย (BMI)" or bmi_option == "Body Mass Index (BMI)":
                BMI =  st.slider(label.label_bmi,0.00, 75.00, 25.00)
            elif bmi_option == "ค่าน้ำหนักและส่วนสูง" or bmi_option == "Body Weight/Height":
                BW =  st.slider(label.label_bw,0.00, 200.00, 50.00)
                HT = st.slider(label.label_ht,0.00, 220.00, 150.00)
                BMI = BW/((HT/100)**2)
        
            st.write(label.label_med)
            ANY_HTN_MED = st.selectbox(label.label_htn_med, label.choice_yesno)
            ANY_ORAL_DM = st.selectbox(label.label_dm_med, label.choice_yesno)
            INSULIN = st.selectbox(label.label_insulin_med, label.choice_yesno)

            if st.form_submit_button(label.prev_data_use):
                
                input_tranfrom2 = transform_input_bmi(AGE,SEX,SMOKE,AF,BMI,ANY_HTN_MED,ANY_ORAL_DM,INSULIN)
                st.write(label.label_bmi_pred, format(input_tranfrom2[4],".2f"))

                new_prediction2 = pd.DataFrame.from_dict({1:input_tranfrom2}, columns=["AGE","SEX","SMOKE","AF","BMI","ANY_HTN_MED","ANY_ORAL_DM","INSULIN"], orient='index')

                pred2, pred_5yr2 = prediction_survival(new_prediction2, bmi_mace)
                mace_5yr_prop2 = format(pred_5yr2,".2f")

                if pred_5yr2 < 2.80:
                    label_risk2 = label.low_risk
                if pred_5yr2 >= 2.80 and pred_5yr2 < 5.40:
                    label_risk2 = label.borderline_risk
                if pred_5yr2 >= 5.40 and pred_5yr2 < 10.00:
                    label_risk2 = label.intermediate_risk
                if pred_5yr2 >= 10.00:
                    label_risk2 = label.high_risk

                text2 = f""" {label.previous_risk}
                {label.pred_text} \n
                {str(mace_5yr_prop2)}%   {label_risk2} 
                """
                
                text3 = f""" {label.current_risk}
                {label.pred_text} \n
                {str(st.session_state.mace_5yr_prop)}%   {st.session_state.label_risk}
                """
                
                st.success(text3)
                st.warning(text2)
                st.error(label.disclaimer, icon="🚨")

st.caption(label.text_caption)

