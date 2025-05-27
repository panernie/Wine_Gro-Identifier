import streamlit as st
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import joblib
from PIL import Image
import random
from tqdm import tqdm
import time
import os
import base64
import pickle

#give the path of the model you want to use from the folder Model
features = pd.read_csv('feature/feature.csv')
feas = list(features.feature.values)
model_rf= joblib.load('model/rf_model_classf_kf_cv_best_0.pkl')
model_et=   joblib.load('model/et_model_classf_kf_cv_best_0.pkl')
model_lightgbm = joblib.load('model/lightgbm_model_classf_kf_cv_best_1.pkl')
replacement_map = {0: "HL", 1: "YC", 2: "YN", 3: "QTX", 4: "HSP"}

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

st.set_page_config()

# Logo image
image = Image.open('cover.png')
st.image(image, use_column_width=True)
# Page title
st.markdown(
    """
# Ningxia Wine Geo-Identifier
This tool is an intelligent tool specially developed for the geographical identification of wines in the five major regions of Ningxia. It aims to help users accurately identify the origin of wines by providing HS-SPME-GC-MS data analysis of wines.
This tool adopts advanced machine learning models and integrates multiple models for synchronous discrimination, ensuring the high accuracy and reliability of the recognition results.
It aims to help professionals, research institutions and enthusiasts better understand the uniqueness of the five major wine regions in Ningxia and promote the development of the local wine industry.
"""
)

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("OR, upload your input file", type=['csv'])
    st.sidebar.markdown(
        """
[Example input file](https://github.com/panernie/AHTPeptideFusion/blob/main/test192.csv)
"""
    )

    #title = st.text_input("Input your sequence, eg. IRW")
    
    
if st.sidebar.button('Predict'):

    T1 = time.time()

    df = pd.read_csv(uploaded_file)
    new_df = pd.DataFrame(0, index=df.index, columns=feas)
    common_columns = df.columns.intersection(new_df.columns)
    new_df[common_columns] = df[common_columns]
    st.header('**Original input data**')
    st.write(f"{new_df.shape[0]} samples were identified")
    with st.spinner("Please wait..."):
        time.sleep(1)
        # load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)
        rf_pred = model_rf.predict(new_df)
        et_pred = model_et.predict(new_df)
        lightgbm_pre = model_lightgbm.predict(new_df)
        dfa = pd.DataFrame(np.array([rf_pred,et_pred,lightgbm_pre]).T,columns=["RandomForest model","ExtraTrees model","Lightgbmmodel"])
        dfa.index.name = 'sample'
        dfa.index = pd.RangeIndex(start=1, stop=len(dfa)+1, step=1, name='sample')
        dfa.replace(replacement_map, inplace=True)

        dfa.to_csv("output\prediction.csv")
    file_names = time.time()

    # print(df_all)
    #df_all = pd.read_csv("output\prediction.csv")
    df_10 = dfa[:10]
    T2 = time.time()
    st.success("Done!")
    st.write('Program run time:%sms' % ((T2 - T1) * 1000))
    st.header('**Output data**')
    st.write("Only the first 10 results are displayed!")
    st.write(df_10)
    st.markdown(filedownload(dfa), unsafe_allow_html=True)
else:
    st.info('Upload input data in the sidebar to start!')
