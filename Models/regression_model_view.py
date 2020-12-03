import streamlit as st
import numpy as np
from . import regression_model_data as dt

def present_regression_model():
  st.markdown(
    """
    # Modelos de Regressão 

    Os modelos de regressão aqui apresentados têm por objetivo prever a taxa de suicídios nos municípios brasileiros.

    As variáveis preditivas utilizadas foram as taxas de doenças e a taxa de suicídios por município no ano anterior, 
    bem como o estado onde se encontram.
    
    Os modelos foram treinados com dados do **período de 2008 a 2017**.
    """ 
  )

  options = np.append(['Selecione um modelo'], ["Regressão Linear", "ElasticNet", "SVR"])            
  model = st.selectbox('Selecione um modelo:', options)

  dt.run_model(model)