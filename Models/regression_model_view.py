import streamlit as st
import numpy as np
from . import regression_model_data as dt

def present_regression_model():
  st.markdown(
    """
    # Modelos de Regressão 
    """ 
  )

  options = np.append(['Selecione um modelo'], ["Adaboost", "Gradient Boost", "SVR"])            
  model = st.selectbox('Selecione um modelo:', options)

  dt.run_model(model)