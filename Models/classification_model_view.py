import streamlit as st
from . import classification_model_data as dt
import numpy as np

def present_classification_model():
  st.markdown(
    """
    # Modelos de Classificação
    """

    
  )
  analysis = st.radio("Qual modelo?",('Mediana Nacional', 'Clusters do SatScan'))
  if analysis == "Mediana Nacional":
    st.markdown(
      """
      ## Modelo da Mediana Nacional

      O objetivo deste modelo é prever os municípios cuja taxa de suicídio é superior à mediana das taxas no país. 
      Para esses municípios, é atribuído o valor 1, e para os municípios restantes, é atribuído o valor 0.
      """
    )
  else:
    st.markdown(
      """
      ## Modelo de Clusters do SatScan

      Este modelo utilia as saídas do software de clusterização **SatScan** para determinar municípios pertencentes a 
      clusters de alto risco de suicídio. Tais municípios são classificados como 1, e os municípios restantes são classificados como 0.
      """
    )
  
  options = np.append(['Selecione um modelo'], ["Naive Bayes", "Regressão Logística", "Random Forest", "SVC (Linear)", "SVC (RBF)"])
  model = st.selectbox('Selecione um modelo:', options)

  if analysis == "Mediana Nacional":
    dt.highest_rates_model(model)
  else:
    dt.satscan_model(model)