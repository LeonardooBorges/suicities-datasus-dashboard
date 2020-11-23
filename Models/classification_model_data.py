import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import pickle
import joblib
import altair as alt
from sklearn import metrics

# Plotting
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mapclassify
import py7zr
import os

root = "../"

def remove_last_digit(x):
    return np.floor(x.astype(int) / 10).astype(int)

def get_test_df(satscan=False):
    if satscan:
        test_df = pd.read_csv("Models/test_data_classification_highest_rates.csv", index_col=0)
        X_test = test_df.drop(columns=["TARGET", "MUNCOD"])
        y_test = test_df["TARGET"]
        return test_df, X_test, y_test
    else:
        test_df = pd.read_csv("Models/test_data_classification_satscan.csv", index_col=0)
        X_test = test_df.drop(columns=["RISK", "MUNCOD"])
        y_test = test_df["RISK"]
        return test_df, X_test, y_test

def plot_map(y_test, y_pred, test_df, model):
    zipFileName = 'Maps/BRMUE250GC_SIR.7z'
    if not os.path.isfile('Maps/BRMUE250GC_SIR.shp'):
        print('Unzipping BRMUE250GC_SIR files')
        with py7zr.SevenZipFile(zipFileName, 'r') as archive:
            archive.extractall("Maps/")
    
    gd = gpd.read_file('Maps/BRMUE250GC_SIR.shp')
    
    mun_risk_ids_pred = test_df[y_pred == 1]['MUNCOD'].astype(int).tolist()
    mun_risk_ids_true = test_df[y_test == 1]['MUNCOD'].astype(int).tolist()
    mun_risk_ids_1_correct = [x for x in mun_risk_ids_pred if x in mun_risk_ids_true]

    mun_risk_ids_pred_0 = test_df[y_pred == 0]['MUNCOD'].astype(int).tolist()
    mun_risk_ids_true_0 = test_df[y_test == 0]['MUNCOD'].astype(int).tolist()
    mun_risk_ids_0_correct = [x for x in mun_risk_ids_pred_0 if x in mun_risk_ids_true_0]

    mun_risk_ids = mun_risk_ids_1_correct + mun_risk_ids_0_correct
    mun_risk_ids_wrong = [x for x in mun_risk_ids_pred if x not in mun_risk_ids_true] + [x for x in mun_risk_ids_true if x not in mun_risk_ids_pred]


    fig, ax = plt.subplots(figsize=(15,15))
    gd.plot(ax=ax, color="white", edgecolor='black')
    gd_risk = gd[remove_last_digit(gd['CD_GEOCMU']).apply(lambda x: x in mun_risk_ids)]
    plot_risk = gd_risk.plot(ax=ax, color="blue")

    gd_risk_wrong = gd[remove_last_digit(gd['CD_GEOCMU']).apply(lambda x: x in mun_risk_ids_wrong)]
    plot_risk_wrong = gd_risk_wrong.plot(ax=ax, color="red")

    blue_patch = mpatches.Patch(color='blue', label='Previsão Correta')
    red_patch = mpatches.Patch(color='red', label='Previsão Incorreta')
    plt.title("Previsões do modelo de " + model + " para o ano de 2018")
    plt.legend(handles=[red_patch,blue_patch])
    plt.axis('off')
    st.pyplot(fig)


def calculate_metrics(y_test, y_pred):
    st.markdown("""
        ### Matriz de Confusão

        A matriz de confusão é uma tabela que permite a visualização das previsões efetuadas por um modelo. 
        As colunas indicam os **valores preditos** pelo modelo, enquanto as **linhas** correspondem aos **valores reais** das classes das observações.
        Assim, espera-se obter a maior quantidade possível de observações na diagonal principal da tabela. 
        """)
    st.write(metrics.confusion_matrix(y_test, y_pred))

    st.markdown("""
        ### Métricas do Modelo

        Na tabela abaixo, são mostradas as métricas de teste para o modelo.
        """)
    accuracy =  metrics.accuracy_score(y_test,y_pred)
    prfs = metrics.precision_recall_fscore_support(y_test,y_pred, zero_division=0)
    precision = prfs[0].mean()
    recall = prfs[1].mean()
    fscore = prfs[2].mean()
    metrics_df = pd.DataFrame({"Acurácia": [accuracy], "Precisão": [precision], "Revocação": [recall], "F1-Score":[fscore]})
    st.write(metrics_df)

    metrics_expander = st.beta_expander("Descrição das métricas:", expanded=False)
    with metrics_expander:
        st.markdown("""
            - **Acurácia**: é a fração de observações cujas saídas foram corretamente previstas pelo modelo.
            - **Precisão**: é a proporção de observações de previsão positiva corretamente classificadas pelo modelo.
            - **Revocação**: é a proporção de observações de classe positiva corretamente identificadas pelo modelo.
            - **F1-Score**: é uma combinação da precisão com a revocação, sendo definido pela seguinte fórmula: $F1-Score = 2*Precisao*Revocacao/(Precisão + Revocação)$
        """)
    
def highest_rates_model(model):
    test_df, X_test, y_test = get_test_df(satscan=False)
    filename = ""
    if model != "Selecione um modelo":
        if model == "Naive Bayes":
            filename = "Models/sav/naive_bayes_highest_rates.sav"
        elif model == "Regressão Logística":
            options_scaler_highest_rates = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_highest_rates)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/logistic_regression_highest_rates_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/logistic_regression_highest_rates_mm.sav"
        elif model == "Random Forest":
            filename = "Models/sav/random_forest_highest_rates.sav"
        elif model == "SVC (Linear)":
            options_scaler_highest_rates = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_highest_rates)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/svm_linear_highest_rates_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/svm_linear_highest_rates_mm.sav"
        elif model == "SVC (RBF)":
            options_scaler_highest_rates = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_highest_rates)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/svm_rbf_highest_rates_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_highest_rate.save")
                X_test = scaler.transform(X_test) 
                filename = "Models/sav/svm_rbf_highest_rates_mm.sav"  
        else:
            return
        
        classifier = pickle.load(open(filename, 'rb'))
        y_pred = classifier.predict(X_test) 
        calculate_metrics(y_test, y_pred)

        st.markdown("""
            ### Previsões do Modelo
        """)

        st.markdown(
        '<p> O mapa abaixo mostra o resultado das previsões do modelo para o ano de 2018.</p>'
        'Em <span style="color:red;"><b>vermelho</b></span> são destacados os municípios para os quais a previsão do modelo foi <b>incorreta</b>, e em <span style="color:blue;"><b>azul</b></span>, aqueles cuja previsão foi <b>correta</b>.', unsafe_allow_html=True
        )
        plot_map(y_test, y_pred, test_df, model)

def satscan_model(model):
    test_df, X_test, y_test = get_test_df(satscan=True)
    filename = ""
    if model != "Selecione um modelo":
        if model == "Naive Bayes":
            filename = "Models/sav/naive_bayes_satscan.sav"
        elif model == "Regressão Logística":
            options_scaler_satscan = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_satscan)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/logistic_regression_satscan_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/logistic_regression_satscan_mm.sav"
            st.write("Modelo escolhido: ", model + " com " + scaler + " Scaling")
        elif model == "Random Forest":
            filename = "Models/sav/random_forest_satscan.sav"
        elif model == "SVC (Linear)":
            options_scaler_satscan = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_satscan)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/svm_linear_satscan_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/svm_linear_satscan_mm.sav"
            st.write("Modelo escolhido: ", model + " com " + scaler + " Scaling")
        elif model == "SVC (RBF)":
            options_scaler_satscan = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler_satscan)
            if scaler == "Standard":
                scaler = joblib.load("Models/sav/sc_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/svm_rbf_satscan_sc.sav"
            else:
                scaler = joblib.load("Models/sav/mm_x_satscan.save")
                X_test = scaler.transform(X_test)
                filename = "Models/sav/svm_rbf_satscan_mm.sav"  
            st.write("Modelo escolhido: ", model + " com " + scaler + " Scaling") 
        else:
            return
        
        classifier = pickle.load(open(filename, 'rb'))
        y_pred = classifier.predict(X_test) 

        calculate_metrics(y_test, y_pred)
        plot_map(y_test, y_pred, test_df, model)