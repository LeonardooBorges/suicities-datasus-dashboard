import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import pickle
import joblib
import altair as alt
from sklearn import metrics
import shap
import streamlit.components.v1 as components

# Plotting
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mapclassify
import py7zr
import os

root = "../"

dict_uf_cod = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE',
29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC', 43: 'RS',
50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}

def remove_last_digit(x):
    return np.floor(x.astype(int) / 10).astype(int)

def get_train_df(satscan=False):
    if satscan:
        return pd.read_csv("Models/train_data_classification_satscan.csv", index_col=0)
    else:
        return pd.read_csv("Models/train_data_classification_highest_rates.csv", index_col=0)

def get_test_df(satscan=False):
    if satscan:
        return pd.read_csv("Models/test_data_classification_satscan.csv", index_col=0)
    else:
        return pd.read_csv("Models/test_data_classification_highest_rates.csv", index_col=0)
        
def get_test_data(test_df, satscan=False):
    if satscan:
        X_test = test_df.drop(columns=["RISK", "MUNCOD"])
        y_test = test_df["RISK"]
    else:
        X_test = test_df.drop(columns=["TARGET", "MUNCOD"])
        y_test = test_df["TARGET"]
    return X_test, y_test

def get_train_data(train_df, satscan=False):
    if satscan:
        X_train = train_df.drop(columns=["RISK"])
        y_train = train_df["RISK"]
    else:
        X_train = train_df.drop(columns=["TARGET"])
        y_train = train_df["TARGET"]
    return X_train, y_train

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def get_confusion_matrix(y_test, y_pred):
    st.markdown("""
        ### Matriz de Confusão

        A matriz de confusão é uma tabela que permite a visualização das previsões efetuadas por um modelo. 
        As colunas indicam os **valores preditos** pelo modelo, enquanto as **linhas** correspondem aos **valores reais** das classes das observações.
        Assim, espera-se obter a maior quantidade possível de observações na diagonal principal da tabela. 
        """)
    st.write(metrics.confusion_matrix(y_test, y_pred))

def calculate_metrics(y_test, y_pred):
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
    st.markdown("""
            #### Descrição das métricas
            - **Acurácia**: é a fração de observações cujas saídas foram corretamente previstas pelo modelo.
            - **Precisão**: é a proporção de observações de previsão positiva corretamente classificadas pelo modelo.
            - **Revocação**: é a proporção de observações de classe positiva corretamente identificadas pelo modelo.
            - **F1-Score**: é uma combinação da precisão com a revocação, sendo definido pela seguinte fórmula: $F1-Score = 2*Precisao*Revocacao/(Precisão + Revocação)$
        """)
    
def get_predictions_2018(model, satscan):
    st.markdown("""
            ### Previsões do Modelo
    """)

    st.markdown(
    '<p> O mapa abaixo mostra o resultado das previsões do modelo para o ano de 2018.</p>'
    'Em <span style="color:red;"><b>vermelho</b></span> são destacados os municípios para os quais a previsão do modelo foi <b>incorreta</b>, e em <span style="color:blue;"><b>azul</b></span>, aqueles cuja previsão foi <b>correta</b>.', unsafe_allow_html=True
    )
    if satscan:
        prefix = "./Models/img/satscan_"
    else:
        prefix = "./Models/img/highest_rates_"
    if model == "Naive Bayes":
        st.image(prefix + "naive_bayes.png", use_column_width=True)
    elif model == "Regressão Logística":
        st.image(prefix + "logistic_regression.png", use_column_width=True)
    elif model == "Random Forest":
        st.image(prefix + "random_forest.png", use_column_width=True)
    elif model == "SVC (Linear)":
        st.image(prefix + "svc_linear.png", use_column_width=True)
    elif model == "SVC (RBF)":
        st.image(prefix + "svc_rbf.png", use_column_width=True)

def run_model(model, satscan):
    train_df = get_train_df(satscan=satscan)
    test_df = get_test_df(satscan=satscan)
    X_test, y_test = get_test_data(test_df, satscan=satscan)
    X_train, y_train = get_train_data(train_df, satscan=satscan)
    suffix = "satscan" if satscan else "highest_rates"
    filename = ""
    classifier = None
    mm_scaler = joblib.load("Models/sav/mm_x_{}.save".format(suffix))
    if model != "Selecione um modelo":
        if model == "Naive Bayes":
            classifier = pickle.load(open("Models/sav/naive_bayes_{}.sav".format(suffix), 'rb'))
            X_test  = X_test.drop(columns=list(dict_uf_cod.values()))
        elif model == "Regressão Logística":
            X_train = mm_scaler.transform(X_train)
            X_test = mm_scaler.transform(X_test) 
            classifier = pickle.load(open("Models/sav/logistic_regression_{}_mm.sav".format(suffix), 'rb'))
        elif model == "Random Forest":
            classifier = pickle.load(open("Models/sav/random_forest_{}.sav".format(suffix), 'rb'))
        elif model == "SVC (Linear)":
            X_train = mm_scaler.transform(X_train)
            X_test = mm_scaler.transform(X_test) 
            classifier = pickle.load(open("Models/sav/svm_linear_{}_mm.sav".format(suffix), 'rb'))
        elif model == "SVC (RBF)":
            X_train = mm_scaler.transform(X_train)
            X_test = mm_scaler.transform(X_test) 
            classifier = pickle.load(open("Models/sav/svm_rbf_{}_mm.sav".format(suffix), 'rb'))
        else:
            return
    if classifier != None:
        y_pred = classifier.predict(X_test) 
        shap_model_list = ["SVC (Linear)", "Random Forest", "Regressão Logística"]
        if model not in shap_model_list:
            analysis = st.radio("Visualizar resultados:",('Matriz de Confusão', 'Métricas do modelo', 'Previsões para 2018'))
            if analysis == "Matriz de Confusão":
                get_confusion_matrix(y_test, y_pred)
            elif analysis == "Métricas do modelo":
                calculate_metrics(y_test, y_pred)
            else:
                get_predictions_2018(model, satscan)
        else:
            analysis = st.radio("Visualizar resultados:",('Matriz de Confusão', 'Métricas do modelo', 'Previsões para 2018', 'Análise SHAP'))
            if analysis == "Matriz de Confusão":
                get_confusion_matrix(y_test, y_pred)
            elif analysis == "Métricas do modelo":
                calculate_metrics(y_test, y_pred)
            elif analysis == "Previsões para 2018":
                get_predictions_2018(model, satscan)
            else:
                get_shap_analysis(model, classifier, X_train, X_test, satscan=satscan)


def get_cadmun(test_df):
    cadmun = pd.read_csv("Models/CADMUN.csv")
    cadmun = cadmun[cadmun["SITUACAO"] == "ATIVO"]
    cadmun = cadmun.sort_values(by="MUNNOMEX")
    cadmun = cadmun[["MUNCOD", "MUNNOME"]]
    muncod_list = list(test_df["MUNCOD"])
    cadmun = cadmun[cadmun['MUNCOD'].isin(muncod_list)]
    return cadmun

        
def get_shap_analysis(model, classifier, X_train, X_test, satscan=False):
    test_df = get_test_df(satscan=satscan)
    st.markdown("""
        ### Análise SHAP
        A biblioteca SHAP permite uma melhor compreensão dos atributos que mais impactaram a decisão do classificador.
        - **Output value**: é a previsão para o município em questão
        - **Base value**: é o valor que seria predito caso não houvessem informações sobre os atributos daquele município, ou seja, é o valor médio das previsões
        - **Vermelho/Azul**: as cores indicam os atributos que mais "empurram" a previsão para a direita (mostrados em vermelho) ou para a esquerda (mostrados em azul)
    """)

    cadmun = get_cadmun(test_df)
    city = st.selectbox('Selecione uma cidade:', list(cadmun['MUNNOME']))
    muncod = int(cadmun[cadmun["MUNNOME"] == city]["MUNCOD"])
    col_target = "TARGET" if satscan == False else "RISK"
    data_for_prediction = test_df.loc[test_df['MUNCOD'] == muncod].drop(columns=["MUNCOD", col_target])
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    if muncod:
        if model == "SVC (Linear)" or model == "Regressão Logística":
            explainer = shap.LinearExplainer(classifier, X_train, feature_perturbation="interventional")
            shap_values = explainer.shap_values(data_for_prediction)
            shap.initjs()
            st_shap(shap.force_plot(explainer.expected_value, shap_values, data_for_prediction), 400)
        elif model == "Random Forest":
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(data_for_prediction)
            shap.initjs()
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction), 400)
   