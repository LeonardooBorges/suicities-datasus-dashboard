import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import pickle
import joblib
from sklearn import metrics
import shap
import altair as alt
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


def calculate_metrics(y_test, y_pred, X_test):
    st.markdown("""
        ### Métricas do Modelo

        A primeira métrica calculada foi o RMSE, que consiste na raiz quadrada dos erros médios do modelo (sendo o erro a diferença entre o valor previsto e o real).
        O valor obtido pode ser comparado com o RMSE do modelo *baseline*, que consiste em prever que a taxa de suicídios em 2018 será a mesma do ano anterior.
        """)

    rmse = np.sqrt(metrics.mean_squared_error(y_pred,y_test))
    rmse_baseline = np.sqrt(metrics.mean_squared_error(X_test["PREVIOUS"],y_test))
    st.write("**RMSE**: {:.2f}".format(rmse))
    st.write("**RMSE da baseline**: {:.2f}".format(rmse_baseline))
    

    st.markdown("""

        O modelo de regressão também foi avaliado sob a óptica das métricas de classificação. Para isso, as taxas de suicídios previstas foram comapradas
        com as taxas do ano anterior, sendo classificadas como 1 as observações para as quais a taxa sofreu um aumento, e 0 caso contrário.
        Na tabela abaixo, são mostradas os resultados para essas métricas:
        """)

    up_df = pd.DataFrame({"Pred": y_pred, "Real": y_test, "Previous": X_test["PREVIOUS"]})
    up_df["UP"] = up_df["Previous"] < up_df["Real"]
    up_df["UP_PRED"] = up_df["Previous"] < up_df["Pred"]
    up_df["UP"] = up_df["UP"].astype(int)
    up_df["UP_PRED"] = up_df["UP_PRED"].astype(int)
    accuracy = metrics.accuracy_score(up_df["UP"], up_df["UP_PRED"])
    recall = metrics.recall_score(up_df["UP"], up_df["UP_PRED"])
    precision = metrics.precision_score(up_df["UP"], up_df["UP_PRED"])
    fscore = metrics.f1_score(up_df["UP"], up_df["UP_PRED"])
    metrics_df = pd.DataFrame({"Acurácia": [accuracy], "Precisão": [precision], "Revocação": [recall], "F1-Score":[fscore]})
    st.write(metrics_df)

def get_cadmun(df):
    cadmun = pd.read_csv("CSV/Cadmun/CADMUN.csv")
    cadmun = cadmun[cadmun["SITUACAO"] == "ATIVO"]
    cadmun = cadmun.sort_values(by="MUNNOMEX")
    cadmun = cadmun[["MUNCOD", "MUNNOME"]]
    muncod_list = list(df["MUNCOD"])
    cadmun = cadmun[cadmun['MUNCOD'].isin(muncod_list)]
    return cadmun

def get_predictions_2018(X_test_original, y_pred):
    df = pd.DataFrame({"MUNCOD": list(X_test_original["MUNCOD"]), "y_pred": list(y_pred)})
    suicide_df = pd.read_csv("./CSV/Suicide/suicide_rates_08_18.csv", index_col=0)
    result = pd.merge(suicide_df, df, on="MUNCOD", how="inner")

    cadmun = get_cadmun(result)
    city = st.selectbox('Selecione uma cidade:', list(cadmun['MUNNOME']))
    muncod = int(cadmun[cadmun["MUNNOME"] == city]["MUNCOD"])

    final_df = result.loc[result['MUNCOD'] == muncod]
    plot_df = final_df.drop(columns=["MUNCOD", "RATE_18"])
    plot_df.columns = [str(x) for x in range(2008,2019)]
    plot_df = plot_df.T
    plot_df.columns = ['Prevista']
    real = final_df.drop(columns=["MUNCOD", "y_pred"]).T
    plot_df["Real"] = list(real.iloc[:, 0])
    plot_df.index.name = "x"
    data = plot_df.reset_index().melt('x')

    scales = alt.selection_interval(bind='scales')
    graph = alt.Chart(data, title="Evolução da taxa de suicídios (2008-2018)").mark_line(point=True).encode(
        x=alt.X('x', title='Ano'),
        y=alt.Y('value', title='Taxa'),
        color=alt.Color('variable', title="Taxa"),
        tooltip=[alt.Tooltip('x', title='Ano'), alt.Tooltip('value', title='Taxa')]
    ).add_selection(scales).interactive()

    st.altair_chart((graph).configure_view(strokeOpacity=0).configure_title(fontSize=12).properties(width=700, height=410))

def run_model(model):
    test_df = pd.read_csv("Models/test_data_regression.csv", index_col=0)
    X_test_original = test_df.drop(columns=["RATE"])
    X_test = X_test_original.copy()
    y_test = test_df["RATE"]
    features = joblib.load("Models/sav/selected_cor_features")
    mm_scaler_x = joblib.load("Models/sav/mm_x_regression.save")
    mm_scaler_y = joblib.load("Models/sav/mm_y_regression.save")
    filename = ""
    regressor = None

    if model != "Selecione um modelo":
        if model == "Regressão Linear":
            X_test = X_test[features]
            X_test_transf = pd.DataFrame(mm_scaler_x.transform(X_test), index=X_test.index, columns=X_test.columns)
            regressor = pickle.load(open("Models/sav/linear_regression.sav", 'rb'))
            y_pred = regressor.predict(X_test_transf)
            y_pred = mm_scaler_y.inverse_transform(y_pred.reshape(-1,1))
            y_pred = y_pred.ravel()
        elif model == "ElasticNet":
            X_test = X_test[features]
            X_test_transf = pd.DataFrame(mm_scaler_x.transform(X_test), index=X_test.index, columns=X_test.columns)
            regressor = pickle.load(open("Models/sav/elastic_net_regression.sav", 'rb'))
            y_pred = regressor.predict(X_test_transf)
            y_pred = mm_scaler_y.inverse_transform(y_pred.reshape(-1,1))
            y_pred = y_pred.ravel()
        elif model == "Random Forest":
            st.write("Not implemented")
        elif model == "SVR":
            X_test = X_test[features]
            X_test_transf = pd.DataFrame(mm_scaler_x.transform(X_test), index=X_test.index, columns=X_test.columns)
            regressor = pickle.load(open("Models/sav/svr_mm_regression.sav", 'rb'))
            y_pred = regressor.predict(X_test_transf)
            y_pred = mm_scaler_y.inverse_transform(y_pred.reshape(-1,1))
            y_pred = y_pred.ravel()
        else:
            return
    if regressor != None:
        my_expander = st.beta_expander("Descrição dos hiperparâmetros do modelo:", expanded=False)
        with my_expander:
            params = regressor.get_params()
            params_df = pd.DataFrame(params.items(), columns=['Parâmetro', 'Valor'])
            st.write(params_df)
        analysis = st.radio("Visualizar resultados:",('Métricas do modelo', 'Previsões para 2018',))
        if analysis == "Métricas do modelo":
            calculate_metrics(y_test, y_pred, X_test)
        elif analysis == "Previsões para 2018":
            get_predictions_2018(X_test_original, y_pred)
       