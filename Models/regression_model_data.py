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

        Na tabela abaixo, são mostradas as métricas de teste para o modelo.
        """)
    rmse = np.sqrt(metrics.mean_squared_error(y_pred,y_test))
    st.write("RMSE: {:.2f}".format(rmse))

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
    plot_df = final_df.drop(columns=["MUNCOD", "y_pred"]).T
    # plot_df = plot_df.reset_index()
    # plot_df.columns = ['Year', 'Real']
    plot_df.columns = ['Real']
    pred =final_df.drop(columns=["MUNCOD", "RATE_18"]).T
    plot_df["Pred"] = list(pred.iloc[:, 0])

    df.index.name = "x"
    data = df.reset_index().melt('x')
    st.write(data)
    graph = alt.Chart(data).mark_line().encode(
        x='x',
        y='value',
        color='variable'
    )

    st.altair_chart((graph).configure_view(strokeOpacity=0).configure_title(fontSize=12).properties(width=700, height=410))

    
    # scales = alt.selection_interval(bind='scales')
    # graph = alt.Chart(real_df, title="Evolução da taxa de suicídios (2008-2018").mark_line(point=True).encode(
    #     x=alt.X('Year', title='Ano'),
    #     y=alt.Y('Real', title='Taxa Real'),
    #     tooltip=[alt.Tooltip('Year', title='Ano'), alt.Tooltip('Real', title='Taxa Real')]
    # ).add_selection(scales).interactive()

    # st.altair_chart((graph).configure_view(strokeOpacity=0).configure_title(fontSize=12).properties(width=700, height=410))


    #st.write(real_df)
    #st.write(pred_df)
#     zipFileName = 'Maps/BRMUE250GC_SIR.7z'
#     if not os.path.isfile('Maps/BRMUE250GC_SIR.shp'):
#         print('Unzipping BRMUE250GC_SIR files')
#         with py7zr.SevenZipFile(zipFileName, 'r') as archive:
#             archive.extractall("Maps/")
    
#     gdf = gpd.read_file('Maps/BRMUE250GC_SIR.shp')
    
#     cadmun = pd.read_csv('./EDA/CADMUN.csv')
#     cadmun = cadmun[["MUNCOD", "MUNCODDV"]]
  
#     gdf["CD_GEOCMU"] = gdf["CD_GEOCMU"].astype(int)
#     gdf_city = pd.merge(gdf, cadmun, left_on="CD_GEOCMU", right_on="MUNCODDV", how="left")

#     result = pd.merge(gdf_city, df, left_on="MUNCOD", right_on="MUNCOD", how="left")
#     result = result[["NM_MUNICIP", "CD_GEOCMU", "geometry", "y_pred"]]
#     fig, ax = plt.subplots(figsize = (15,15))
#     result.plot(column="y_pred", ax=ax, legend=True,cmap='RdPu', scheme='user_defined', 
#         classification_kwds={'bins':[2, 5, 10, 50]},
#         missing_kwds={'color': 'lightgrey', "label": "Valores ausentes"})
#     plt.title('Taxa de suicídios no Brasil em 2018 (por 100 mil habitantes)',fontsize=25)
#     st.pyplot(fig)

#     result = pd.merge(gdf_city, df, left_on="MUNCOD", right_on="MUNCOD", how="left")
#     result = result[["NM_MUNICIP", "CD_GEOCMU", "geometry", "y_test"]]
#     fig, ax = plt.subplots(figsize = (15,15))
#     result.plot(column="y_test", ax=ax, legend=True,cmap='RdPu', scheme='user_defined', 
#         classification_kwds={'bins':[2, 5, 10, 50]},
#         missing_kwds={'color': 'lightgrey', "label": "Valores ausentes"})
#     plt.title('Taxa de suicídios no Brasil em 2018 (por 100 mil habitantes)',fontsize=25)
#     st.pyplot(fig)

def run_model(model):
    test_df = pd.read_csv("Models/test_data_regression.csv", index_col=0)
    X_test_original = test_df.drop(columns=["RATE"])
    X_test = X_test_original.copy()
    y_test = test_df["RATE"]
    features = joblib.load("Models/sav/selected_cor_features")
    sc_scaler_x = joblib.load("Models/sav/sc_x_regression.save")
    mm_scaler_x = joblib.load("Models/sav/mm_x_regression.save")
    sc_scaler_y = joblib.load("Models/sav/sc_y_regression.save")
    mm_scaler_y = joblib.load("Models/sav/mm_y_regression.save")
    filename = ""
    regressor = None

    if model != "Selecione um modelo":
        if model == "Adaboost":
            X_test = X_test[features]
            regressor = pickle.load(open("Models/sav/adaboost_regression.sav", 'rb'))
            y_pred = regressor.predict(X_test)
        elif model == "Gradient Boost":
            X_test = X_test[features]
            regressor = pickle.load(open("Models/sav/gradient_boost_regression.sav", 'rb'))
            y_pred = regressor.predict(X_test)
        elif model == "SVR":
            X_test = X_test[features]
            options_scaler = np.append(['MinMax'], ["Standard"])
            scaler = st.selectbox('Selecione um scaler:', options_scaler)
            if scaler == "Standard":
                X_test_transf = pd.DataFrame(sc_scaler_x.transform(X_test), index=X_test.index, columns=X_test.columns)
                regressor = pickle.load(open("Models/sav/svr_sc_regression.sav", 'rb')) 
                y_pred = regressor.predict(X_test_transf)
                y_pred = sc_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                y_pred = y_pred.ravel()
            else:
                X_test_transf = pd.DataFrame(mm_scaler_x.transform(X_test), index=X_test.index, columns=X_test.columns)
                regressor = pickle.load(open("Models/sav/svr_mm_regression.sav", 'rb'))
                y_pred = regressor.predict(X_test_transf)
                y_pred = mm_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                y_pred = y_pred.ravel()
        else:
            return
    if regressor != None:
        analysis = st.radio("Visualizar resultados:",('Métricas do modelo', 'Previsões para 2018',))
        if analysis == "Métricas do modelo":
            calculate_metrics(y_test, y_pred, X_test)
        elif analysis == "Previsões para 2018":
            get_predictions_2018(X_test_original, y_pred)
       