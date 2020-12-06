import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import streamlit as st
import altair as alt

path = "CSV/TabNet/Internacoes_Rate/"
corrs_df = pd.read_csv("Spearman/correlation.csv",index_col=0)

def get_top_correlations(ascending = False):
    corrs_df["Doenças"] = corrs_df["Doenças"].str.title().str.replace('_', ' ')
    return corrs_df.sort_values(by=['Correlação com suicidio'], ascending=ascending).reset_index(drop=True)

def plot_disease_vs_suicide(disease):
    csv_disease = mapper[mapper['SELECT_NAME'] == disease]['CSV'].values[0]
    suicide_df = pd.read_csv('CSV/Suicide/suicide_rates_08_18.csv', sep=',', index_col=0)
    suicide_df["SUICIDE"] = suicide_df.drop(columns="MUNCOD").sum(axis=1)/(len(suicide_df.columns) - 1)
    disease_df = pd.read_csv(path + csv_disease + '.csv', sep=',', index_col=0) 
    disease_df["Total"] = disease_df.drop(columns="MUNCOD").sum(axis=1)/(len(disease_df.columns) - 1)
    disease_df = disease_df[["MUNCOD", "Total"]]
    disease_df = disease_df[(disease_df["Total"] != 0)] # Exclude rows with 0 suicides
    final_df = pd.merge(disease_df, suicide_df, on="MUNCOD")
    base = alt.Chart(final_df, title="Suicídios vs " + disease + " (2008-2018)").mark_circle(color="black").encode(
            x=alt.X('Total', title="Taxa de Internações (" + disease + ")"),
            y=alt.Y("SUICIDE", title="Taxa de suicídios"),
            tooltip=[alt.Tooltip('SUICIDE', title='Taxa de suicídios'), alt.Tooltip('Total', title='Taxa de Internações')]
    )
    regression_line = base.transform_regression(
             "Total", "SUICIDE", method="poly", order=1).mark_line()

    st.altair_chart(alt.layer(base, regression_line).properties(width=700, height=410))

def get_diseases_select_names():
    global mapper
    diseases_files = glob.glob("diseases_select_list.csv")
    file_found = (len(diseases_files) > 0)

    if file_found:
        mapper = pd.read_csv('diseases_select_list.csv', index_col=0)
        print("Loaded preexisting diseases_select_list.csv")
        return np.array(mapper['SELECT_NAME'])
    else:
        print("File diseases_select_list.csv not found")
        return []