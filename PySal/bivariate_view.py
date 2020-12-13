import streamlit as st
import numpy as np
import pandas as pd
from . import bivariate_data as dt
import pickle

def present_moran():
    st.markdown("""
        # Análises de Autocorrelação Espacial

        A metodologia da autocorrelação espacial foi adotada para verificar a presença (ou ausência) de variações espaciais na taxa de suicídios no território brasileiro.
    
    """)
    analysis = st.radio("Escolha um tipo de análise:",('Univariada', 'Bivariada'))
    if analysis == "Univariada":
        st.markdown("""
            ## **Análise Univariada**

            Inicialmente, foi realizada a análise de autocorrelação espacial univariada, utilizando apenas os dados referentes às taxas de suicídio.

            ### Moran's I

            Foi calculado o índice de *Moran's I* global com base no valor médio da taxa de suicídios em cada município entre 2008 e 2018, obtendo-se um valor de **0.461**, com um p-value de de $0.001$. 
            Esse índice corresponde a um **coeficiente de correlação** que mede a relação entre uma variável de interesse em determinado local (no caso, as taxas de suicídio) 
            e os valores dessa mesma variável na vizinhança da cidade considerada.

            Este cálculo é feito tomando o valor da variável dependente $x$ em uma região $r$, e calculando uma função de agregação de
            uma variável $y$ nas regiões vizinhas da região $r$. Essa função costuma ser a **média aritmética**, sendo chamada de $lag(y)$.
            Na análise em questão, tanto $x$ quando $y$ representam a **taxa de suicídios**.

            Após esse cálculo para cada região $r$ do mapa, calcula-se uma reta de regressão que passa pelos pontos $(x_r, 
            lag(y_r))$ de cada região. O coeficiente dessa reta é o **Moran's I**, e representa o quanto $lag(y_r)$ aumenta com $x_r$.

            ### LISA

            O cálculo do índice global fornece um panorama geral sobre a autocorrelação espacial da variável-alvo, 
            porém não detalha a localização de possíveis *clusters*. 
            Assim, essa análise foi decomposta em componentes, gerando uma medida localizada de autocorrelação conhecida
             como LISA (*Local Indicators of Spatial Association*).

            A coloração dos municípios é detalhada abaixo:
        """)

        st.markdown(
                        '<ul>'
                            '<li>HH (<i>High-High</i>): em <span style="color:red;"><b>vermelho</b></span>, representa um município com alta taxa de suicídio, cuja vizinhança também possui valores elevados.</li>'
                            '<li>HL (<i>High-Low</i>): em <span style="color:yellow;"><b>amarelo</b></span>, representa um município com alta taxa de suicídio, cuja vizinhança possui valores baixos.</li>'
                            '<li>LL (<i>Low-Low</i>): em <span style="color:blue;"><b>azul escuro</b></span>, representa um município com baixa taxa de suicídio, cuja vizinhança também possui valores baixos.</li>'
                            '<li>LH (<i>Low-High</i>): em <span style="color:#0099ff;"><b>azul claro</b></span>, representa um município com baixa taxa de suicídio, cuja vizinhança possui valores elevados.</li>'
                        '</ul>', unsafe_allow_html=True
                    )


        st.image("./PySal/img/moran_scatterplot.png",use_column_width=True)
        st.image("./PySal/img/lisa_cluster.png",use_column_width=True)
        
    else:
        st.markdown(
            """
            ## Análise Bivariada
            """

            """
            Para compreender as relações entre a distribuição espacial das taxas de suicídio e a de doenças nos municípios brasileiros, foi realizada uma análise bivariada.

            ### Bivariate Moran's I

            O Bivariate Moran's I é uma estatística calculada para medir a **correlação espacial** entre duas grandezas.

            Este cálculo é feito tomando o valor da variável dependente $x$ em uma região $r$, e calculando uma função de agregação de
            uma variável $y$ nas regiões vizinhas da região $r$. Essa função costuma ser a **média aritmética**, sendo chamada de $lag(y)$.
            Na análise em questão, $x$ corresponde à **taxa de suicídios por 100.000 habitantes**, e $y$ é uma **doença do DATASUS** a ser escolhida pelo usuário.
            """
        )

        disease_names = dt.get_diseases_select_names()
        options = np.append(['Selecione uma doença'], disease_names)
        selected_disease = st.selectbox('Selecione uma doença:', options)
        
        st.markdown(
            '<p>Para mais informações sobre uma doença, acesse o <a href="http://tabnet.datasus.gov.br/cgi/sih/mxcid10lm.htm", target="_blank">DATASUS</a>.</p>', unsafe_allow_html=True
        )

        if (selected_disease != 'Selecione uma doença'):
            disease_list_dic = dt.get_diseases_dic()
            selected_disease_csv_name = disease_list_dic[selected_disease]["CSV"]
            moran_bv_dic = pickle.load( open( "./PySal/moran_bv_dic.pkl", "rb" ) )
            moran_bv_i = moran_bv_dic[selected_disease_csv_name]
            st.write("**Bivariate Moran's I: **" + str(round(moran_bv_i, 3)))

            st.markdown(
                """
                ### Bivariate LISA

                Os municípios são coloridos de acordo com o grau de correlação espacial entre a variável escolhida e 
                a taxa de suicídios nos municípios adjacentes, para cada município do mapa, conforme explicado na legenda abaixo:

                """
            )

            st.markdown(
                '<ul>'
                    '<li>HH (<i>High-High</i>): em <span style="color:red;"><b>vermelho</b></span>, representa um município com alta taxa de suicídio, cuja vizinhança também possui uma taxa elevada da doença selectionada.</li>'
                    '<li>HL (<i>High-Low</i>): em <span style="color:yellow;"><b>amarelo</b></span>, representa um município com alta taxa de suicídio, cuja vizinhança possui uma taxa baixa da doença selectionada.</li>'
                    '<li>LL (<i>Low-Low</i>): em <span style="color:blue;"><b>azul escuro</b></span>, representa um município com baixa taxa de suicídio, cuja vizinhança também possui uma taxa baixa da doença selectionada.</li>'
                    '<li>LH (<i>Low-High</i>): em <span style="color:#0099ff;"><b>azul claro</b></span>, representa um município com baixa taxa de suicídio, cuja vizinhança possui uma taxa elevada da doença selectionada.</li>'
                '</ul>', unsafe_allow_html=True
            )

            st.image("./PySal/img/moran_loc_bv_" + selected_disease_csv_name + ".png",use_column_width=True)
            st.image("./PySal/img/lisa_cluster_" + selected_disease_csv_name + ".png",use_column_width=True)
           