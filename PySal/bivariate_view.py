import streamlit as st
import numpy as np
import pandas as pd
import time
from splot.esda import moran_scatterplot, lisa_cluster, plot_local_autocorrelation
from . import bivariate_data as dt
import matplotlib.pyplot as plt

def moran_scatterplt(moran, bivariate=False, disease=''):
    if bivariate:
        fig, ax = moran_scatterplot(moran, p=0.05)
        ax.set_title("Bivariate Moran's I")
        ax.set_ylabel('Spatial lag (' + disease + ')')
    else:
        fig, ax = moran_scatterplot(moran, aspect_equal=True)
        ax.set_title("Moran's I")
        ax.set_ylabel('Spatial lag de Suicídios')

    ax.set_xlabel('Suicídios')
    st.pyplot(fig)

def moran_map(moran, dataset, title):
    chart = lisa_cluster(moran, dataset, p=0.05, figsize=(9,9))
    fig = chart[0]
    plt.title("Mapa de clusters LISA (" + title + ")")
    st.pyplot(fig)

def present_moran():
    weights_file_found = dt.compute_weights()
    data = dt.get_dataset()

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


        if weights_file_found:
            moran_loc = dt.moran_local(data)
            moran_scatterplt(moran_loc, bivariate=False)
            moran_map(moran_loc, data, title="Suicídio")
            #fig, ax = plot_local_autocorrelation(moran_loc, data, 'AVG_SUICIDE_RATE')
            #st.pyplot(fig)
        else:
            st.markdown(
                """
                ### **_Erro ao processar fronteiras._**
                """
            )
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

        if weights_file_found:
            if len(disease_names) > 0:
                options = np.append(['Selecione uma doença'], disease_names)
                selected_disease = st.selectbox('Selecione uma doença:', options)

                st.markdown(
                    '<p>Para mais informações sobre uma doença, acesse o <a href="http://tabnet.datasus.gov.br/cgi/sih/mxcid10lm.htm", target="_blank">DATASUS</a>.</p>', unsafe_allow_html=True
                )

                if (selected_disease != 'Selecione uma doença'):
                    dt_disease = dt.get_disease_dataset(selected_disease)
                    dt_result = dt.merge_dataset_disease(data, dt_disease)
                    moran_bv = dt.moran_bv(dt_result)
                    st.write("**Bivariate Moran's I: **" + str(round(moran_bv.I, 3)))

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


                    moran_local_bv = dt.moran_local_bv(dt_result)
                    moran_scatterplt(moran_local_bv, bivariate=True, disease=selected_disease)
                    moran_map(moran_local_bv, dt_result, selected_disease)
            else:
                st.markdown(
                    """
                    ### **_Erro ao carregar nomes das doenças._**
                    """
                )
        else:
            st.markdown(
                """
                ### **_Erro ao processar fronteiras._**
                """
            )


        

        
        
