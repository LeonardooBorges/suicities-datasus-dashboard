import streamlit as st

def present_homepage():
    st.markdown(
        """
        # O impacto da incidência de doenças sobre a taxa de suicídio em cidades brasileiras: um estudo com variáveis do DATASUS
        """

        """
        ## Objetivo
        """

        """
        O objetivo do projeto é determinar a relação entre a incidência de determinadas doenças e 
        a ocorrência de suicídios nas cidades brasileiras, a partir de dados de internações extraídos do DATASUS.

        """

        """
        ## Sobre o Dashboard
        """

        """
        Este Dashboard é constituído de sete painéis distintos, além do atual, que sumarizam dados da análise do suicídio nos municípios brasileiros:
        """

        """
        - **Análise Exploratória de Dados:** Análise exploratória dos dados contidos nas Declarações de Óbito do DATASUS.

        - **Análise de Autocorrelação Espacial:** Análise espacial das taxas de suicídio e de doenças.

        - **Análise de Correlação de Spearman:** Correlação de Spearman entre as taxas de doenças e a ocorrência de suicídios.

        - **Determinação de Clusters de Suicídio:** Clusters de suicídio identificados pelo software *SaTScan*.

        - **Modelos Preditivos de Regressão:** Modelos de regressão para a previsão da evolução das taxas de suicídio nos municípios brasileiros.
        
        - **Modelos Preditivos de Classificação:** Modelos de classificação para a previsão de informações sobre as taxas de suicídio nos municípios brasileiros.
        
        - **Conclusões:** Ranqueamento final das doenças mais associadas à ocorrência de suicidio, a partir das análises anteriores.
        
        """

        """
        ## Autores
        """

        """
        Este projeto foi desenvolvido pelos alunos de Engenharia de Computação Quadrimestral (2020) da Escola Politécnica da USP:

        - Leonardo Borges Mafra Machado - 9345213

        - Marcos Paulo Pereira Moretti - 9345363

        - Paula Yumi Pasqualini - 9345280

        O projeto foi orientado pelo Professor Dr. Ricardo Luis de Azevedo da Rocha.
        """

        """
        ## Colaboradores
        """

        """
        Este projeto foi realizado em parceria com o C²D e o Itaú Unibanco.
        """
    )