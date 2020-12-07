import streamlit as st
import pandas as pd
import altair as alt

def present_conclusion():
    df = pd.read_csv("Conclusion/most_important_diseases.csv", index_col=0)
    df["Doenças"] = df["Doenças"].str.title().str.replace('_', ' ')

    st.markdown(
        """
        # Conclusões

        As diferentes análises realizadas ao longo do projeto produziram listas de doenças ranqueadas pela importância 
        relativa em termos de impacto nas taxas de suicídio. Para unificar as análises e oferecer um resultado final
        da pesquisa efetuada pelo grupo, foi feita a consolidação das listas através do seguinte procedimento:

        - Foram resgatadas as listas geradas:
            - Pela análise de Correlação de Spearman;
            - Pelo cálculo do Índice de *Bivariate Moran's I*;
            - Pelas análises SHAP dos modelos de Classificação e Regressão (utilizando apenas um classificador/regressor por modelo);
        - Foi aplicado um filtro para remover os atributos que não correspondem a doenças:
            - *Previous* (taxa de suicídios do ano anterior), no caso do modelo de Regressão; 
            -  Variáveis de Estado, nos modelos de Classificação e Regressão;
        - Realizou-se a ordenação das listas de forma decrescente, colocando no topo as doenças de maior impacto e utilizando apenas as 100 primeiras posições;
        -  Associou-se a cada elemento da lista um peso inversamente proporcional a sua posição na lista, de forma que o primeiro elemento possuísse peso 100, o segundo peso 99, e assim por diante;
        -  Foi feita a soma de pesos por doença;
        -  Foi ordenada a lista obtida, produzindo um ranqueamento final;

        O ranqueamento final pode ser observado abaixo:
        """
    )
    graph = alt.Chart(df.head(20), title="Doenças mais associadas com a ocorrência de suicídio no Brasil").mark_bar().encode(
        x=alt.X('Importância', title="Importância"),
        color=alt.Color('Doenças:N', legend=None),
        y=alt.Y('Doenças', title="Doenças", sort='-x'),
        tooltip=[alt.Tooltip('Doenças', title='Doenças'), alt.Tooltip('Importância', title='Importância')]
    )
    st.altair_chart((graph).properties(width=700, height=410))