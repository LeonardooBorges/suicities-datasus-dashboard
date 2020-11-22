import streamlit as st
from . import eda_data as dt
import numpy as np

def my_widget(key):
    st.subheader('Hello there!')
    clicked = st.button("Click me " + key)



def present_eda():
  st.markdown(
    """
    # Análise Exploratória de Dados
    """

    """
    Inicialmente, foi realizada uma análise exploratória dos dados contidos nas Declarações de Óbito do DATASUS.
    Foram selecionados os registros associados às mortes por suicídio, que correspondem aos códigos CID-10 entre X60-X84.

    As principais colunas foram extraídas, formando a tabela abaixo. A tabela original contém 62097 registros, sendo que
    aqui serão mostradas apenas as 100 primeiras entradas.

    *OBS: os valores com valor **nan** correspondem a entradas nulas.*
    """
  )

  suicide_df = dt.get_suicide_data()
  st.write(suicide_df.drop(columns=["YEAR", "CIRCOBITO"]).head(100))

  my_expander = st.beta_expander("Descrição das colunas:", expanded=False)
  with my_expander:
      st.markdown(
        """
        - DTOBITO: Data do óbito;

        - IDADE: Idade do falecido;

        - SEXO: Sexo do falecido (Ignorado/Masculino/Feminino);

        - RACACOR: Raça/cor do falecido (Branca/Preta/Amarela/Parda/Indígena);

        - ESTCIV: Estado civil do falecido (Solteiro/Casado/Viúvo/Separado judicialmente/União consensual/Ignorado);

        - ESC: Nível de escolaridade do falecido;

        - CODMUNRES: Município de residência do falecido;

        - CODMUNOCOR: Município de ocorrência do óbito;

        - LINHAII: Outras condições mórbidas pré-existentes e sem relação direta com a morte, segundo a classificação do CID-10;

        - CAUSABAS: Causa básica do óbito, segundo a classificação do CID-10;

        - OCUP: Ocupação do falecido, conforme a Classificação Brasileira de Ocupações;
        """
      )

  st.markdown(
    """
    ## Análises por coluna
    """
  )

  options = np.append(['Selecione uma análise'], ["Data de Óbito", "Causa Básica", "Linha II", "Município de Residência", "Município de Ocorrência","Idade", "Sexo", "Estado Civil", "Raça/Cor", "Ocupação", "Escolaridade"])
  column = st.selectbox('Selecione uma análise:', options)
  

  if column == "Data de Óbito":
    st.markdown(
      """
      Abaixo, pode-se observar o número de suicídios por data de óbito.
      """
    )
  elif column == "Causa Básica":
    st.markdown(
      """
      A coluna CAUSABAS da declaração de óbito corresponde à doença ou lesão que iniciou a cadeia de acontecimentos patológicos que conduziram diretamente à morte, ou as circunstâncias do acidente ou violência que produziram a lesão fatal.
      """
    )
  elif column == "Linha II":
    st.markdown(
      """
      A linha II da declaração de óbito corresponde às condições mórbidas pré-existentes no indivíduo e sem relação direta com sua morte.
      """
    )
  elif column == "Município de Residência":
    st.markdown(
      """
      O mapa abaixo mostra a taxa de suicídios por **município de residência** do indivíduo.
      """
    )
  elif column == "Município de Ocorrência":
    st.markdown(
      """
      O mapa abaixo mostra a taxa de suicídios por **município de ocorrência** da morte.
      """
    )
  elif column == "Idade":
    st.markdown(
      """
      Abaixo, observa-se a distribuição de suicídios por idade, sendo a média destacada em laranja.
      """
    )
  elif column == "Sexo":
    st.markdown(
      """
      O gráfico abaixo mostra a evolução da quantidade suicídios por sexo, no período de 2008 a 2018.
      """
    )
  dt.plot_column(column)