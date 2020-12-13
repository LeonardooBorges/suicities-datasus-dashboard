import numpy as np
import pandas as pd

disease_list = pd.read_csv("diseases_select_list.csv", index_col=0)

def get_diseases_select_names():
  return np.array(disease_list['SELECT_NAME'])

def get_diseases_dic():
  return disease_list.set_index('SELECT_NAME').to_dict("index")