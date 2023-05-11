import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct



def read_data(data, cntry, yr):
    d1 = pd.read_csv(data, skiprows=4)
    d1.iloc[cntry, yr] 
    d1.drop(columns=['Country Code'], axis=1, inplace=True)
    d2 = d1.T
    print(d2.describe())

'''
def plot(data, kind, title, x, y):
    data.plot(kind=kind)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
    plt.show()
'''


year = [35, 40, 45, 50, 55, 60, 65]
country = [257, 208, 115, 180, 125, 61, 41, 83]



urb = pd.read_csv('API_SP.URB.TOTL_DS2_en_csv_v2_5359282.csv',country, year)

plt.scatter(urb.iloc[country], urb.iloc[year])
plt.title("Urban Population")
plt.xlabel("Country")
plt.ylabel("Years")
plt.show()
