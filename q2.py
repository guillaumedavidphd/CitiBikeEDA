# coding: utf-8

import pandas as pd
import numpy as np
import io
import requests


filenames = []
with open('q2_url_datasets.txt', 'r') as f:
    for line in f:
        filenames.append(line.rstrip())


fraction_most_common = np.zeros(len(filenames))
for ff in np.arange(0, len(filenames)):
    print("File ", ff+1, '/', len(filenames))
    s = requests.get(filenames[ff]).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    fraction_most_common[ff] = df["Type_"].value_counts()[0]/len(df)
print("Fraction of the most common call type: ", np.mean(fraction_most_common))


f.index = np.arange(0, len(df))
df['TimeCreate'] = pd.to_datetime(df['TimeCreate'], format='%m/%d/%Y %I:%M:%S %p')


fraction_most_common = df["Type_"].value_counts()[0]/len(df)
print("Fraction of the most common call type: ", fraction_most_common)


df_time = df[["TimeDispatch", "TimeArrive", 'PoliceDistrict']]
df_time = df_time.dropna()
df_time.index = np.arange(0, len(df_time))


df_time.head(n=5)


df_time["TimeDispatch"] = pd.to_datetime(df_time["TimeDispatch"], format='%m/%d/%Y %I:%M:%S %p')
df_time['TimeArrive'] = pd.to_datetime(df_time['TimeArrive'], format='%m/%d/%Y %I:%M:%S %p')


df_time['ResponseTime'] = df_time["TimeArrive"] - df_time["TimeDispatch"]
df_time['ResponseTime[s]'] = df_time['ResponseTime'].astype('timedelta64[s]')
df_time = df_time[df_time['ResponseTime[s]']>=0]


df_time['ResponseTime[s]'].median()


sf_meanResponseTimePerDistrict = df_time.groupby(['PoliceDistrict']).mean()
sf_meanResponseTimePerDistrict.max() - sf_meanResponseTimePerDistrict.min()


df_2011 = df[df['TimeCreate']<pd.to_datetime('2012')]
df_2011 = df_2011['Type_']
df_2011.name = 'Type_2011'
df_2015 = df[df['TimeCreate']>=pd.to_datetime('2015')]
df_2015 = df_2015['Type_']
df_2015.name = 'Type_2015'


df_calltypeVolumes = pd.concat((df_2011.value_counts(), df_2015.value_counts()), axis=1)
df_calltypeVolumes = df_calltypeVolumes.dropna()


df_calltypeVolumes['PercentChange'] = (df_calltypeVolumes['Type_2011'] - df_calltypeVolumes['Type_2015'])/df_calltypeVolumes['Type_2011']
df_calltypeVolumes = df_calltypeVolumes[df_calltypeVolumes['PercentChange']>0]


df_calltypeVolumes[df_calltypeVolumes['PercentChange'] == df_calltypeVolumes['PercentChange'].max()]
