#-----------------------------------------------------------------------------#
#-----------------------------Import and stuff--------------------------------#
#-----------------------------------------------------------------------------#
import pickle
import glob
import sys
import numpy as np

import pandas as pd
pd.set_option('display.precision',1)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',10)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#----------------------------------End Imports--------------------------------#




#---------------------------------Get PA Data---------------------------------#
f=open('./PA_KC_proc_full_v2.pkl','rb')
var_list=pickle.load(f)
f.close()
del f
df=var_list[0]
metadata_purpleair=var_list[1]




#----------------------------------ADD EPA------------------------------------#
f=open('C:/a_Projects/air_pollution/epa_monitors/new_python_dl_20230810/KCnew_pm25_EPA.pkl','rb')
var_list=pickle.load(f)
f.close()

df_epa=var_list[0]
metadata_epa=var_list[1]

#uniform naming convention: pm25xxx_stnid
for i in range(len(df_epa.columns)):
  tempstr=df_epa.columns[i]
  newstr=tempstr[0:4]+'epa'+tempstr[4:]
  df_epa=df_epa.rename(columns={tempstr:newstr})

#add it to main dataframe
df=df.join(df_epa)




#---------------------------Add MKC/MCI ASOS----------------------------------#
#read MCI ASOS file
df_MCI=pd.read_csv('ASOS_MCI_2015_202308.csv')
#create a date object from time columns
df_MCI['date']=pd.to_datetime(df_MCI[['year','month','day','hour']])
#set the index as the date
df_MCI.set_index('date',inplace=True)
#join the ASOS dataframe with the Purple air dataframe
df=df.join(df_MCI)
#remove the old time columns
df=df.drop(columns=['year','month','day','hour'])

#read MKC ASOS file
df_MKC=pd.read_csv('ASOS_MKC_2015_202308.csv')
#create a date object from time columns
df_MKC['date']=pd.to_datetime(df_MKC[['year','month','day','hour']])
#set the index as the date
df_MKC.set_index('date',inplace=True)
#join the ASOS dataframe with the Purple air dataframe
df=df.join(df_MKC)
#remove the old time columns
df=df.drop(columns=['year','month','day','hour'])

metadata_asos={'MKC':{'lats':39.1229444,'lons':-94.5928333,'alt':230.7},
               'MCI':{'lats':39.2976111,'lons':-94.7138889,'alt':313.0}}





#---------------------------------ADD URBFRAC---------------------------------#
f=open('URBFRAC_KC_v0_1km.pkl','rb')    
var_list=pickle.load(f)
f.close()

urbfrac_lons=np.array(var_list[0])
urbfrac_lats=np.array(var_list[1])
urbfrac=np.array(var_list[2])
del f,var_list




#----------------------------------SAVE DATA----------------------------------#
var_list=[df,metadata_purpleair,metadata_epa,metadata_asos,urbfrac,urbfrac_lats,urbfrac_lons]
f=open('./KC_alldata_v2.pkl','wb')
pickle.dump(var_list,f)
f.close()














