#-----------------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------------#
#-----------------------------------------------------------------------------------#
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
#-----------------------------------End Imports-------------------------------------#

#setup dataframe
thestart=pd.Timestamp(2018,1,1,0,0,0)
theend=pd.Timestamp(2023,8,1,0,0,0)

idx=pd.date_range(start=thestart,end=theend,freq='60min')
df=pd.DataFrame({'Date':idx})
df=df.set_index('Date')

files=glob.glob('C:/a_Projects/air_pollution/purple_air_get_auto/raw_api_20230817/*KC*')

for i in range(0,len(files)):
  f=open(files[i],'rb')
  var_list=pickle.load(f)
  f.close()
  df_temp=var_list[0]
  metadata=var_list[1]
  
  #subset and add suffix
  df_subset=df_temp[['temperature','humidity','pressure','pm2.5_alt',
                     'pm2.5_atm']].add_suffix('_'+files[i][-27:-21])
  
  
  #some weird duplications?
  #print('df[df.index.duplicated()])
  print('Original Shape: '+str(df_subset.shape)+'  Size of duplicates: '
        +str(df_subset[df_subset.index.duplicated()].shape))
  #remove duplicates
  df_subset=df_subset[~df_subset.index.duplicated(keep='first')]
  #print('New Size')
  #print(df_subset.shape)
  
  df=df.join(df_subset)
  #print(i/len(files))
  
  

#clean up names
thelist=list(df)
for match in thelist:
  if 'temperature' in match:
    df['tmpf'+match[11:]]=df[match]
    df=df.drop(columns=[match])
  if 'humidity' in match:
    df['relh'+match[8:]]=df[match]
    df=df.drop(columns=[match])
  if 'pressure' in match:
    df['pres'+match[8:]]=df[match]
    df=df.drop(columns=[match])
  
  if 'pm2.5_atm' in match:
    df['pm25atm_'+match[10:]]=df[match]
    df=df.drop(columns=[match])
  if 'pm2.5_alt' in match:
    df['pm25alt_'+match[10:]]=df[match]
    df=df.drop(columns=[match])
  
'''
metadata2=np.array(metadata['data'])
stn_id=metadata2[:,0].astype(float)
lats=metadata2[:,4].astype(float)
lons=metadata2[:,5].astype(float)
stn_name=metadata2[:,2].astype(str)
'''



var_list=[df,metadata]
f=open('./PA_KC_proc_full_v2.pkl','wb')
pickle.dump(var_list,f)
f.close()




