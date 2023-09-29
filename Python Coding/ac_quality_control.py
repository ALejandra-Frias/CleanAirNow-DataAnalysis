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



#---------------------------------Get All Data---------------------------------#
f=open('./KC_alldata_v2.pkl','rb')
var_list=pickle.load(f)
f.close()
del f

df=var_list[0]
metadata_purpleair=var_list[1]
metadata_epa=var_list[2]
metadata_asos=var_list[3]
urbfrac=var_list[4]
urbfrac_lats=var_list[5]
urbfrac_lons=var_list[6]




#-----------------------------------------------------------------------------#
#--------------------------------PURPLE AIR QC--------------------------------#
#-----------------------------------------------------------------------------#



#-----------MANUALLY FIND AND REMOVE BAD STATIONS ENTIRELY--------------------#

#NOTE: ALT better than ATM

#mainly inside monitors
#bad_list=['016889','024105','044921','150644','150648','059017',
#longterm sensor, but pm25 is messed up?
#          '009738',
#no/few data (just installed) <-----------revise with updated data later
#          '109006','127621','127641','128853','128859','128867','150418','150612','164635',
#          '181043','181085','181201']

#mainly inside monitors
bad_list_pm25=['016889','024105','044921','150644','150648','059017',
#longterm sensor, but pm25 is messed up?
          '009738']

#remove all data
for match in bad_list_pm25:
  df=df.drop(columns='pm25alt_'+match)
  df=df.drop(columns='pm25atm_'+match)
  
#Check time series manually for bad data
'''
thelist=list(df)
tmpf_list=[]
for match in thelist:
  if 'pm25alt' in match:
    plt.plot(df[match],'k.')
    plt.title(match)
    plt.show()
sys.exit()
'''

#---TMPF REMOVE---#
#mainly inside monitors
bad_list_tmpf=['016889','024105','044921','150644','150648','059017']

#remove all tmpf data
for match in bad_list_tmpf:
  df=df.drop(columns='tmpf_'+match)



#-----Simple Temperature QC-----#
thelist=list(df)
for match in thelist:
  if 'tmpf' in match:
    df[match][df[match]>130]=np.nan
    df[match][df[match]<-100]=np.nan

'''
#Check time series
thelist=list(df)
for match in thelist:
  if 'tmpf' in match:
    plt.plot(df[match],'k.')
    plt.title(match)
    plt.show()

thelist=list(df)
for match in thelist:
  if 'pm25alt' in match:
    plt.plot(df[match],'k.')
    plt.title(match)
    plt.show()
'''





#----------------remove testing or other weirdness----------------------------#

#Manual QC Cleanup for odd times, not finished...
'''
df['tmpf_007596']['2018-01-03 00:00:00':'2018-02-15 00:00:00']=np.nan #few start
df['tmpf_009738']['2018-04-01 00:00:00':'2018-05-01 00:00:00']=np.nan #few start
df['tmpf_054769']['2020-05-01 00:00:00':'2020-06-01 00:00:00']=np.nan #few start
df['tmpf_058911']['2020-07-01 00:00:00':'2020-08-01 00:00:00']=np.nan #few start
df['tmpf_058937']['2020-07-01 00:00:00':'2020-08-01 00:00:00']=np.nan #few start
df['tmpf_058973']['2020-07-01 00:00:00':'2020-08-01 00:00:00']=np.nan #few start
'''

#-----------------------------------------------------------------------------#









#-----------------------------------------------------------------------------#
#----------------------------------SAVE DATA----------------------------------#
#-----------------------------------------------------------------------------#

var_list=[df,metadata_purpleair,metadata_epa,metadata_asos,urbfrac,urbfrac_lats,urbfrac_lons]
f=open('./KC_alldata_QC_v2.pkl','wb')
pickle.dump(var_list,f)
f.close()
















