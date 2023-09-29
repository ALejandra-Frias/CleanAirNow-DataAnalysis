#-----------------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------------#
#-----------------------------------------------------------------------------------#
import requests
import json
from io import StringIO

import pandas as pd
pd.set_option('display.precision',1)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',10)
#pd.set_option('display.width',1000)

import numpy as np
from datetime import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pickle



#-----------------------------------End Imports-------------------------------------#



urlhead='https://api.purpleair.com/v1/sensors/'


#1.GET METADATA
#Setup the request for the metadata in a lat/lon box
parameters = {
  'fields': 'sensor_index,primary_id_a,primary_key_a,longitude,latitude,date_created,name,location_type,altitude',
  'max_age': 0,
#-----Kansas City-----#
#  'nwlng': -94.9,
#  'nwlat': 39.3,
#  'selng': -94.3,
#  'selat': 38.8,
#-----Dallas/Fort Worth-----#
  'nwlng': -97.6,
  'nwlat': 33.6,
  'selng': -96.6,
  'selat': 32.2,
#-----LAWRENCE-----#
#  'nwlng': -95.4,
#  'nwlat': 39.1,
#  'selng': -95.1,
#  'selat': 38.9,
  'api_key': 'XXXX-XXXX-XXXX-XXXX-XXXX'#<------------------------------------CHANGE API KEY
}
response=requests.get(urlhead,params=parameters)
metadata=response.json()

#to see lat/lon locations
for stns in range(0,np.shape(response.json()['data'])[0]):
  plt.plot(response.json()['data'][stns][5],response.json()['data'][stns][4],'kx')
plt.show()


#------------------------------------------------------------------------Global Stuff

#apikey for url
apikey='&api_key=XXXX-XXXX-XXXX-XXXX-XXXX'#<------------------------------------CHANGE API KEY

#fields for url
#apifields='&fields=temperature'
#apifields='&fields=humidity,humidity_a,humidity_b,temperature,temperature_a,temperature_b,pressure,pressure_a,pressure_b'
apifields='&fields=humidity,humidity_a,humidity_b,temperature,temperature_a,temperature_b,pressure,pressure_a,pressure_b,pm2.5_alt,pm2.5_alt_a,pm2.5_alt_b,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b'

#get history for url (using hourly averages)
apihist='/history/csv?average=60'

#Select end time for all - maybe do every three months?
theend=pd.Timestamp(2023,8,15,0,0,0)#<------------------------------------------CHANGE!!!!!!!!!!!!!!!



#------------------------------------------Loop through each station to grab all data
for k in range(0,np.shape(metadata['data'])[0]):
  stn_id=str(metadata['data'][k][0])
  
  created_date=datetime.fromtimestamp(metadata['data'][k][1])
  thestart=pd.Timestamp(created_date.year,created_date.month,1,0,0,0)
  
  print(str(k)+': '+stn_id+' created on: '+str(thestart))
  
  #create date range to loop through, only accepts ~8000 records to query at a time
  #range always cuts off at larger values, so just keep to 5-Day windows...
  rng=pd.date_range(start=thestart,end=theend,freq='5D')
  
  #can only download little chunks at a time, so loop through time chunks
  first=1 #reset first to 1
  for i in range(0,len(rng)-1):
    dstart=rng[i]
    dend=rng[i+1]
    
    #print(f'{(i/len(rng)):.2f}')
    
    #convert to unix for header
    thestart_unix=str(int(time.mktime(dstart.timetuple())))
    theend_unix=str(int(time.mktime(dend.timetuple())))
    #put into text form for url
    urltimes='&start_timestamp='+thestart_unix+'&end_timestamp='+theend_unix
    
    #put all url text together
    api_url=urlhead+stn_id+apihist+apifields+apikey+urltimes
    response=requests.get(api_url)
    #print(response.text)
    df_temp=pd.read_csv(StringIO(response.text),sep=",",header=0)
    
    if first==1:
      #create new df
      all_df=df_temp
      first=0
    else:
      #append to the end
      all_df=all_df._append(df_temp)
  
  
  all_df['date']=pd.to_datetime(all_df['time_stamp'],unit='s')
  all_df=all_df.set_index('date')
  all_df=all_df.sort_index()
  
  print( k/np.shape(metadata['data'])[0] )
  
  #-------------------------------------Save Data-------------------------------------#
  t1=str(thestart.year)+f'{thestart.month:02d}'
  t2=str(theend.year)+f'{theend.month:02d}'
  
  var_list=[all_df,metadata]
  f=open('./raw_api_20230817/PA_DFW_'+stn_id.zfill(6)+'_'+t1+'_'+t2+'_v1.pkl','wb') #<---------CHANGE HERE
  pickle.dump(var_list,f)
  f.close()
  

#a little beep notification is nice
#import winsound
#winsound.Beep(300,2000)











