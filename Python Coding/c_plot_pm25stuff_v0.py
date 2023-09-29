#-----------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------#
#-----------------------------------------------------------------------------#
import numpy as np
import pickle
import pandas as pd
pd.set_option('display.precision',1)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',10)
#pd.set_option('display.width',1000)

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

import matplotlib.dates as mdates

#import matplotlib as mpl
#mpl.use('svg')
#new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
#mpl.rcParams.update(new_rc_params)

from cartopy.feature import NaturalEarthFeature
import cartopy.crs as crs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature

import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy import stats

from sklearn import linear_model

import warnings
warnings.filterwarnings("ignore")

import sys
#-----------------------------------End Imports-------------------------------------#



#-----------------------------------------------------------------------------------#
#------------------------------------Get Data---------------------------------------#
#-----------------------------------------------------------------------------------#

f=open('./KC_alldata_QC_v2.pkl','rb')
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


#FIND AND REMOVE SPARSE DATA
thelist=list(df)
for match in thelist:
  if 'pm25' in match:
    count=df[match].count()
    #print(count)
    if count<3000:
      #print(count)
      df=df.drop(columns=[match])




'''
df_pm25_only=df[pm25_list+pm25_list_epa]
#rename with stn_name
cnames=df_pm25_only.columns
for match in cnames:
  theid=float(match[8:])
  thename=stn_name[np.where(stn_id==theid)]
  if thename.size>0:
    df_pm25_only[thename[0][:23]]=df_pm25_only[match]
    df_pm25_only=df_pm25_only.drop(columns=match)
    #print(thename)
'''


thelist=list(df)
pm25_list=[]
for match in thelist:
  if 'pm25alt' in match:
    pm25_list.append(match)
    
#identify PA vs epa sensors
thelist=list(df)
pm25_list_PA=[]
for match in thelist:
  if 'pm25alt' in match:
    pm25_list_PA.append(match)
pm25_list_epa=[]
for match in thelist:
  if 'pm25epa' in match:
    pm25_list_epa.append(match)




'''
#---Gantt-like chart for data availability---#
fig=plt.figure(figsize=(7,8),dpi=600)
labels=[]
nn=0
for match in pm25_list:
  plt.plot(1+nn*df[match]/df[match],'k.')  #yes/no for data
  labels.append(match[8:])
  nn=nn+1

plt.plot([mdates.datetime.datetime(2020,1,1),mdates.datetime.datetime(2020,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2021,1,1),mdates.datetime.datetime(2021,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2022,1,1),mdates.datetime.datetime(2022,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2023,1,1),mdates.datetime.datetime(2023,1,1)],[-2,nn+2],'k-')

plt.xlim(mdates.datetime.datetime(2020,1,1),mdates.datetime.datetime(2023,8,1))
plt.ylim([0.5,nn+1])
plt.yticks(np.arange(1,nn+1),labels,fontsize=8)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=()))
#plt.xticks(rotation=-20)
fig.savefig('./figs/gantt_pm25.png',format='png',dpi=600) #-------------------------FIGURE: Gantt



#---Gantt-like chart for data availability---# WITH STR NAMES
fig=plt.figure(figsize=(7,8),dpi=600)
labels=[]
nn=0
for match in pm25_list:
  plt.plot(1+nn*df[match]/df[match],'k.')  #yes/no for data
  thestn=match[5:]
  #find corresponding match in metadata
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      #print(temp[xxx])
      labels.append(temp[xxx][2])
  nn=nn+1

plt.plot([mdates.datetime.datetime(2020,1,1),mdates.datetime.datetime(2020,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2021,1,1),mdates.datetime.datetime(2021,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2022,1,1),mdates.datetime.datetime(2022,1,1)],[-2,nn+2],'k-')
plt.plot([mdates.datetime.datetime(2023,1,1),mdates.datetime.datetime(2023,1,1)],[-2,nn+2],'k-')

plt.xlim(mdates.datetime.datetime(2020,1,1),mdates.datetime.datetime(2023,8,15))
plt.ylim([0.5,nn+1])
plt.yticks(np.arange(1,nn+1),labels,fontsize=8)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=()))
#plt.xticks(rotation=-20)
fig.savefig('./figs/gantt_pm25.png',format='png',dpi=600) #-------------------------FIGURE: Gantt
'''



#---Time Series of active sensors---#
fig=plt.figure(figsize=(5,3.6),dpi=600)
df_temp=df[pm25_list]
#plt.plot(df_temp.count(axis=1))
df_temp=df_temp.groupby(df.index.date).mean()
plt.plot(df_temp.count(axis=1),'k-')
plt.ylabel('Number of active stations')
plt.show()
#fig.savefig('./figs/active_stn_number.png',format='png',dpi=600) #-------------FIGURE: # of stations
fig.savefig('./figs/active_stn_number.svg',format='svg',dpi=600) #-------------FIGURE: # of stations






#-----------------------------------------------------------------------------#
#-------------------------------Spatial Maps----------------------------------#
#-----------------------------------------------------------------------------#



#GET CARTOPY DATA
reader=shpreader.Reader('C:/a_Projects/zMASTER_PYTHON/tl_2016_us_primaryroads/tl_2016_us_primaryroads.shp')
names=[]
geoms=[]
for rec in reader.records():
    if (rec.attributes['FULLNAME'][0]=='I'):
        names.append(rec)
        geoms.append(rec.geometry)

#make interstate feature
roads=ShapelyFeature(geoms,crs.PlateCarree(),edgecolor='darkblue',lw=0.5,facecolor='none')
#States and coastlines
states=NaturalEarthFeature(category='cultural',scale='50m',facecolor='none',name='admin_1_states_provinces_shp')


#------------------------------Urban Frac Maps--------------------------------#


#identify PA vs epa sensors
thelist=list(df)
pm25_list_PA=[]
for match in thelist:
  if 'pm25alt' in match:
    pm25_list_PA.append(match)
pm25_list_epa=[]
for match in thelist:
  if 'pm25epa' in match:
    pm25_list_epa.append(match)

pm25_list_both=pm25_list_PA+pm25_list_epa


thecmap='gist_yarg'
thecmap='summer_r'

fig=plt.figure(figsize=(5,3.6),dpi=600)
ax=plt.axes(projection=crs.PlateCarree())
ax.add_feature(roads,linewidth=1.25,edgecolor='black')
plt.pcolor(urbfrac_lons,urbfrac_lats,urbfrac,cmap=thecmap)

for match in pm25_list_both:
#find corresponding match in metadata
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      #plt.text(temp[xxx][5],temp[xxx][4],temp[xxx][2])
      plt.plot(temp[xxx][5],temp[xxx][4],'mo',markersize=5,markeredgecolor='w')
  for xxx in range(len(metadata_epa)):    #epa
    if int(metadata_epa[xxx]['stn_id'])==int(match[8:]):
      plt.plot(metadata_epa[xxx]['longitude'],metadata_epa[xxx]['latitude'],'kx',markersize=5)
      plt.text(metadata_epa[xxx]['longitude'],metadata_epa[xxx]['latitude'],metadata_epa[xxx]['stn_id'],color='k',ma='center',fontsize=7)

plt.xlim([-94.9,-94.25])
plt.ylim([38.75,39.35])
plt.clim(0,1.0)
plt.colorbar()
plt.show()
#fig.savefig('./figs/urbfrac_stns.png',format='png',dpi=600) #------------------FIG: Urbfrac+Stns
#fig.savefig('./figs/urbfrac_stns.svg',format='svg',dpi=600) #------------------FIG: Urbfrac+Stns





'''
#station urbfrac histogram
fig=plt.figure(figsize=(5,3.6),dpi=600)
plt.hist(urbfrac_mean,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],rwidth=0.95)
plt.xlim([-0.005,1.0])
plt.xlabel('Urban Fraction')
plt.ylabel('Number of Stations')
plt.show()
#fig.savefig('./figs/hist_urbfrac.png',format='png',dpi=600) #-------------FIG: Urbfrac Stn Histogram
#fig.savefig('./figs/hist_urbfrac.svg',format='svg',dpi=600) #-------------FIG: Urbfrac Stn Histogram
'''





#Alternative urb fractions
'''
thecmap='YlOrRd'
cmap=get_cmap('YlOrRd')
cmap_21=cmap(np.linspace(0,1,101))
cmap_21[0][0:3]=[1,1,1]
thecmap=ListedColormap(cmap_21)
'''

'''
thecmap='gist_yarg'
fig=plt.figure(figsize=(4,3.6),dpi=300)
#ax=plt.axes(projection=crs.PlateCarree())
#ax.add_feature(roads,linewidth=1.25,edgecolor='black')
plt.pcolor(lons_urbfrac,lats_urbfrac,urbfrac,cmap=thecmap)
for i in range(len(lats)):
  if urbfrac_median[i]>0.4:
    plt.scatter(lons[i],lats[i],c=urbfrac_median[i],cmap=thecmap,zorder=10+100*urbfrac_median[i],edgecolors='w',linewidths=0.5)
  if urbfrac_median[i]<=0.4:
    plt.scatter(lons[i],lats[i],c=urbfrac_median[i],cmap=thecmap,zorder=10+100*urbfrac_median[i],edgecolors='k',linewidths=0.5)
  plt.clim(0,0.7)

plt.xlim([-94.9,-94.25])
plt.ylim([38.75,39.35])
plt.clim(0,0.8)
plt.colorbar()
plt.show()
#fig.savefig('./figs/urbfrac_stns.png',format='png',dpi=600) #------------------FIG: Urbfrac+Stns
fig.savefig('./figs/urbfrac_stns.svg',format='svg',dpi=800) #------------------FIG: Urbfrac+Stns
'''




#-------------------------------Time series, bite sized chunks-----------------#
'''
from matplotlib.patches import Rectangle

myFmt=mdates.DateFormatter('%d-%b-%Y')

#Plot all time series
rng=pd.date_range(start=pd.Timestamp(2018,4,1,0,0,0),end=pd.Timestamp(2023,6,1,0,0,0),freq='3D')
for i in range(0,len(rng)-1):
  dstart=rng[i]
  dend=rng[i+1]
  
  lines,labels=[],[]
  
  fig=plt.figure(figsize=(5,3.6),dpi=200)
  for match in pm25_list_PA:
    if ~np.isnan(np.array(df[match][dstart:dend])).all():
      lines+=plt.plot(df[match][dstart:dend],'-',color='m',zorder=100)
      #labels.append(match)
      #--label with str name
      thestn=match[8:]
      temp=metadata_purpleair['data']
      for xxx in range(len(temp)):
        if temp[xxx][0]==int(match[8:]):
          labels.append(temp[xxx][2])
  
  for match in pm25_list_epa:
    if ~np.isnan(np.array(df[match][dstart:dend])).all():
      lines+=plt.plot(df[match][dstart:dend],'k-')
      labels.append(match)
  
  width=dend-dstart
  plt.gca().add_patch(Rectangle((dstart,0),width,12,color='limegreen',alpha=0.9,linewidth=0,zorder=0))
  plt.gca().add_patch(Rectangle((dstart,12),width,35,color='yellow',alpha=0.9,linewidth=0,zorder=0))
  plt.gca().add_patch(Rectangle((dstart,35),width,55,color='orange',alpha=0.9,linewidth=0,zorder=0))
  plt.gca().add_patch(Rectangle((dstart,55),width,150,color='red',alpha=0.9,linewidth=0,zorder=0))

  
  plt.xlim(dstart,dend)
  plt.ylim(0,80)
  #plt.ylim(0,np.nanquantile(temp_data,.99)+10)
  plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
  plt.gca().xaxis.set_major_formatter(myFmt)
  #fig.autofmt_xdate()
  
  plt.legend(lines,labels,ncol=3,loc=2,prop={'size': 5})
  plt.show()

'''





#--------Descriptive stats
fig=plt.figure(figsize=(8,4),dpi=600)

df_pm25_only=df[pm25_list_PA+pm25_list_epa]
#remove <0...
df_pm25_only[df_pm25_only<0]=0


#rename with stn_name
cnames=df_pm25_only.columns
for match in cnames:
  #--label with str name
  thestn=match[8:]
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      df_pm25_only[temp[xxx][2][:23]]=df_pm25_only[match]
      df_pm25_only=df_pm25_only.drop(columns=match)

df_pm25_only.boxplot(showfliers=False,grid=False,whis=[10,90])
plt.xticks(rotation=90)
plt.ylabel('pm2.5')
plt.show()




#--------Descriptive stats with names
fig=plt.figure(figsize=(8,4),dpi=600)
df_pm25_only=df[pm25_list_PA+pm25_list_epa]
#remove <0...
df_pm25_only[df_pm25_only<0]=0

#rename with stn_name
cnames=df_pm25_only.columns
for match in cnames:
  #--label with str name
  thestn=match[8:]
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      df_pm25_only[temp[xxx][2][:23]]=df_pm25_only[match]
      df_pm25_only=df_pm25_only.drop(columns=match)

df_pm25_only.boxplot(showfliers=False,grid=False)
plt.xticks(rotation=90)
plt.ylabel('pm2.5')
plt.show()




#--------Descriptive stats with names --sorted by median
'''
fig=plt.figure(figsize=(8,4),dpi=600)
df_pm25_only=df[pm25_list_PA+pm25_list_epa]
#remove <0...
df_pm25_only[df_pm25_only<0]=0

grouped=df_pm25_only.median().sort_values()
sns.boxplot(df_pm25_only,order=grouped.index)

#rename with stn_name
cnames=df_pm25_only.columns
for match in cnames:
  #--label with str name
  thestn=match[8:]
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      df_pm25_only[temp[xxx][2][:23]]=df_pm25_only[match]
      df_pm25_only=df_pm25_only.drop(columns=match)

df_pm25_only.boxplot(showfliers=False,grid=False)
plt.xticks(rotation=90)
plt.ylabel('pm2.5')
plt.show()
'''



#####Monthly
fig,ax=plt.subplots()
ax.plot(df_pm25_only.groupby(df_pm25_only.index.month).mean(),color='m',marker='.')
ax.plot(df_pm25_only[pm25_list_epa].groupby(df_pm25_only[pm25_list_epa].index.month).mean(),color='k',marker='.')
ax.set_xlabel('Month',fontsize=14)
ax.set_ylabel('PM2.5 (ug/m3)',fontsize=14)
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
ax.set_ylim([0,20])
#ax.tick_params(axis='y', colors='blue')




for ss in range(1,5):
  #ss=4 #<-----------------------------------------------------------Select Season
  if ss==1:
    m1,m2,m3,season=12,1,2,'Winter'
  elif ss==2:
    m1,m2,m3,season=3,4,5,'Spring'
  elif ss==3:
    m1,m2,m3,season=6,7,8,'Summer'
  elif ss==4:
    m1,m2,m3,season=9,10,11,'Fall'  

  df_subset=df_pm25_only[(df_pm25_only.index.month==m1) | (df_pm25_only.index.month==m2) | (df_pm25_only.index.month==m3)]
  season_pm25=df_subset.groupby(df_subset.index.hour).mean()
  
  df_subset_epa=df_pm25_only[pm25_list_epa][(df_pm25_only[pm25_list_epa].index.month==m1) | (df_pm25_only[pm25_list_epa].index.month==m2) | (df_pm25_only[pm25_list_epa].index.month==m3)]
  season_pm25_epa=df_subset_epa.groupby(df_subset_epa.index.hour).mean()
  
  
  fig,ax=plt.subplots()
  ax.plot(season_pm25,color='m',marker='.')
  ax.plot(season_pm25_epa,color='k',marker='.')
  ax.set_xlabel('Hour (UTC)',fontsize=14)
  ax.set_ylabel('PM2.5 (ug/m3)',fontsize=14)
  ax.set_xticks([0,3,6,9,12,15,18,21])
  ax.set_ylim([0,20])
  ax.set_title('Average diurnal PM2.5 during '+season)
  plt.show()









#-----------------------------------circle plots-------------------------------#

#reset pm25 df
df_pm25_only=df[pm25_list_PA+pm25_list_epa]
thelist=list(df_pm25_only)
#remove <0...
df_pm25_only[df_pm25_only<0]=0

import math

#add wind stuff
df_pm25_only['wspd']=(df['uwnd_MKC']**2+df['vwnd_MKC']**2)**0.5
df_pm25_only['wdir']=(270-(np.arctan2(df['vwnd_MKC'],df['uwnd_MKC'])*180/math.pi))%360
df_pm25_only['wdir_rad']=np.deg2rad(df_pm25_only['wdir'])


#abins=np.linspace(0,2*np.pi,17)
abins=np.linspace(0,2*np.pi,13)
#rbins=np.linspace(0,12,5)
rbins=np.linspace(0,12,4)
A,R=np.meshgrid(abins, rbins)
zval=np.zeros((len(rbins)-1,len(abins)-1))


#subsets
minhr=6
maxhr=18
m1=3
m2=4
m3=5

#stn_list=['pm25_202090021','pm25_200910010','pm25_290470005','pm25_290370003','pm25_290950042']
#stn_list=thelist[:-3]

for x in thelist:
  for i in range(0,len(rbins)-1):
    for j in range(0,len(abins)-1):
      zval[i,j]=df_pm25_only[x][(~df_pm25_only[x].isna()) &
                                 (df_pm25_only.index.hour>=minhr) & (df_pm25_only.index.hour<=maxhr) &
                                 ((df_pm25_only.index.month==m1) | (df_pm25_only.index.month==m2) | (df_pm25_only.index.month==m3)) &
                                 (df_pm25_only['wspd']>=rbins[i]) &
                                 (df_pm25_only['wspd']< rbins[i+1]) &
                                 (df_pm25_only['wdir_rad']>=abins[j]) &
                                 (df_pm25_only['wdir_rad']< abins[j+1])].mean()
  
  fig,ax=plt.subplots(subplot_kw=dict(projection="polar"))
  pc=ax.pcolormesh(A,R,zval,cmap="coolwarm")
  pc=ax.pcolormesh(A,R,zval,cmap="coolwarm",clim=(0,14))
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)
  #fig.colorbar(pc)
  cbar=fig.colorbar(pc)
  #cbar = plt.colorbar(heatmap)
  #cbar.ax.set_yticklabels(['0','1','2','>3'])
  cbar.set_label('PM 2.5 (µg/m³)', rotation=270)
  cbar.ax.get_yaxis().labelpad=12
  
  #get name for title
  titlename=x
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(x[8:]):
      titlename=temp[xxx][2]
  
  plt.title('Spring midnight-morning at \n '+titlename)
  plt.show()





sys.exit()







#--------------------------------SNS plots------------------------------------#


#------------Correlation Heatmap--------------#
import seaborn as sns

#reset df again
df_pm25_only=df[pm25_list_PA+pm25_list_epa]
#remove <0...
df_pm25_only[df_pm25_only<0]=0

#rename with stn_name
cnames=df_pm25_only.columns
for match in cnames:
  #--label with str name
  thestn=match[8:]
  temp=metadata_purpleair['data']
  for xxx in range(len(temp)):
    if temp[xxx][0]==int(match[8:]):
      df_pm25_only[temp[xxx][2][:23]]=df_pm25_only[match]
      df_pm25_only=df_pm25_only.drop(columns=match)


#use all
#df_pm25_only_clean=df_pm25_only
#remove stations with few obs
df_pm25_only_clean=df_pm25_only.drop(columns=['Crimson Ridge','ROCKHILL AND EMMANUEAL ',
                                              'CleanAirNow/Paco','CleanAirNow/Webster',])


fig=plt.figure(figsize=(20,16),dpi=600)

cmap=get_cmap('brg')
cmap_new=cmap(np.linspace(0,1,10))
#cmap_new[0][0:3]=[1,1,1]
thecmap=ListedColormap(cmap_new)

myColors=((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
#cmap=LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax=sns.heatmap(df_pm25_only_clean.corr(),annot=True,cmap=thecmap,vmin=0,vmax=1)

colorbar=ax.collections[0].colorbar
colorbar.set_ticks([0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
#colorbar.set_ticklabels(['B', 'A', 'C'])
plt.show()




#---------------GIANT Paired-Plots-------------#

g=sns.PairGrid(df_pm25_only_clean)
g.map_diag(sns.kdeplot)

import time

start_time=time.time()
#g.map_lower(sns.regplot,scatter=False)
g.map_upper(sns.kdeplot,cmap='PuRd_r',shade=True,bw_adjust=2)
print(time.time()-start_time)

start_time=time.time()
g.map_upper(sns.regplot,scatter=False,x_ci='None',robust=True)
print(time.time()-start_time)

#g.map_upper(sns.kdeplot,clip=[(0,40),(0,40)],cmap="PuRd_d",shade=True)

for i in range(df_pm25_only_clean.shape[1]):
  for j in range(df_pm25_only_clean.shape[1]):
    if i!=j:
      #print('a:',i,'  b:',j)
      g.axes[i,j].set_xlim((0,40))
      g.axes[i,j].set_ylim((0,40))

for i in range(df_pm25_only_clean.shape[1]):
  for j in range(df_pm25_only_clean.shape[1]):
    if i<j:
      g.axes[i,j].plot([0,40],[0,40],'k-')

plt.show()


#sys.exit()


#---------------Test small Paired-Plots-------------#
'''
#remove few obs
df_pm25_only_clean=df_pm25_only.drop(columns=['Crimson Ridge','MandMsouth40','SA3','Royall S',
                                        'FH410', 'SA5', 'SA1', 'AlexisHouse','Spencer 233',
                                        'pm25_200910007'])

df_mini=df_pm25_only[['pm25_200910007','pm25_200910010','CleanAirNow/Brown',
                      'CleanAirNow/StrawberryH']]

g=sns.PairGrid(df_mini)
g.map_diag(sns.kdeplot)

import time

start_time=time.time()
#g.map_lower(sns.regplot,scatter=False)
g.map_upper(sns.kdeplot,cmap='PuRd_r',shade=True,bw_adjust=2)
print(time.time()-start_time)

start_time=time.time()
g.map_upper(sns.regplot,scatter=False,x_ci='None',robust=True)
print(time.time()-start_time)

#g.map_upper(sns.kdeplot,clip=[(0,40),(0,40)],cmap="PuRd_d",shade=True)

for i in range(df_mini.shape[1]):
  for j in range(df_mini.shape[1]):
    if i!=j:
      #print('a:',i,'  b:',j)
      g.axes[i,j].set_xlim((0,40))
      g.axes[i,j].set_ylim((0,40))

for i in range(df_mini.shape[1]):
  for j in range(df_mini.shape[1]):
    if i<j:
      g.axes[i,j].plot([0,40],[0,40],'k-')

plt.show()
'''


#----------------------------------individual plots--------------------#
#remove few obs
#df_pm25_only_clean=df_pm25_only.drop(columns=['Crimson Ridge','MandMsouth40','SA3','Royall S',
#                                        'FH410', 'SA5', 'SA1', 'AlexisHouse','Spencer 233',
#                                        'pm25_200910007'])


'''
for i in range(df_pm25_only_clean.shape[1]):
  for j in range(df_pm25_only_clean.shape[1]):
    if i<j:
      sns.regplot(data=df_pm25_only_clean,x=df_pm25_only_clean.columns[i],y=df_pm25_only_clean.columns[j],robust=False,
                  marker='.',scatter_kws={'color':'blue'},line_kws={'color':'black'})
      plt.plot([0,40],[0,40],'r--',zorder=100)
      #plt.xlim(0,40)
      #plt.ylim(0,40)
      plt.axis('square')
      plt.show()
'''





#----------------------------------individual plots--------------------#
#remove few obs
df_pm25_only_clean_QCzero=df_pm25_only_clean
df_pm25_only_clean_QCzero[df_pm25_only_clean_QCzero<0]=0

#sns.regplot(data=df_pm25_only_clean,x='pm25_290470005',y='Spencer Chemical Storer',robust=False,
#            marker='.',scatter_kws={'color':'blue'},line_kws={'color':'black'})
'''
sns.kdeplot(data=df_pm25_only_clean_QCzero,y='pm25_290470005',x='Spencer Chemical Storer',
            cmap='binary',shade=True,bw_adjust=1,zorder=10,alpha=0.5)

sns.regplot(data=df_pm25_only_clean_QCzero,y='pm25_290470005',x='Spencer Chemical Storer',
            x_ci='None',robust=False,
            marker='.',scatter_kws={'color':'k','zorder':0,'s':3},
            line_kws={'color':'black','zorder':100})

plt.plot([0,40],[0,40],'r--',zorder=100)
plt.axis('square')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.show()
'''


for i in range(df_pm25_only_clean_QCzero.shape[1]):
  for j in range(df_pm25_only_clean_QCzero.shape[1]):
    if i>j:
      sns.kdeplot(data=df_pm25_only_clean_QCzero,x=df_pm25_only_clean_QCzero.columns[i],y=df_pm25_only_clean_QCzero.columns[j],
                  cmap='binary',shade=True,bw_adjust=1,zorder=10,alpha=0.5)
      
      sns.regplot(data=df_pm25_only_clean_QCzero,x=df_pm25_only_clean_QCzero.columns[i],y=df_pm25_only_clean_QCzero.columns[j],
                  x_ci='None',robust=False,
                  marker='.',scatter_kws={'color':'k','zorder':0,'s':3},
                  line_kws={'color':'black','zorder':100})
      
      #plt.plot([0,40],[0,40],'r--',zorder=100)
      plt.axis('square')
      plt.plot([0,plt.gca().get_ylim()[1]],[0,plt.gca().get_ylim()[1]],'r--',zorder=100)
      plt.ylim(bottom=0)
      plt.xlim(left=0)
      plt.show()



'''
for i in range(df_pm25_only_clean.shape[1]):
  for j in range(df_pm25_only_clean.shape[1]):
    if i<j:
      sns.regplot(data=df_pm25_only_clean,x=df_pm25_only_clean.columns[i],y=df_pm25_only_clean.columns[j],robust=False,
                  marker='.',scatter_kws={'color':'blue'},line_kws={'color':'black'})
      plt.plot([0,40],[0,40],'r--',zorder=100)
      #plt.xlim(0,40)
      #plt.ylim(0,40)
      plt.axis('square')
      plt.show()
'''

sys.exit()

























































'''
plt.plot(df['tmpf_max'].groupby(df.index.hour).mean(),'r--')
plt.plot(df['tmpf_75'].groupby(df.index.hour).mean(),'r-')
plt.plot(df['tmpf_mean'].groupby(df.index.hour).mean(),'k-')
plt.plot(df['tmpf_25'].groupby(df.index.hour).mean(),'b-')
plt.plot(df['tmpf_min'].groupby(df.index.hour).mean(),'b--')
plt.show()


df_mon_max=df['tmpf_max'].groupby([df.index.month,df.index.hour]).max()
df_mon_75=df['tmpf_75'].groupby([df.index.month,df.index.hour]).quantile(q=0.75)
df_mon_mean=df['tmpf_mean'].groupby([df.index.month,df.index.hour]).mean()
df_mon_median=df['tmpf_median'].groupby([df.index.month,df.index.hour]).quantile(q=0.5)
df_mon_25=df['tmpf_25'].groupby([df.index.month,df.index.hour]).quantile(q=0.25)
df_mon_min=df['tmpf_min'].groupby([df.index.month,df.index.hour]).min()

for mmm in range(1,13):
  #plt.plot(df_mon_75[mmm,:]-df_mon_75[mmm,:].mean())
  plt.plot(df_mon_median[mmm,:])
plt.show()



#interquartile range large during afternoon, night steady
plt.plot(df['tmpf_75'].groupby(df.index.hour).mean()-df['tmpf_25'].groupby(df.index.hour).mean())
#plt.plot(df['tmpf_max'].groupby(df.index.hour).mean()-df['tmpf_min'].groupby(df.index.hour).mean())
plt.show()
'''

#just filter...----------------------------------------------------------------#Fig: Diurnal Boxplots
'''
for mm in range(1,13,1):
  print(mm)
  for hh in range(0,24):
    temp=df[tmpf_list].loc[(df[tmpf_list].index.hour==hh) & (df[tmpf_list].index.month==mm)]
    temp=(temp.to_numpy()).flatten()
    temp=temp[~np.isnan(temp)]
  
    #plt.boxplot(temp,positions=[2.3])
    plt.boxplot(temp,positions=[hh],whis=[10,90],sym='None',widths=0.7)
    plt.ylim([40,115])
  plt.show()
'''
#-----------------------------------------------------------------------------#
#-----------------------------end temperatures--------------------------------#
#-----------------------------------------------------------------------------#




#-----------------------------------------------------------------------------#
#-------------------------------Spatial Maps----------------------------------#
#-----------------------------------------------------------------------------#

#------------READ URBFRAC-------------#
f=open('URBFRAC_KC_v0_1km.pkl','rb')    
var_list=pickle.load(f)
f.close()

lons_urbfrac=np.array(var_list[0])
lats_urbfrac=np.array(var_list[1])
urbfrac=np.array(var_list[2])
del f,var_list
#-------------------------------------#

'''
#GET CARTOPY DATA
reader=shpreader.Reader('C:/a_Projects/zMASTER_PYTHON/tl_2016_us_primaryroads/tl_2016_us_primaryroads.shp')
names=[]
geoms=[]
for rec in reader.records():
    if (rec.attributes['FULLNAME'][0]=='I'):
        names.append(rec)
        geoms.append(rec.geometry)

#make interstate feature
roads=ShapelyFeature(geoms,crs.PlateCarree(),edgecolor='darkblue',lw=0.5,facecolor='none')
#States and coastlines
states=NaturalEarthFeature(category='cultural',scale='50m',facecolor='none',name='admin_1_states_provinces_shp')
'''



#------------------------------Urban Frac Maps--------------------------------#
'''
thecmap='gist_yarg_r'

fig=plt.figure(figsize=(5,3.6),dpi=600)
#ax=plt.axes(projection=crs.PlateCarree())
#ax.add_feature(roads,linewidth=1.25,edgecolor='black')
plt.pcolor(lons_urbfrac,lats_urbfrac,urbfrac,cmap=thecmap)
for i in range(len(lats)):
  plt.scatter(lons[i],lats[i],c=urbfrac_median[i],cmap=thecmap,zorder=100,edgecolors='yellow',linewidths=0.5)
  plt.clim(0,0.7)

plt.xlim([-97.8,-96.25])
plt.xticks([-97.8,-97.6,-97.4,-97.2,-97,-96.8,-96.6,-96.4,-96.2])
plt.ylim([32.15,33.53])
plt.clim(0,0.8)
plt.colorbar()
plt.show()
#fig.savefig('./figs/urbfrac_stns.png',format='png',dpi=600) #------------------FIG: Urbfrac+Stns
fig.savefig('./figs/urbfrac_stns.svg',format='svg',dpi=600) #------------------FIG: Urbfrac+Stns
'''


'''
#station urbfrac histogram
fig=plt.figure(figsize=(5,3.6),dpi=600)
plt.hist(urbfrac_mean,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],rwidth=0.95)
plt.xlim([-0.005,1.0])
plt.xlabel('Urban Fraction')
plt.ylabel('Number of Stations')
plt.show()
#fig.savefig('./figs/hist_urbfrac.png',format='png',dpi=600) #-------------FIG: Urbfrac Stn Histogram
#fig.savefig('./figs/hist_urbfrac.svg',format='svg',dpi=600) #-------------FIG: Urbfrac Stn Histogram
'''





#Alternative urb fractions
'''
thecmap='YlOrRd'
cmap=get_cmap('YlOrRd')
cmap_21=cmap(np.linspace(0,1,101))
cmap_21[0][0:3]=[1,1,1]
thecmap=ListedColormap(cmap_21)
'''

thecmap='gist_yarg'
fig=plt.figure(figsize=(4,3.6),dpi=300)
#ax=plt.axes(projection=crs.PlateCarree())
#ax.add_feature(roads,linewidth=1.25,edgecolor='black')
plt.pcolor(lons_urbfrac,lats_urbfrac,urbfrac,cmap=thecmap)
for i in range(len(lats)):
  if urbfrac_median[i]>0.4:
    plt.scatter(lons[i],lats[i],c=urbfrac_median[i],cmap=thecmap,zorder=10+100*urbfrac_median[i],edgecolors='w',linewidths=0.5)
  if urbfrac_median[i]<=0.4:
    plt.scatter(lons[i],lats[i],c=urbfrac_median[i],cmap=thecmap,zorder=10+100*urbfrac_median[i],edgecolors='k',linewidths=0.5)
  plt.clim(0,0.7)

plt.xlim([-94.9,-94.25])
plt.ylim([38.75,39.35])
plt.clim(0,0.8)
plt.colorbar()
plt.show()
#fig.savefig('./figs/urbfrac_stns.png',format='png',dpi=600) #------------------FIG: Urbfrac+Stns
fig.savefig('./figs/urbfrac_stns.svg',format='svg',dpi=800) #------------------FIG: Urbfrac+Stns


#sys.exit()









#------------------------------choose conditions------------------------------#
thevwnd=-999
theokta=2
thewspd=3
theurbfrac=0.4
#-----------------------------------------------------------------------------#
#cmap=cm.turbo(np.linspace(-3,1,30)) #set colormap




fig=plt.figure(figsize=(5,3.6),dpi=600)
plt.hist(df_slopes['wnd_spd_ave'],bins=[0,1,2,3,4,5,6,7,8,9,10,11,12],rwidth=0.95,density=True)
plt.xlim([-0.1,12.0])
plt.xlabel('Wind Speed')
plt.ylabel('Density')
plt.show()
#fig.savefig('./figs/hist_wspd.png',format='png',dpi=600) #-------------FIG: Urbfrac Stn Histogram
fig.savefig('./figs/hist_wspd.svg',format='svg',dpi=600) #-------------FIG: Urbfrac Stn Histogram


fig=plt.figure(figsize=(5,3.6),dpi=600)
plt.hist(df_slopes['okta_ave'],bins=[0,1,2,3,4,5,6,7,8],rwidth=0.95,density=True)
plt.xlim([-0.1,8.1])
plt.xlabel('Cloud Okta')
plt.ylabel('Density')
plt.show()
#fig.savefig('./figs/hist_okta.png',format='png',dpi=600) #-------------FIG: Urbfrac Stn Histogram
fig.savefig('./figs/hist_okta.svg',format='svg',dpi=600) #-------------FIG: Urbfrac Stn Histogram




#----------------------------------Plot Slopes--------------------------------#
#-----------------------------Compare 2 stations------------------------------#

#----------------------------------loop lots---------------------------------
'''
urb_stns=stn_id[urbfrac_median>0.6]
rur_stns=stn_id[urbfrac_median<0.1]

#stnID_1_slope='tmpf_053239_slope'
#stnID_1_storeT='tmpf_053239_storeT'

#stnID_2_slope='tmpf_016271_slope'
#stnID_2_storeT='tmpf_016271_storeT'

for aa in range(0,len(urb_stns)):
  theid1=f'{str(round(urb_stns[aa])):0>6}'
  stnID_1_slope='tmpf_'+theid1+'_slope'
  stnID_1_storeT='tmpf_'+theid1+'_storeT'
  
  for bb in range(0,len(rur_stns)):
    theid2=f'{str(round(rur_stns[bb])):0>6}'
    stnID_2_slope='tmpf_'+theid2+'_slope'
    stnID_2_storeT='tmpf_'+theid2+'_storeT'
    
    conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
               (df_slopes['wnd_spd_ave']<=thewspd) & \
               ( (df_slopes['month']>=3) & (df_slopes['month']<=5) ) & \
               (df_slopes['year']>=2020) & \
               (~df_slopes[stnID_1_slope].isna()) & (~df_slopes[stnID_2_slope].isna())
    
    #print(conditions[conditions==True].shape[0])
    
    #-----------------------BoxPlot-ish
    temp_stn1,temp_stn2=[],[]
    for i in range(0,df_slopes[stnID_1_slope].loc[conditions].size):
      temp=df_slopes[stnID_1_storeT].loc[conditions][i]
      if len(temp_stn1)==0:
        temp_stn1=temp
      else:
        temp_stn1=np.vstack([temp_stn1,temp])
      
      temp=df_slopes[stnID_2_storeT].loc[conditions][i]
      if len(temp_stn2)==0:
        temp_stn2=temp
      else:
        temp_stn2=np.vstack([temp_stn2,temp])
    
    if conditions[conditions==True].shape[0]>15:
      stn1_pert=temp_stn1-np.nanmean(temp_stn1,axis=1,keepdims=True)
      stn2_pert=temp_stn2-np.nanmean(temp_stn2,axis=1,keepdims=True)
      
      #Get linear regression--------1
      #deal with data...
      temps=stn1_pert.flatten()
      X=np.linspace(0,stn1_pert.shape[1]-1,stn1_pert.shape[1])
      X=np.tile(X,stn1_pert.shape[0])
      temps2=temps[~np.isnan(temps)]
      X2=X[~np.isnan(temps)]
      X2=X2.reshape(-1,1)
      
      #FIT: Regular
      lr=linear_model.LinearRegression()
      lr.fit(X2,temps2)
      slope1,inter1=lr.coef_[0],lr.intercept_
            
      #FIT: HUBER
      huber=linear_model.HuberRegressor()
      huber.fit(X2,temps2)
      #slope1,inter1=huber.coef_[0],huber.intercept_
      
      #Get linear regression--------2
      #deal with data...
      temps=stn2_pert.flatten()
      X=np.linspace(0,stn1_pert.shape[1]-1,stn1_pert.shape[1])
      X=np.tile(X,stn1_pert.shape[0])
      temps2=temps[~np.isnan(temps)]
      X2=X[~np.isnan(temps)]
      X2=X2.reshape(-1,1)
      
      #FIT: Regular
      lr=linear_model.LinearRegression()
      lr.fit(X2,temps2)
      slope2,inter2=lr.coef_[0],lr.intercept_
      
      #FIT: HUBER
      huber=linear_model.HuberRegressor()
      huber.fit(X2,temps2)
      #slope2,inter2=huber.coef_[0],huber.intercept_
      
      
      fig=plt.figure(figsize=(5,3),dpi=600)
      PROPS1={'boxprops':{'edgecolor':'r'},'medianprops':{'color':'r'},
              'whiskerprops':{'color':'r'},'capprops':{'color':'r'},
              'flierprops':{'markeredgecolor':'r','markerfacecolor':'None','marker':'o','markersize':'3'} }
      PROPS2={'boxprops':{'edgecolor':'b'},'medianprops':{'color':'b'},
              'whiskerprops':{'color':'b'},'capprops':{'color':'b'},
              'flierprops':{'markeredgecolor':'b','markerfacecolor':'None','marker':'o','markersize':'3'} }
      
      sns.boxplot(stn1_pert,width=0.3,color='r',**PROPS1)
      sns.boxplot(stn2_pert,width=0.3,color='b',**PROPS2)
      
      end_pt=stn2_pert.shape[1]-1
      center_pt=(stn2_pert.shape[1]-1)/2
      
      plt.xlabel('Hour relative to sunset')
      plt.ylabel('Normalized temperature')
      plt.title('stns: '+theid1+'/'+theid2+' number: '+str(conditions[conditions==True].shape[0]))
      
      plt.plot([0,end_pt],[-center_pt*slope1,-center_pt*slope1+(end_pt)*slope1],'r-',zorder=1)
      plt.plot([0,end_pt],[-center_pt*slope2,-center_pt*slope2+(end_pt)*slope2],'b-',zorder=100)
      plt.ylim([-5,6])
      plt.show()

sys.exit()
'''







#-------------------------
#rural long 
#urban long 

#Possible:
#054769 (0.31)
#058911 (0.86)
#059049 (0.70)
#044809 (0.58)
#009738 (0.28)

stnID_1_slope='tmpf_009738_slope'
stnID_1_storeT='tmpf_009738_storeT'
stnID_2_slope='tmpf_058911_slope'
stnID_2_storeT='tmpf_058911_storeT'

#plt.plot(df_slopes[stnID_1_slope],'ro')
#plt.plot(df_slopes[stnID_2_slope],'ko')
#plt.show()



conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
           (df_slopes['wnd_spd_ave']<=thewspd) & \
           ( (df_slopes['month']>=3) & (df_slopes['month']<=5) ) & \
           (df_slopes['year']>=2020) & \
           (~df_slopes[stnID_1_slope].isna()) & (~df_slopes[stnID_2_slope].isna())


print('Matching days for boxplot: '+str(conditions[conditions==True].shape[0]))

'''
plt.plot(df_slopes[stnID_1].loc[conditions],'b.')
plt.plot(df_slopes['tmpf_113642_slope'].loc[conditions],'r.')
plt.show()


for i in range(0,df_slopes['tmpf_053193_slope'].loc[conditions].size):
  slope1=df_slopes['tmpf_053193_slope'].loc[conditions][i]
  slope2=df_slopes['tmpf_113642_slope'].loc[conditions][i]
  plt.plot([0,5],[-2.5*slope1,-2.5*slope1+5*slope1],'b-',zorder=1)
  plt.plot([0,5],[-2.5*slope2,-2.5*slope2+5*slope2],'r-',zorder=100)
plt.show()


#-----------------------Scatterish
for i in range(0,df_slopes['tmpf_053193_slope'].loc[conditions].size):
  temp=df_slopes['tmpf_053193_storeT'].loc[conditions][i]
  plt.plot(temp-np.nanmean(temp),'bx')
  
  temp=df_slopes['tmpf_113642_storeT'].loc[conditions][i]
  plt.plot(temp-np.nanmean(temp),'rx')
  
  slope1=df_slopes['tmpf_053193_slope'].loc[conditions][i]
  slope2=df_slopes['tmpf_113642_slope'].loc[conditions][i]
  plt.plot([0,5],[-2.5*slope1,-2.5*slope1+5*slope1],'b-',zorder=1)
  plt.plot([0,5],[-2.5*slope2,-2.5*slope2+5*slope2],'r-',zorder=100)
plt.show()
'''


#-----------------------BoxPlot-ish
temp_stn1,temp_stn2=[],[]
for i in range(0,df_slopes[stnID_1_slope].loc[conditions].size):
  temp=df_slopes[stnID_1_storeT].loc[conditions][i]
  if len(temp_stn1)==0:
    temp_stn1=temp
  else:
    temp_stn1=np.vstack([temp_stn1,temp])
  
  temp=df_slopes[stnID_2_storeT].loc[conditions][i]
  if len(temp_stn2)==0:
    temp_stn2=temp
  else:
    temp_stn2=np.vstack([temp_stn2,temp])

stn1_pert=temp_stn1-np.nanmean(temp_stn1,axis=1,keepdims=True)
stn2_pert=temp_stn2-np.nanmean(temp_stn2,axis=1,keepdims=True)

#Get linear regression--------1
#deal with data...
temps=stn1_pert.flatten()
X=np.linspace(0,stn1_pert.shape[1]-1,stn1_pert.shape[1])
X=np.tile(X,stn1_pert.shape[0])

temps2=temps[~np.isnan(temps)]
X2=X[~np.isnan(temps)]
X2=X2.reshape(-1,1)

#FIT: Regular
lr=linear_model.LinearRegression()
lr.fit(X2,temps2)
slope1,inter1=lr.coef_[0],lr.intercept_
      
#FIT: HUBER
huber=linear_model.HuberRegressor()
huber.fit(X2,temps2)
#slope1,inter1=huber.coef_[0],huber.intercept_



#Get linear regression--------2
#deal with data...
temps=stn2_pert.flatten()
X=np.linspace(0,stn1_pert.shape[1]-1,stn1_pert.shape[1])
X=np.tile(X,stn1_pert.shape[0])

temps2=temps[~np.isnan(temps)]
X2=X[~np.isnan(temps)]
X2=X2.reshape(-1,1)

#FIT: Regular
lr=linear_model.LinearRegression()
lr.fit(X2,temps2)
slope2,inter2=lr.coef_[0],lr.intercept_

#FIT: HUBER
huber=linear_model.HuberRegressor()
huber.fit(X2,temps2)
#slope2,inter2=huber.coef_[0],huber.intercept_



fig=plt.figure(figsize=(5,3),dpi=600)
PROPS1={'boxprops':{'edgecolor':'b'},'medianprops':{'color':'b'},
        'whiskerprops':{'color':'b'},'capprops':{'color':'b'},
        'flierprops':{'markeredgecolor':'b','markerfacecolor':'None','marker':'o','markersize':'3'} }
PROPS2={'boxprops':{'edgecolor':'r'},'medianprops':{'color':'r'},
        'whiskerprops':{'color':'r'},'capprops':{'color':'r'},
        'flierprops':{'markeredgecolor':'r','markerfacecolor':'None','marker':'o','markersize':'3'} }

sns.boxplot(stn1_pert,width=0.3,color='b',**PROPS1)
sns.boxplot(stn2_pert,width=0.3,color='r',**PROPS2)

end_pt=stn2_pert.shape[1]-1
center_pt=(stn2_pert.shape[1]-1)/2

plt.xlabel('Hour relative to sunset')
plt.ylabel('Normalized temperature')

plt.plot([0,end_pt],[-center_pt*slope1,-center_pt*slope1+(end_pt)*slope1],'b-',zorder=1)
plt.plot([0,end_pt],[-center_pt*slope2,-center_pt*slope2+(end_pt)*slope2],'r-',zorder=100)
plt.ylim([-5,6])
plt.show()
#fig.savefig('./figs/boxplot_slopes.png',format='png',dpi=600) #-------------FIG: Boxplot Slopse
fig.savefig('./figs/boxplot_slopes.svg',format='svg',dpi=600) #-------------FIG: Boxplot Slopse





'''
#Raw Data
plt.plot(df_slopes['tmpf_053193_slope'].loc[conditions],'b.')
plt.plot(df_slopes['tmpf_113642_slope'].loc[conditions],'r.')
plt.show()


for i in range(0,df_slopes['tmpf_053193_slope'].loc[conditions].size):
  slope1=df_slopes['tmpf_053193_slope'].loc[conditions][i]
  slope2=df_slopes['tmpf_113642_slope'].loc[conditions][i]
  plt.plot([0,5],[-2.5*slope1,-2.5*slope1+5*slope1],'b-',zorder=1)
  plt.plot([0,5],[-2.5*slope2,-2.5*slope2+5*slope2],'r-',zorder=100)
plt.show()
'''








#-----MONTHLY------#
'''
for d in range(6,7):
  conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
             (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==d) & \
             (df_slopes['year']>=2020)
  
  
  clim_low,clim_high=getclims(d)
  
  fig=plt.figure(figsize=(5,3.5),dpi=150)
  for i in range(len(lats)):
    value=df_slopes[tmpf_list[i]+'_slope'].loc[conditions].median(axis=0)
    if ~np.isnan(value):
      plt.scatter(lons[i],lats[i],c=value,cmap='turbo',zorder=100*value)
      plt.clim(clim_low,clim_high)
  plt.xlim([-97.6,-96.55])
  plt.ylim([32.4,33.6])
  plt.title(str(d)+' slope')
  plt.clim(clim_low,clim_high)
  plt.colorbar()
  plt.show()
'''


#-----SEASONALLY------#
for ii in range(4):
  if ii==0:
    m1,m2,m3=12,1,2
    clim_low,clim_high=-2.0,-0.5
  if ii==1:
    m1,m2,m3=3,4,5
    clim_low,clim_high=-2.0,-0.5
  if ii==2:
    m1,m2,m3=6,7,8
    clim_low,clim_high=-2.0,-0.5
  if ii==3:
    m1,m2,m3=9,10,11
    clim_low,clim_high=-2.0,-0.5
  
  conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
             (df_slopes['wnd_spd_ave']<=thewspd) & \
             ((df_slopes['month']==m1) | (df_slopes['month']==m2) | (df_slopes['month']==m3)) & \
             (df_slopes['year']>=2020)
  
  fig=plt.figure(figsize=(5,3.6),dpi=600)
  for i in range(len(lats)):
    value=df_slopes[tmpf_list[i]+'_slope'].loc[conditions].median(axis=0)
    if ~np.isnan(value):
      plt.scatter(lons[i],lats[i],c=value,cmap='turbo',zorder=100*value)
      plt.clim(clim_low,clim_high)
  plt.xlim([-94.9,-94.25])
  plt.ylim([38.75,39.35])
  plt.title(str(m1)+' slope')
  #plt.clim(clim_low,clim_high)
  plt.colorbar(ticks=np.arange(-4,0,0.25))
  plt.show()
#  fig.savefig('./figs/seasonal_slopes'+str(ii)+'.png',format='png',dpi=600) #--FIGURE: Seasonal Slopes
  fig.savefig('./figs/seasonal_slopes'+str(ii)+'.svg',format='svg',dpi=600) #--FIGURE: Seasonal Slopes





'''
#-----ANNUAL------#
conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
           (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['year']>=2020)

clim_low,clim_high=-2.0,-0.75

fig=plt.figure(figsize=(5,3.6),dpi=600)
for i in range(len(lats)):
  value=df_slopes[tmpf_list[i]+'_slope'].loc[conditions].median(axis=0)
  if ~np.isnan(value):
    plt.scatter(lons[i],lats[i],c=value,cmap='turbo',zorder=100*value)
    plt.clim(clim_low,clim_high)
plt.xlim([-97.6,-96.55])
plt.ylim([32.4,33.6])
plt.title('Annual slope')
plt.clim(clim_low,clim_high)
plt.colorbar(ticks=np.arange(-4,0,0.25))
plt.show()
fig.savefig('./figs/annual_slopes.png',format='png',dpi=600) #-----------------FIGURE: Annual Slopes
'''










#---HISTOGRAMS for weather
'''
for d in range(1,13):
  conditionsA=(df_slopes['okta_ave']<=3) & (df_slopes['vwnd_ave']>=-999) & \
              (df_slopes['wnd_spd_ave']<=3) & (df_slopes['month']==d) & \
              (df_slopes['year']>=2020)
  conditionsB=(df_slopes['okta_ave']>=5) & (df_slopes['vwnd_ave']>=-999) & \
              (df_slopes['wnd_spd_ave']<=3) & (df_slopes['month']==d) & \
              (df_slopes['year']>=2020)
  
  #clear store values
  store1=np.array([])
  store2=np.array([])
  
  for i in range(len(lats)):
    store1=np.append(store1,df_slopes[tmpf_list[i]+'_slope'].loc[conditionsA])
    store2=np.append(store2,df_slopes[tmpf_list[i]+'_slope'].loc[conditionsB])
    
  sns.distplot(store1,kde=True,kde_kws={'clip':(-5,2),'linewidth':3,'color':'b'}, \
               bins=np.arange(-5.5,2,0.5),hist=True)
  sns.distplot(store2,kde=True,kde_kws={'clip':(-5,2),'linewidth':3,'color':'r'}, \
               bins=np.arange(-5.5,2,0.5),hist=True)  
  plt.title('Weather Diffs')
  plt.show()
'''





#---HISTOGRAMS for urbfrac
coolrates=[]
for d in range(1,13):
  conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
              (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==d) & \
              (df_slopes['year']>=2020)
  
  #clear store values
  store1,store2=np.array([]),np.array([])
  
  #instead of looping hrough all stations, loop through the list of stations like, stns_urbfrac_high=['tmpf_55555'
  for i in range(len(lats)):
    if urbfrac_median[i]<=theurbfrac:
      store1=np.append(store1,df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions,None])
    if urbfrac_median[i]>=theurbfrac:
      store2=np.append(store2,df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions,None])
  
  #store values
  coolrates.append(np.nanmean(store2)-np.nanmean(store1))
  
  '''
  fig=plt.figure(figsize=(5,3.6),dpi=600)
  sns.distplot(store1,kde=True,kde_kws={'clip':(-4.5,0.5),'linewidth':3,'color':'b'}, \
               bins=np.arange(-4.5,0.5,0.25),hist=True)
  sns.distplot(store2,kde=True,kde_kws={'clip':(-4.5,0.5),'linewidth':3,'color':'r'}, \
               bins=np.arange(-4.5,0.5,0.25),hist=True)
  
  plt.plot([np.nanmean(store1),np.nanmean(store1)],[0,2],'b-')
  #plt.plot([np.nanmedian(store1),np.nanmedian(store1)],[0,1],'b--')
  
  plt.plot([np.nanmean(store2),np.nanmean(store2)],[0,2],'r-')
  #plt.plot([np.nanmedian(store2),np.nanmedian(store2)],[0,1],'r--')
  
  plt.xlim([-4.1,0.1])
  plt.ylim([0,1.5])
  plt.title('Cooling rates during month '+str(d)+' 2020-2022'+
            ' \n below (blue) and above (red) '+str(theurbfrac)+' '+
            str(np.count_nonzero(~np.isnan(store1)))+'/'+
            str(np.count_nonzero(~np.isnan(store2))))
  plt.show()
  '''



'''
print('cld: '+str(theokta)+'  wspd: '+str(thewspd)+'  vwnd: '+str(thevwnd))
for ttt in range(0,len(coolrates)):
  print(f'{coolrates[ttt]:.2f}')
print(np.nanmean(coolrates))


months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig=plt.figure(figsize=(5,3.6),dpi=600)
plt.plot(np.linspace(1,12,12),coolrates,'ko-')
plt.ylim([0,0.4])
plt.xticks(np.linspace(1,12,12),months)
plt.ylabel('Cooling Rate Difference (C/hr)')
plt.show()
#fig.savefig('./figs/monthly_coolrate_diffs.png',format='png',dpi=600) #-----------------FIGURE: Annual Slopes
#fig.savefig('./figs/monthly_coolrate_diffs.svg',format='svg',dpi=600) #-----------------FIGURE: Annual Slopes
'''







#---HISTOGRAMS for urbfrac------SEASONAL
coolrates_season=[]
for ii in range(4):
  if ii==0:
    m1,m2,m3=12,1,2
  if ii==1:
    m1,m2,m3=3,4,5
  if ii==2:
    m1,m2,m3=6,7,8
  if ii==3:
    m1,m2,m3=9,10,11
  
  #coolrates_season=0*lats
  conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
             (df_slopes['wnd_spd_ave']<=thewspd) & \
             ((df_slopes['month']==m1) | (df_slopes['month']==m2) | (df_slopes['month']==m3)) & \
             (df_slopes['year']>=2020)
  
  #clear store values
  store1,store2=np.array([]),np.array([])
  
  #instead of looping hrough all stations, loop through the list of stations like, stns_urbfrac_high=['tmpf_55555'
  for i in range(len(lats)):
    if urbfrac_median[i]<=theurbfrac:
      store1=np.append(store1,df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions,None])
    if urbfrac_median[i]>=theurbfrac:
      store2=np.append(store2,df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions,None])
  
  #store values
  coolrates_season.append(np.nanmean(store2)-np.nanmean(store1))
  
  print(stats.ttest_ind(a=store1[~np.isnan(store1)],b=store2[~np.isnan(store2)],equal_var=False))
  
  fig=plt.figure(figsize=(5,3.6),dpi=600)
  sns.distplot(store1,kde=True,kde_kws={'clip':(-4.5,0.5),'linewidth':3,'color':'b'}, \
               bins=np.arange(-4.5,0.5,0.25),hist=True)
  sns.distplot(store2,kde=True,kde_kws={'clip':(-4.5,0.5),'linewidth':3,'color':'r'}, \
               bins=np.arange(-4.5,0.5,0.25),hist=True)
  
  plt.plot([np.nanmean(store1),np.nanmean(store1)],[0,2],'b-')
  #plt.plot([np.nanmedian(store1),np.nanmedian(store1)],[0,1],'b--')
  plt.plot([np.nanmean(store2),np.nanmean(store2)],[0,2],'r-')
  #plt.plot([np.nanmedian(store2),np.nanmedian(store2)],[0,1],'r--')
  
  
  plt.xlim([-3.25,0.1])
  plt.ylim([0,1.6])
  plt.xlabel('Cooling Rate C/hr')
  plt.title('Cooling rates during month '+str(m1)+' 2020-2022'+
            ' \n below (blue) and above (red) '+str(theurbfrac)+' '+
            str(np.count_nonzero(~np.isnan(store1)))+'/'+
            str(np.count_nonzero(~np.isnan(store2))))
  plt.show()
#  fig.savefig('./figs/hist_seasonal_'+str(ii)+'.png',format='png',dpi=600) #-----------------FIGURE: Annual Slopes
  fig.savefig('./figs/hist_seasonal_'+str(ii)+'.svg',format='svg',dpi=600) #-----------------FIGURE: Annual Slopes



print('cld: '+str(theokta)+'  wspd: '+str(thewspd)+'  vwnd: '+str(thevwnd))
for ttt in range(0,len(coolrates_season)):
  print(f'{coolrates_season[ttt]:.2f}')
print(np.nanmean(coolrates_season))






#---scatter plt urb frac vs. cooling rates
'''
#loop through all stations to get average cooling rates
for mm in range(0,12):
  cooling_rates=0*lats
  for i in range(len(lats)):
    conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
               (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==mm+1) & \
               (df_slopes['year']>=2020)
    
    cooling_rates[i]=np.nanmean(df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions])
  
  
  xxx=urbfrac_mean[~np.isnan(cooling_rates)]
  yyy=cooling_rates[~np.isnan(cooling_rates)]
#  if len(yyy)>0:
  slope,intercept,r,p,std_err=stats.linregress(xxx,yyy)
#    print(r)
  
  plt.plot(urbfrac_mean,cooling_rates,'k.')
  plt.plot([0,0.8],[intercept,intercept+slope*0.8])
  
  plt.ylim([-5,0.5])
  plt.text(0,intercept+0.1,f'{r:.2f}')
  plt.title(str(mm+1))
  plt.show()
'''



#---scatter plt urb frac vs. cooling rates------------------------------------SEASONAL URB vs SLOPE
#-----SEASONALLY------#
for ii in range(4):
  if ii==0:
    m1,m2,m3=12,1,2
    clim_low,clim_high=-2.0,-0.75
  if ii==1:
    m1,m2,m3=3,4,5
    clim_low,clim_high=-2.0,-0.75
  if ii==2:
    m1,m2,m3=6,7,8
    clim_low,clim_high=-2.0,-0.75
  if ii==3:
    m1,m2,m3=9,10,11
    clim_low,clim_high=-2.0,-0.75
  
  cooling_rates=0*lats
  for i in range(len(lats)):
    conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
               (df_slopes['wnd_spd_ave']<=thewspd) & \
               ((df_slopes['month']==m1) | (df_slopes['month']==m2) | (df_slopes['month']==m3)) & \
               (df_slopes['year']>=2020)
    
    cooling_rates[i]=np.nanmean(df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions])
    
  xxx=urbfrac_mean[~np.isnan(cooling_rates)]
  yyy=cooling_rates[~np.isnan(cooling_rates)]
  slope,intercept,r,p,std_err=stats.linregress(xxx,yyy)
  print(p)
  
  fig=plt.figure(figsize=(5,3.6),dpi=600)
  plt.plot(xxx,yyy,'k.')
  plt.plot([0,0.9],[intercept,intercept+slope*0.9])
  
  plt.text(0.02,-0.4,f'y = {slope:0.2f} x + {intercept:0.2f}')
  plt.text(0.02,-0.5,f'R = {r:.2f}')
  
  plt.xlim([0,0.825])
  plt.ylim([-2.0,-0.25])
  
  plt.xlabel('Urban Fraction')
  plt.ylabel('Cooling Rate')
  plt.title(str(m1))
  plt.show()
#  fig.savefig('./figs/slopes_seasonal_'+str(ii)+'.png',format='png',dpi=600) #-----------------FIGURE: Annual Slopes
  fig.savefig('./figs/slopes_seasonal_'+str(ii)+'.svg',format='svg',dpi=600) #-----------------FIGURE: Annual Slopes






'''
for mm in range(0,12):
  cooling_rates=0*lats
  for i in range(len(lats)):
    conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
               (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==mm+1) & \
               (df_slopes['year']>=2020)
    
    cooling_rates[i]=np.nanmean(df_slopes[tmpf_list[i]+'_slope'].to_numpy()[conditions])
  
  
  xxx=urbfrac_mean[~np.isnan(cooling_rates)]
  yyy=cooling_rates[~np.isnan(cooling_rates)]
#  if len(yyy)>0:
  slope,intercept,r,p,std_err=stats.linregress(xxx,yyy)
#    print(r)
  
  plt.plot(urbfrac_mean,cooling_rates,'k.')
  plt.plot([0,0.8],[intercept,intercept+slope*0.8])
  
  plt.ylim([-5,0.5])
  plt.text(0,intercept+0.1,f'{r:.2f}')
  plt.title(str(mm+1))
  plt.show()

'''




#sys.exit()

#plt.plot(coolrates,'ko')










#------------------------------Just for table---------------------------------#
coolrates=np.nan*np.empty([4,12])
coolrates_sig=np.nan*np.empty([4,12])

for theyear in range(2020,2023):
  for d in range(1,13):
    conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
               (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==d) & \
               (df_slopes['year']==theyear)
    
    #clear store values
    store1,store2=np.array([]),np.array([])
    
    #instead of looping hrough all stations, loop through the list of stations like, stns_urbfrac_high=['tmpf_55555'
    for i in range(len(lats)):
      if urbfrac_median[i]<=theurbfrac:
        store1=np.append(store1,df_slopes[tmpf_list[i]+'_slope'].loc[conditions])
      if urbfrac_median[i]>=theurbfrac:
        store2=np.append(store2,df_slopes[tmpf_list[i]+'_slope'].loc[conditions])
    
    #if np.any(store1):
    #print(ttest_ind(store1,store2,nan_policy='omit',equal_var='False'))
    tinfo=ttest_ind(store1,store2,nan_policy='omit',equal_var='False')
    
    coolrates[theyear-2020,d-1]=np.nanmean(store2)-np.nanmean(store1)
    if tinfo[1]<0.01:
      coolrates_sig[theyear-2020,d-1]=0
    if tinfo[1]>=0.01:
      coolrates_sig[theyear-2020,d-1]=1
      


#repeate for all years    
for d in range(1,13):
  conditions=(df_slopes['okta_ave']<=theokta) & (df_slopes['vwnd_ave']>=thevwnd) & \
             (df_slopes['wnd_spd_ave']<=thewspd) & (df_slopes['month']==d) & \
             (df_slopes['year']>=2020)
  
  #clear store values
  store1,store2=np.array([]),np.array([])
  
  #instead of looping through all stations, loop through the list of stations like, stns_urbfrac_high=['tmpf_55555'
  for i in range(len(lats)):
    if urbfrac_median[i]<=theurbfrac:
      store1=np.append(store1,df_slopes[tmpf_list[i]+'_slope'].loc[conditions])
    if urbfrac_median[i]>=theurbfrac:
      store2=np.append(store2,df_slopes[tmpf_list[i]+'_slope'].loc[conditions])
  
  #if np.any(store1):
  #print(ttest_ind(store1,store2,nan_policy='omit',equal_var='False'))
  tinfo=ttest_ind(store1,store2,nan_policy='omit',equal_var='False')
  
  coolrates[3,d-1]=np.nanmean(store2)-np.nanmean(store1)
  if tinfo[1]<0.05:
    coolrates_sig[3,d-1]=0
  if tinfo[1]>=0.05:
    coolrates_sig[3,d-1]=1


df_cr=pd.DataFrame(coolrates)
#round for table
df_cr=df_cr.round(2)


def make_pretty(styler):
    styler.set_caption("Weather Conditions")
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
    return styler
  
df_cr.style.pipe(make_pretty)

#save to excel with colors
#df_cr.style.pipe(make_pretty).to_excel('testNN.xlsx',engine='openpyxl')



#plt.plot(df_cr.iloc[3])
#plt.show()



'''
import matplotlib as mpl
mpl.use('svg')
new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
'''

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig=plt.figure(figsize=(5,3.6),dpi=600)
#plt.plot(np.linspace(1,12,12),df_cr.iloc[0],'o--',color='0.80',markersize=5)
#plt.plot(np.linspace(1,12,12),df_cr.iloc[1],'o--',color='0.50',markersize=5)
#plt.plot(np.linspace(1,12,12),df_cr.iloc[2],'o--',color='0.30',markersize=5)
plt.plot(np.linspace(1,12,12),df_cr.iloc[0],'--',color='g',markersize=5)
plt.plot(np.linspace(1,12,12),df_cr.iloc[1],'--',color='r',markersize=5)
plt.plot(np.linspace(1,12,12),df_cr.iloc[2],'--',color='b',markersize=5)

plt.plot(np.linspace(1,12,12),df_cr.iloc[3],'o-',color='0.00',markersize=8,linewidth=3)
plt.ylim([0.0,0.4])
plt.xticks(np.linspace(1,12,12),months)
plt.ylabel('Cooling Rate Difference (C/hr)')
plt.show()
#fig.savefig('./figs/monthly_coolrate_diffs_YearALL.png',format='png',dpi=600) #-----------------FIGURE: Annual Slopes
fig.savefig('./figs/monthly_coolrate_diffs_YearALL.svg',format='svg',dpi=600) #-----------------FIGURE: Annual Slopes








sys.exit()


#heatmap etc
df_tmpfonly=df[tmpf_list]

#drop problem     (50909: long no drop? negative) (002508)
#df_tmpfonly=df_tmpfonly.drop(['tmpf_050909'],axis=1)
#df_tmpfonly=df_tmpfonly.drop(['tmpf_002508'],axis=1)
#df_tmpfonly=df_tmpfonly.drop(['tmpf_114329'],axis=1)
#df_tmpfonly=df_tmpfonly.drop(['tmpf_114305'],axis=1)
#df_tmpfonly=df_tmpfonly.drop(['tmpf_124993'],axis=1)
#asdf=np.array(corr_df)
#np.where(asdf==np.nanmin(asdf))
#tmpf_list[101]
#Corr of 0.97

df_tmpfonly.columns=df_tmpfonly.columns.str.lstrip('tmpf_')

sns.heatmap(df_tmpfonly.corr(),annot=False,vmin=0.8,vmax=1.0,cmap='plasma',
            xticklabels=False,yticklabels=False)
plt.show()

corr_df=np.array(df_tmpfonly.corr())

mask=np.tril(np.ones_like(corr_df,dtype=bool))
np.fill_diagonal(mask,False) #remove auto-correlations

plt.plot(corr_df[mask],'k.')
#plt.ylim([0.5,1.05])
plt.show()


corr_df_clean=np.array(corr_df[mask])
np.nanmean(corr_df[mask])

