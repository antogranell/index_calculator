import pandas as pd
import numpy as np
import datetime as dt
import sys
import pyodbc
import time

dloc ='S:/Stoxx/Product Development and Research/Projects/2412 iSTOXX MUTB Value family/'
histloc = dloc + '05 History/'
calcloc = histloc + '01_calc_steps/'

sys.path.append('S:/Stoxx/Product Development and Research/Python')

#selection

def get_reg_filter(reg):
    if reg=='Europe':
        get_reg_filter = list(['EA','EB','ED','EE','EF','EG','EH','EI','EK','EL','EM','EO',
                               'EP','ER','ES','EU','EY','AT','BE','CH','CZ','DE','DK','FI',
                               'FR','GB','GR','IE','IT','LU','NL','NO','PT','SE'])
    if reg=='America':
        get_reg_filter = list(['AA','AC','US','CA'])
        
    elif reg=='AsiaPac':
        get_reg_filter = list(['PJ','PA','PH','PS','PZ','JP','AU','CN','HK','NZ','SG'])
        
    return get_reg_filter

def two_stage_standardisation(df, boundary = 4, outlier = 10):
    
    df_more_than_ub = df[df.iloc[:, 1] > outlier]
    df_less_than_lb = df[df.iloc[:, 1] < -1 * outlier]
    df1 =  df[(df.iloc[:, 1] <= outlier) & (df.iloc[:, 1] >= -1 * outlier)]
    
    value_med = df1.median()[len(df1.median())-1]
    value_std = df1.std()[len(df1.std())-1]

    ub = value_med + boundary * value_std
    lb = value_med - boundary * value_std

    df1_more_than_ub = df1[df1.iloc[:, 1] > ub]
    df1_less_than_lb = df1[df1.iloc[:, 1] < lb]
    df2 = df1[(df1.iloc[:, 1] <= ub) & (df1.iloc[:, 1] >= lb)]

    value2_med = df2.median()[len(df2.median())-1]
    value2_std = df2.std()[len(df2.std())-1]
    
    df2.iloc[:, 1] = (df2.iloc[:, 1] - value2_med) / value2_std

    df2_more_than_ub = df2[df2.iloc[:, 1] > boundary]
    df2_less_than_lb = df2[df2.iloc[:, 1] < -1 * boundary]
    df3 = df2[(df2.iloc[:, 1] <= boundary) & (df2.iloc[:, 1] >= -1 * boundary)]

    df_more_than_ub.iloc[:, 1] = boundary
    df_less_than_lb.iloc[:, 1] = -1 * boundary
    df1_more_than_ub.iloc[:, 1] = boundary
    df1_less_than_lb.iloc[:, 1] = -1 * boundary
    df2_more_than_ub.iloc[:, 1] = boundary
    df2_less_than_lb.iloc[:, 1] = -1 * boundary
    
    dfres = pd.concat([df3, df_more_than_ub, df_less_than_lb, df1_more_than_ub, df1_less_than_lb, df2_more_than_ub, df2_less_than_lb])
    dfres.columns = [dfres.columns[0], 'n' + dfres.columns[1]]
    
    dfres['median_step1'] = value_med
    dfres['stdev_step1'] = value_std
    dfres['lb'] = lb
    dfres['ub'] = ub
    dfres['median_step2'] = value2_med
    dfres['stdev_step2'] = value2_std
    
    return dfres

fs = ['jp600', 'gl1800', 'gl1800xjp']

idxname = (['ISMJVP', 'ISMJVN', 'ISMJVG'], #Japan
           ['ISMGVP', 'ISMGVN', 'ISMGVG'], #Global
           ['ISMGXJVP', 'ISMGXJVN', 'ISMGXJVG']) #Global ex Japan

idxct=0
for f in fs:
    
    print(f)
    dfjp = pd.read_csv(calcloc + '06_' + f + '_qad_wspit_adtv.csv', sep=';', dtype={'sedol': object})
    
    if f=='jp600':
        component_number = 300
    else:
        component_number = 600
    
    #start Dec 2002
    dfjp = dfjp[dfjp.date>'2002-12-01'].copy()
    
    del dfjp['dayct']
    
    #delete reits
    icb_reit = ['8671','8672','8673','8674','8675','8676','8677','8737','8733',8671,8672,8673,8674,8675,8676,8677,8737,8733,'REA']
    dfjp = dfjp[-dfjp.icb_sub.isin(list(icb_reit))]
    
    #delete jp for exjp
    if f=='gl1800xjp':
        dfjp = dfjp[-dfjp.country.isin(list(['PJ','JP']))]

    factors = ['BPR','EPR','CFPR']

    dfjp['BPR'] = 1 / dfjp['9302_PB']
    dfjp['EPR'] = 1 / dfjp['9102_PE']
    dfjp['CFPR'] = 1 / dfjp['9602_PCF']

    dfjp['accruals'] = (dfjp['NetIncome'] - dfjp['4860_OpCF']) / dfjp['2999_TotalAssets']
    dfjp['screen'] = ''

    #rkadtv = dfjp[dfjp.eligible.isnull()].groupby('date')['adtv_usd'].rank(method='first', ascending=1, pct=True)
    #df2.loc[dfjp[dfjp.eligible.isnull()].index,'rkadtv'] = rkadtv
    
    for d in dfjp.sort_values('date', ascending=True)['date'].drop_duplicates():
        dfidx1 = dfjp[(dfjp.date==d) & -(dfjp['60m_stdev'].isnull())].sort_values(['60m_stdev','ffmcap'], ascending=[True, False])[['60m_stdev','ffmcap']].index
        rk1 = np.arange(len(dfjp[(dfjp.date==d) & -(dfjp['60m_stdev'].isnull())])-1, -1, -1)
        dfjp.loc[dfidx1, 'rkstdev'] = rk1
        dfjp.loc[dfidx1, 'rkstdev'] = dfjp.loc[dfidx1, 'rkstdev'].map(lambda x: stats.percentileofscore(rk1, x, kind='strict')/100.)
        dfjp.loc[dfidx1, 'rkstdev'] = dfjp.loc[dfidx1, 'rkstdev'] / max(dfjp.loc[dfidx1, 'rkstdev'])

        dfidx1 = dfjp[(dfjp.date==d) & -(dfjp['accruals'].isnull())].sort_values(['accruals','ffmcap'], ascending=[True, False])[['accruals','ffmcap']].index
        rk1 = np.arange(len(dfjp[(dfjp.date==d) & -(dfjp['accruals'].isnull())])-1, -1, -1)
        dfjp.loc[dfidx1, 'rkaccr'] = rk1
        dfjp.loc[dfidx1, 'rkaccr'] = dfjp.loc[dfidx1, 'rkaccr'].map(lambda x: stats.percentileofscore(rk1, x, kind='strict')/100.)
        dfjp.loc[dfidx1, 'rkaccr'] = dfjp.loc[dfidx1, 'rkaccr'] / max(dfjp.loc[dfidx1, 'rkaccr'])

        dfidx1 = dfjp[(dfjp.date==d) & -(dfjp['adtv_usd'].isnull())].sort_values(['adtv_usd','ffmcap'], ascending=[False, False])[['adtv_usd','ffmcap']].index
        rk1 = np.arange(len(dfjp[(dfjp.date==d) & -(dfjp['adtv_usd'].isnull())])-1, -1, -1)
        dfjp.loc[dfidx1, 'rkadtv'] = rk1
        dfjp.loc[dfidx1, 'rkadtv'] = dfjp.loc[dfidx1, 'rkadtv'].map(lambda x: stats.percentileofscore(rk1, x, kind='strict')/100.)
        dfjp.loc[dfidx1, 'rkadtv'] = dfjp.loc[dfidx1, 'rkadtv'] / max(dfjp.loc[dfidx1, 'rkadtv'])
        
    #assign 0.5 rank to null vola values
    dfjp['rkstdev']= dfjp['rkstdev'].fillna(0.5)
    dfjp['rkaccr']= dfjp['rkaccr'].fillna(0.5)

    #screening - applyied up front
    idxout = dfjp[(dfjp.rkstdev<0.1) | (dfjp.rkaccr<0.1) | (dfjp.rkadtv<0.05)].index
    dfjp.loc[idxout, 'screen'] = 'out'    

    for d in dfjp.sort_values('date', ascending=True)['date'].drop_duplicates():
    ##normalized factors   
        for factor in factors:
            indexfactor = dfjp[(dfjp.date==d) & (dfjp.screen!='out')][['sedol', factor]].index
            bpr_stdz = two_stage_standardisation(dfjp[(dfjp.date==d) & (dfjp.screen!='out')][['sedol',factor]], 4, 10)
            #dfjp.loc[indexfactor,'n' + factor] = bpr_stdz.loc[indexfactor, 'n' + factor]

            dftemp = dfjp[(dfjp.date==d) & (dfjp.screen!='out')][['sedol', factor]].copy()
            dftemp = pd.merge(dftemp, bpr_stdz, on='sedol', how='left')
            dftemp.index = dfjp[(dfjp.date==d) & (dfjp.screen!='out')].index
            
            #dfjp.loc[indexfactor, factor + '_median_step1'] = dftemp['median_step1']
            #dfjp.loc[indexfactor, factor + '_stdev_step1'] = dftemp['stdev_step1']
            #dfjp.loc[indexfactor, factor + '_lb'] = dftemp['lb']
            #dfjp.loc[indexfactor, factor + '_ub'] = dftemp['ub']
            #dfjp.loc[indexfactor, factor + '_median_step2'] = dftemp['median_step2']
            #dfjp.loc[indexfactor, factor + '_stdev_step2'] = dftemp['stdev_step2']
            
            dfjp.loc[indexfactor,'n' + factor] = dftemp['n' + factor]
            

    dfjp.loc[(dfjp[dfjp.nCFPR.isnull() & (dfjp.screen!='out')]).index, 'nCFPR'] = -4 #change 8 June per request of MUTB (zero before)
    dfjp.loc[(dfjp[dfjp.nBPR.isnull() & (dfjp.screen!='out')]).index, 'nBPR'] = -4
    dfjp.loc[(dfjp[dfjp.nEPR.isnull() & (dfjp.screen!='out')]).index, 'nEPR'] = -4

    dfjp.loc[dfjp[dfjp.country.isin(get_reg_filter('Europe'))].index, 'region'] = 'Europe'
    dfjp.loc[dfjp[dfjp.country.isin(get_reg_filter('America'))].index, 'region'] = 'America'
    dfjp.loc[dfjp[dfjp.country.isin(get_reg_filter('AsiaPac'))].index, 'region'] = 'AsiaPac'

    #composite factor
    dfjp['comp_factor'] = (dfjp['nCFPR'] + dfjp['nBPR'] + dfjp['nEPR']) / 3

    factor_matrix = dfjp[dfjp.screen!='out'].groupby(['date','region','ICB_ind']).mean()['comp_factor']
    for fmx in factor_matrix.index:
        idx_adj = dfjp[(dfjp.screen!='out') & (dfjp.date==fmx[0]) & (dfjp.region==fmx[1]) & (dfjp.ICB_ind==fmx[2])].index
        
        #dfjp.loc[idx_adj, 'ave_ind_reg'] = factor_matrix[fmx]
        dfjp.loc[idx_adj, 'adj_comp_factor'] = dfjp.loc[idx_adj, 'comp_factor'] - factor_matrix[fmx]

    dfjp['value_score'] = dfjp['adj_comp_factor'].map(lambda x: 1 / (1 + exp(-x)))
    
    #--------------------------------------------
    #selection
    
    ct = 0
    for d in dfjp.sort_values('date', ascending=True)['date'].drop_duplicates():
        ct=ct+1
        if ct==1:
            dfjp.loc[dfjp[dfjp.date==d].index,'old'] = False
        else:
            #mark current components
            dfnew = dfjp[(dfjp.date==d)][['infocode']] #take all current candidates
            dfnew = pd.merge(dfnew, dfold, how='left', on='infocode') #link old (column new)
            dfnew['new'] = dfnew['new'].fillna(False)
            listold = dfnew['new'].tolist()
            dfjp.loc[dfjp[dfjp.date==d].index,'old'] = listold          
        
        idx2rk = dfjp[(dfjp.date==d) & -dfjp.value_score.isnull() & ((dfjp.old==True) | ((dfjp.old==False) & (dfjp.rkadtv>=0.2)))].index
        idxranked = dfjp.loc[idx2rk,:].sort_values(['value_score','ffmcap'], ascending=[False, False])[['value_score','ffmcap']].index
        rk1 = np.arange(1, len(dfjp.loc[idx2rk,:])+1, 1)
        dfjp.loc[idxranked, 'final_rank'] = rk1
        
        dfjp.loc[dfjp[dfjp.final_rank<=component_number].index, 'new'] = True
        dfold=dfjp[(dfjp.date==d) & (dfjp.new==True)][['infocode','new']]
        
#old selection - no 
#
#        dfidx1 = dfjp[(dfjp.date==d) & -dfjp.value_score.isnull()].sort_values(['value_score','ffmcap'], ascending=[False, False])[['value_score','ffmcap']].index
#        rk1 = np.arange(1, len(dfjp[(dfjp.date==d) & -dfjp.value_score.isnull()])+1, 1)
#        dfjp.loc[dfidx1, 'final_rank'] = rk1
#
#    dfjp['new'] = False
#    dfjp.loc[dfjp[dfjp.final_rank<=component_number].index, 'new'] = True
    
    #--------------------------------------------
    
    #weighting
    sum_value_score = dfjp[dfjp.new==True].groupby('date').sum()['value_score']

    for i in range(len(sum_value_score)):
        wgt1 = dfjp.loc[dfjp[(dfjp.date==sum_value_score.index[i]) & (dfjp.new==True)].index, 'value_score'] / sum_value_score[i]
        dfjp.loc[wgt1.index,'weight'] = wgt1

    dfjp = dfjp.sort_values(['date', 'weight','final_rank'], ascending = [True, False,True]).reset_index(drop=True)
    
    #del dfjp['weight'] #temporary
    
    dfjp.drop_duplicates().to_csv(calcloc + '07_' + f + '_value_selection.csv', sep=';', index=False)
    print('weighting done')

    
t = np.arange(-1., 1., 0.01)

def f(x):    
    return 1. / (1. + exp(-x))
def f_(x):
    return sqrt(x)

vecf = np.vectorize(f)
vecf_ = np.vectorize(f_)

fig, ax = plt.subplots()
ax.plot(t, t, 'r--', label='factor value (x)')
ax.plot(t, vecf(t), 'b-', label='value score:  1 / (1 + exp(-x))')
ax.plot(t, vecf_(t), 'g-', label='sqrt')
ax.plot(t, [stats.percentileofscore(t, x, kind='strict')/100. for x in t], 'k-', label='value perc_rank')
legend = ax.legend(loc='lower right', shadow=True)


#loop through files in directory
import os
import pandas as pd
lst=[]
ndir = 'U:/Documents/IPython Notebooks/'
for filename in os.listdir(ndir):
    lst.append(filename)

    
#    ------------------------------------------

import pandas as pd
import time
import datetime as dt
import requests
import sys
import numpy as np
import scipy.stats as stats
%pylab inline

loc = 'S:/Stoxx/Product Development and Research/Team/Antonio/Presentations/'
sys.path.append(loc)

def calccapfacs(df_comp):
    """
    received a df with -> column0:weight ;column1=cap; returns df with additional column2:capfactor; colum3:cappedwgt
    reindexes the df starting with 1
    """

    df_comp = df_comp.sort_values(df_comp.columns[0],ascending=False)
    df_comp.index = range(1,len(df_comp)+1)
    df_comp['capfactor']=1
    if sum(df_comp.iloc[:,1])<=1.:   
        df_comp['cappedwgt'] = 1. / len(df_comp) #equal weight
    else:
        df_comp['cappedwgt'] = df_comp.iloc[:,0]
        while len(df_comp[np.round(df_comp.cappedwgt, 7) > np.round(df_comp.iloc[:,1], 7)]) > 0:
            dblToCap = df_comp[df_comp.cappedwgt >= df_comp.iloc[:,1]].cap.sum()
            weightsnocap = df_comp[df_comp.cappedwgt < df_comp.iloc[:,1]].cappedwgt.sum()
            dblDistFactor = weightsnocap / (1 - dblToCap)
            for index, row in df_comp.iterrows():
                if row['cappedwgt'] >= row[1]: 
                    df_comp.loc[index,'cappedwgt'] = dblDistFactor * row[1]
            dblcappedsum = df_comp.cappedwgt.sum()
            df_comp['cappedwgt'] = df_comp['cappedwgt'] / dblcappedsum
    df_comp['capfactor']=(df_comp['cappedwgt']/df_comp.iloc[:,0])/max(df_comp['cappedwgt']/df_comp.iloc[:,0])
    return df_comp.reset_index(drop=True)

#selection
df = pd.read_excel(loc + '00_portfolio_sample.xlsx', sheetname='pf2')
df['date'] = df['date'].map(lambda x: pd.to_datetime(str(x)[:10], format='%Y-%m-%d', dayfirst=True))

time1=time.time()

#def get_score(series):
#    if isnull(series[0]) or if isnull(series[0]) or if isnull(series[0]):
#        return null
#df['score'] = df[['nPB', 'nPCF', 'nPE']].apply(get_score, axis=1)

rkadtv = df.groupby('date')['adtv_usd'].rank(method='first', ascending=True, pct=True)
df.loc[:,'rkadtv'] = rkadtv

df = df[df.ffmcap>=1000000000] #mcap filter
df = df[df.rkadtv>=0.05] #mcap filter

mean_pb = df.groupby('date').mean()['PB']
mean_pcf = df.groupby('date').mean()['PCF']
mean_pe = df.groupby('date').mean()['PE']

stdev_pb = df.groupby('date').std()['PB']
stdev_pcf = df.groupby('date').std()['PCF']
stdev_pe = df.groupby('date').std()['PE']

for i in range(len(mean_pb)):
    #df.loc[df[df.date==d].index,'nPB'] = df.loc[df[df.date==d].index,'PB'].map(lambda x: (x - mean_pb)/x.stdev(ddof=1))
    idx = df[df.date==mean_pb.index[i]].index
    df.loc[idx,'nPB'] = (df.loc[idx,'PB'] - mean_pb[i]) / stdev_pb[i]
    df.loc[idx,'nPCF'] = (df.loc[idx,'PCF'] - mean_pcf[i]) / stdev_pcf[i]
    df.loc[idx,'nPE'] = (df.loc[idx,'PE'] - mean_pe[i]) / stdev_pe[i]
    
df['score'] = (df['nPB'] + df['nPCF'] + df['nPE']) / 3

#selection
component_number = 100
ct = 0

for d in df.sort_values('date', ascending=True)['date'].drop_duplicates():

    ct=ct+1
    if ct==1:
        df.loc[df[df.date==d].index,'old'] = False
    else:
        #mark current components
        dfnew = df[(df.date==d)][['isin']] #take all current candidates
        dfnew = pd.merge(dfnew, dfold, how='left', on='isin') #link old (column new)
        dfnew['new'] = dfnew['new'].fillna(False)
        listold = dfnew['new'].tolist()
        df.loc[df[df.date==d].index,'old'] = listold          

        
    idx2rk = df[(df.date==d) & ((df.old==True) | ((df.old==False) & (df.rkadtv>=0.2)))].index
    idxranked = df.loc[idx2rk,:].sort_values(['score','ffmcap'], ascending=[False, False])[['score','ffmcap']].index
    rk1 = np.arange(1, len(df.loc[idx2rk,:])+1, 1)
    df.loc[idxranked, 'final_rank'] = rk1
    
    df.loc[(df[(df.date==d) & (df.final_rank<=component_number)].index), 'new'] = True
    dfold=df[(df.date==d) & (df.new==True)][['isin','new']]

#weighting
df['wgt']=0
df['cap']=0.02
sumcap = df[df.new==True].groupby('date').sum()['ffmcap']
ctcomps = df[df.new==True].groupby('date').count()['isin']

for i in range(len(sumcap)):
    wgt = df.loc[df[(df.new==True) & (df.date==sumcap.index[i])].index, 'ffmcap'] / sumcap[i]
    df.loc[wgt.index,'wgt'] = wgt
    dfftrs = calccapfacs(df.loc[df[(df.new==True) & (df.date==sumcap.index[i])].index,['wgt','cap']])
    dfftrs.index = wgt.index #reindex
    df.loc[wgt.index,'weight'] = dfftrs['cappedwgt']

df = df.sort_values(['date','weight'], ascending=[True, False])
df.to_excel(loc + '00_portfolio_results.xlsx', index=True)
time2 = time.time()
print('time elapsed:', (time2-time1), 'seconds')

print(len(df))
df.head()


# ----------------------------------

PB = df.groupby('date').apply(lambda x: (x.PB*x.weight).sum())
PCF = df.groupby('date').apply(lambda x: (x.PCF*x.weight).sum())
dffund = pd.concat([PB,PCF], axis=1)
dffund.columns = ['Price-Book', 'Price-CashFlow']
dffund['date']= dffund.index
cols=dffund.columns.tolist()
dffund=dffund[cols[-1:] + cols[:-1]] 
dffund.index.name = None
dffund.plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.suptitle('Fundamentals', fontsize=15, fontweight='bold')
dffund.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().autofmt_xdate()

#_______________________________________

ct=0
for d in df.iloc[:,:].sort('date', ascending=True)['date'].drop_duplicates():
    print(d)
    gr = df[df.date==d].groupby('country').mean()
    #gr = gr.loc[:,'PB'].copy()
    gr.index.name = None
    if ct==0:
        dfsum = pd.DataFrame(gr.loc[:,'PB'])
    else:
        dfsum = pd.concat([dfsum, pd.DataFrame(gr.loc[:,'PB'])], axis=1)

    dfsum = dfsum.rename(columns={'PB':str(d)[:10]})
    ct +=1

dfsum.T.plot()
plt.suptitle('Historical PB by country', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(rotation=45)
