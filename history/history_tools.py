import pandas as pd
import numpy as np
import datetime as dt
from quant.history import bt
from quant.antonio import helpers as hlp

def wgt_alloc_bm_key(dfport, dfbm, key, wgtcol, dtcol):

    dfport = wgt_field(dfport, wfld=wgtcol, dtfld=dtcol)
    a = wgt_field(dfbm, wfld=wgtcol, dtfld=dtcol).groupby(key).sum()['WEIGHT'] #bm
    b = dfport.groupby(key).sum()['WEIGHT']
    x = pd.concat([a,b], axis=1)
    x = x.fillna(0)
    fctr = x.iloc[:, 0] / x.iloc[:, 1]
    t = pd.DataFrame(fctr)
    t.columns = ['factor']
    dfport = pd.merge(dfport, t, left_on=key, right_index=True)
    dfport['wgt_alloc_bm'] = dfport['WEIGHT'] * dfport['factor']
    del dfport['factor']
    del dfport['WEIGHT']

    sumcap = dfport.groupby(dtcol).sum()['wgt_alloc_bm']

    for i in range(len(sumcap)):
        wgt = dfport.loc[dfport[dfport[dtcol] == sumcap.index[i]].index, 'wgt_alloc_bm'] / sumcap[i]
        dfport.loc[wgt.index, 'wgt_alloc_bm'] = wgt

    return dfport

def get_fx(fx='eurusd'):
    perf_loc = 'S:/products and engineering - 0048187/Antonio/Python/quant/sources/data/'
    dffx = pd.read_csv(perf_loc + hlp.find_last_file(fx, perf_loc), sep=';')
    dffx = dffx.rename(columns={dffx.columns[0]: 'Date'})
    dffx['Date'] = pd.to_datetime(dffx['Date'], format='%Y-%m-%d', dayfirst=True)
    return dffx


def convert_to_cur(df, fx='eurusd', rebase=True):
    dffx = get_fx(fx)
    df['Date'] = df.index
    dffx = pd.merge(dffx, df[['Date']], on='Date', how='outer').sort_values('Date').reset_index(drop=True)
    dffx[dffx.columns[1]] = dffx[dffx.columns[1]].fillna(method='pad')

    df = pd.merge(df, dffx, on='Date', how='left')

    for col in df.columns[:-2]:
        df[col] = ((df[col]) / (df[dffx.columns[1]]))
    df = df[df.columns[:-1]]
    df.index = df.Date
    df.index.name = None
    del df['Date']

    if rebase:
        df = df / df.iloc[0, :] * 100

    return df

def merge_daily2monthly(dfd, dfm):

    dfd['Date'] = dfd.index
    dfm['Date'] = dfm.index
    dfd = pd.merge(dfd, dfm[['Date']], on='Date', how='outer').sort_values('Date').reset_index(drop=True)
    dfd = dfd.fillna(method='pad')
    dfm = pd.merge(dfm, dfd, on='Date', how='left')
    dfm.index = dfm.Date
    del dfm['Date']
    dfm.index.name = None

    return dfm

def ann_ret_2_per_ret(ret_float, freq=12):
    return (1 + ret_float)**(1/freq) - 1

def deduct_ret(dfs, ret_float, freq=12):
    
    moret = ann_ret_2_per_ret(ret_float)
    nreturns = np.array(dfs.iloc[1:len(dfs), :]) / np.array(dfs.iloc[0:len(dfs) - 1, :]) - 1

    df1 = pd.DataFrame((nreturns - moret), index=dfs.index[1:], columns=dfs.columns)
    basedate = str(dfs.index[0])[:10]

    df2 = pd.DataFrame()
    for c in df1.columns:
        df2_ = bt.rets_to_ts(df1[[c]], basedate, baseval=100)
        df2 = pd.concat([df2, df2_], axis=1)
    df2.index = dfs.index
    df2.columns=dfs.columns
    return df2
	
def get_bm(basedate='', curr='usd', baseval=100):
    dfixperf = bt.get_ret_source('benchmark_net_returns_'+curr)
    if basedate == '':
        basedate = str(list(pd.bdate_range(end=dfixperf.index[0], periods=2, freq='M'))[0])[:10]
    dfixperf = dfixperf[dfixperf.index > basedate]
    dfixperf = dfixperf.fillna(0)
    basedate = pd.to_datetime(basedate, format='%Y-%m-%d', dayfirst=True)

    dfrs = pd.DataFrame()
    for c in dfixperf.columns:
        dfx = dfixperf[[c]]
        dfr = bt.rets_to_ts(dfx, basedate, baseval)
        dfrs = pd.concat([dfrs, dfr], axis=1)

    #workaround
    #dfrs.loc[dfrs[dfrs.index<='2013-03-31'].index,'h_DJSWICLN Index_default_curr'] = np.nan

    return dfrs

def wgt_field(df1, wfld='ew', dtfld='Date'):
    '''wgtfld - name of the field to be weighted
       dtfld - name of the date column to group, defalut 'Date'
    '''

    df1 = df1.reset_index(drop=True).copy()

    df1['WEIGHT'] = 0
    if wfld=='ew':
        df1['ew'] = 1
    sumcap = df1.groupby(dtfld).sum()[wfld]

    # mcap weight
    for i in range(len(sumcap)):
        wgt = df1.loc[df1[df1.Date == sumcap.index[i]].index, wfld] / sumcap[i]
        df1.loc[wgt.index, 'WEIGHT'] = wgt

    #df1 = df1.sort_values([dtfld, 'WEIGHT'], ascending=[True, False]).reset_index(drop=True)

    #df1[dtfld] = df1[dtfld].map(lambda x: str(x).replace('-', ''))

    return df1


def get_bins(df, ntile_fld, nbins, group_fld=''):
    df_ = df.copy()
    df_ = df_[-df_[ntile_fld].isnull()]

    if group_fld == '':
        df_['bin'] = df_[ntile_fld].transform(lambda x: pd.qcut(x, nbins, range(nbins, 0, -1)))
    else:
        df_ = df_[df_.groupby(group_fld)[ntile_fld].transform('count') > 1]
        df_['bin'] = df_.groupby(group_fld)[ntile_fld].transform(lambda x: pd.qcut(x,
                                                                nbins, range(nbins, 0, -1), duplicates='drop'))

    df['bin'] = df_['bin']
    return df


def calc_stats(dfs, freq, periods=[], bm_col=''):
    dfs.fillna(method='pad', inplace=True)

    dfrom = dfs.index[0]
    dto = dfs.index[len(dfs) - 1]

    lst_dts = [dfrom]
    lst_ct = [len(dfs) - 1]

    years = [0]
    yrs = ['from ' + str(dfrom)[:10]]

    for per in sorted(periods, reverse=True):
        years.append(per)
        yrs.append(str(per) + 'y')

    # get date list and observation count list
    for m in sorted(years, reverse=True):
        dt_ = dfs[dfs.index >= str(hlp.add_months(dto, -12 * m))].index[0]
        ct_ = len(dfs[(dfs.index <= dfs.index[len(dfs) - 1]) & (dfs.index > dt_)])
        lst_dts.append(dt_)
        lst_ct.append(ct_)

    # get portfolio values at the key dates
    dfvals = pd.DataFrame()
    for de in lst_dts:
        dfvals = pd.concat([dfvals, pd.DataFrame(dfs[dfs.index <= de].sort_index(ascending=False).iloc[0, :]).T],
                           axis=0)

    # actual returns
    actret = np.array(dfvals.iloc[len(dfvals) - 1, :]) / np.array(dfvals.iloc[0:len(dfvals) - 1, :]) - 1
    dfactret = pd.DataFrame(actret, columns=dfs.columns, index=yrs)

    # annaulized returns
    dfannret = dfactret.copy()
    for d in range(len(lst_dts) - 1):
        dfannret.loc[yrs[d], :] = dfannret.loc[yrs[d], :].map(lambda x: (x + 1) ** (freq / lst_ct[d]) - 1)

    # vola/sharpe ratio/TE/IR
    nreturns = np.array(dfs.iloc[1:len(dfs), :]) / np.array(dfs.iloc[0:len(dfs) - 1, :]) - 1
    dfr = pd.DataFrame(nreturns)
    dfr.columns = dfs.columns
    dfr.index = dfs.index[1:]

    annvol = [];
    meanret = [];
    shrp = [];
    annTE = [];
    meanexcret = [];
    IR = [];
    upcap = [];
    downcap = []
    for d in range(len(lst_dts) - 1):

        returns = dfr[(dfr.index <= lst_dts[len(lst_dts) - 1]) & (dfr.index > lst_dts[d])]
        vol = np.std(returns, ddof=1)
        annvol.append(vol * np.sqrt(freq))
        shrp.append((returns.mean() / vol) * np.sqrt(freq))
        meanret.append(returns.mean() * freq)

        if bm_col != '':
            exreturn = returns.copy()
            exreturnbm = exreturn[bm_col].copy()
            for col in exreturn.columns:
                exreturn[col] = exreturn[col] - exreturnbm
            TE = np.std(exreturn, ddof=1)
            annTE.append(TE * np.sqrt(freq))
            IR.append((np.around(exreturn.mean(), 6) / np.around(TE, 6)) * np.sqrt(freq))
            meanexcret.append(exreturn.mean() * freq)

            upcapture = returns[returns[bm_col] > 0]
            downcapture = returns[returns[bm_col] < 0]

            try:
                # upcap.append([((1 + upcapture[col]).prod()**(1/freq)-1)/
                # ((1 + upcapture[bm_col]).prod()**(1/freq)-1) for col in upcapture.columns])
                upcap.append([(upcapture[col]).mean() / (upcapture[bm_col]).mean() for col in upcapture.columns])
            except:
                upcap.append([np.nan for col in upcapture.columns])

            try:
                # downcap.append([((1 + downcapture[col]).prod()**(1/freq)-1) /
                # ((1 + downcapture[bm_col]).prod()**(1/freq)-1) for col in downcapture.columns])
                downcap.append(
                    [(downcapture[col]).mean() / (downcapture[bm_col]).mean() for col in downcapture.columns])
            except:
                downcap.append([np.nan for col in upcapture.columns])

    dfvola = pd.DataFrame(annvol)
    dfvola.columns = dfs.columns
    dfvola.index = yrs

    dfmeanret = pd.DataFrame(meanret)
    dfmeanret.columns = dfs.columns
    dfmeanret.index = yrs

    dfshrp = pd.DataFrame(shrp)
    dfshrp.columns = dfs.columns
    dfshrp.index = yrs

    if bm_col != '':
        dfexret = dfactret.copy()
        dfannexret = dfannret.copy()
        dfexret['b3nchm4rk'] = dfexret[bm_col]
        dfannexret['b3nchm4rk'] = dfannexret[bm_col]
        for col in dfannexret.columns[:-1]:
            dfexret[col] = dfexret[col] - dfexret['b3nchm4rk']
            dfannexret[col] = dfannexret[col] - dfannexret['b3nchm4rk']
        del dfexret['b3nchm4rk']
        del dfannexret['b3nchm4rk']

        dfTE = pd.DataFrame(annTE)
        dfTE.columns = dfs.columns
        dfTE.index = yrs

        dfmeanexcret = pd.DataFrame(meanexcret)
        dfmeanexcret.columns = dfs.columns
        dfmeanexcret.index = yrs

        dfIR = pd.DataFrame(IR)
        dfIR.columns = dfs.columns
        dfIR.index = yrs

        dfupcap = pd.DataFrame(upcap)
        dfupcap.columns = dfs.columns
        dfupcap.index = yrs
        #dfupcap = np.around(dfupcap.astype(np.double), decimals=2)

        dfdowncap = pd.DataFrame(downcap)
        dfdowncap.columns = dfs.columns
        dfdowncap.index = yrs
        #dfdowncap = np.around(dfdowncap.astype(np.double), decimals=2)

    # max drawdown
    mxdd = []
    for d in range(len(lst_dts) - 1):

        dft = dfs[(dfs.index <= lst_dts[len(lst_dts) - 1]) & (dfs.index > lst_dts[d])]
        pk = np.zeros((len(dft), len(dft.columns)))  # peak
        dd = np.zeros((len(dft), len(dft.columns)))  # drawdown
        h = np.array(dft.iloc[:, :])  # history
        pk[0] = h[0]

        for i in range(len(h) - 1):
            mxdd_ = np.zeros(len(dft.columns))
            for j in range(len(dft.columns)):
                pk[i + 1, j] = h[i + 1, j] if h[i + 1, j] > pk[i, j] else pk[i, j]
                dd[i + 1, j] = h[i + 1, j] / pk[i + 1, j] - 1 if h[i + 1, j] < pk[i + 1, j] else 0
                mxdd_[j] = abs(dd[:, j].min())
        mxdd.append(mxdd_)
    dfmaxdd = pd.DataFrame(mxdd, columns=dfs.columns, index=yrs)

    dfres = pd.concat([dfactret, dfannret, dfmeanret, dfvola, dfshrp, dfmaxdd],
                      keys=['Ret actual', 'Ret ann.', 'Ret mean ann.', 'StDev ann.', 'SR ann. rf=0', 'MaxDD'])

    if bm_col != '':
        dfresbm = pd.concat([dfexret, dfannexret, dfmeanexcret, dfTE, dfIR, dfupcap, dfdowncap],
                            keys=['Ex Ret actual', 'Ex Ret ann.', 'Ex Ret mean ann.', 'TE', 'IR', 'Upside Capture',
                                  'Downside Capture'])
        #dfresbm = pd.concat([dfexret, dfannexret, dfmeanexcret, dfTE, dfIR],
        #                    keys=['Ex Ret actual', 'Ex Ret ann.', 'Ex Ret mean ann.', 'TE', 'IR'])

        dfres = pd.concat([dfres, dfresbm], axis=0)

    return dfres


def calc_hit_rate(df1, dfperf, pf_col='pf', bbgid_field_name='ID_BB_GLOBAL', attrfld=''):

    df1 = df1.sort_values('Date').reset_index(drop=True)
    df1.Date = df1.Date.map(lambda x: str(x)[:10])

    wgting = 'ew'
    df = df1.copy()
    df = wgt_field(df, wgting)

    datelist = sorted(df.Date.drop_duplicates())

    dfhit = pd.DataFrame()
    dfgrphit = pd.DataFrame()

    dfweight = pd.DataFrame()

    lsthit = []
    lstgrphit = []

    for d in datelist[:]: #datelist[8:9]
        # define dates
        try:
            nextfiledate = [x for x in datelist if x > d][0]
        except:
            nextfiledate = dt.date.today()
            nextfiledate = pd.to_datetime(nextfiledate.strftime('%Y-%m-%d'), format='%Y-%m-%d', dayfirst=True)

        dfport = df[df.Date == d].reset_index(drop=True).copy()
        dfret = dfperf[dfperf.columns[dfperf.columns.isin(list(dfport[bbgid_field_name]))]].copy()  # select IDs
        dfret = dfret[(dfret.index > d) & (dfret.index <= nextfiledate)].copy()  # select dates
        dfret = dfret.T
        dfret[bbgid_field_name] = dfret.index

        dfweight = dfport[[bbgid_field_name, pf_col]].copy()
        dfweight['cumret'] = 1

        for col in dfret.columns[:-1]:
            df_ = pd.merge(dfweight, dfret[[bbgid_field_name, col]].copy(), on=bbgid_field_name, how='left')
            #df_[col] = df_[col].fillna(0)
            df_['cumret'] = (1 + df_[col]) * df_['cumret']
            dfweight = df_[[bbgid_field_name, pf_col,'cumret']]

        del df_[col]
        totalret = df_['cumret'].mean()
        df_['uni_ret'] = totalret
        df_['exret_vs_uni'] = df_['cumret'] - totalret

        trades_all = len(df_)
        trades_win = len(df_[-df_[pf_col].isnull() & (df_['exret_vs_uni']>=0)]) + len(df_[df_[pf_col].isnull() & (df_['exret_vs_uni']<0)])
        trades_loss = len(df_[df_[pf_col].isnull() & (df_['exret_vs_uni']>=0)]) + len(df_[-df_[pf_col].isnull() & (df_['exret_vs_uni']<0)])
        lsthit.append([d, trades_all, trades_win, trades_loss])

        if attrfld != '':
            column_lst = attrfld.copy()
            column_lst.append(bbgid_field_name)
            df_ = pd.merge(df_, dfport[column_lst].drop_duplicates().copy(), on=bbgid_field_name, how='left')

            for c in dfport.groupby(attrfld).count()[bbgid_field_name].index:
                for j in range(len(attrfld)):
                    ixlist_ = df_[df_[attrfld[j]]==c[j]].index
                    if j==0:
                        ixlist = ixlist_.copy()
                    ixlist = list(set(ixlist).intersection(set(ixlist_)))

                df_.loc[ixlist, 'bucket_ret'] = df_.loc[ixlist, 'cumret'].mean()
                df_.loc[ixlist, 'exret_vs_bucket'] = df_.loc[ixlist, 'cumret']  - df_.loc[ixlist, 'bucket_ret']

            trades_grp_all = len(df_)
            trades_grp_win = len(df_[-df_[pf_col].isnull() & (df_['exret_vs_bucket']>=0)]) + len(df_[df_[pf_col].isnull() & (df_['exret_vs_bucket']<0)])
            trades_grp_loss = len(df_[df_[pf_col].isnull() & (df_['exret_vs_bucket']>=0)]) + len(df_[-df_[pf_col].isnull() & (df_['exret_vs_bucket']<0)])
            lstgrphit.append([d, trades_grp_all, trades_grp_win, trades_grp_loss])


    dfhit = pd.DataFrame(lsthit)
    dfhit.columns = ['Date', 'all', 'win','loss']
    dfhit['hitrate'] = dfhit['win'] / dfhit['loss']
    dfhit['winrate'] = dfhit['win'] / dfhit['all']
    dfhit['lossrate'] = dfhit['loss'] / dfhit['all']

    dfgrphit = pd.DataFrame(lstgrphit)
    dfgrphit.columns = ['Date', 'all', 'win_grp','loss_grp']
    dfgrphit['hitrate_grp'] = dfgrphit['win_grp'] / dfgrphit['loss_grp']
    dfgrphit['winrate_grp'] = dfgrphit['win_grp'] / dfgrphit['all']
    dfgrphit['lossrate_grp'] = dfgrphit['loss_grp'] / dfgrphit['all']

    return dfhit, dfgrphit


def calc_outperf_1vs1(s1, s2, basedate_, title=''):

    x = pd.concat([s1, s2], axis=1)
    x = x[x.index >= basedate_]
    x = x.iloc[:, :] / x.iloc[0, :] * 100
    df = pd.DataFrame(np.array(x.iloc[:, 0]) - np.array(x.iloc[:, 1]))
    df.index = x.index
    df.columns = [title]

    return df


def rebase(x, basedate_, baseval=100):
    x = x[x.index>=basedate_]
    return x.iloc[:,:] / x.iloc[0,:] * baseval


def get_rolled_pf(m0, del_anchor_prev_cam=False):
    '''Adds compnies fom last campaign that don't have an asessment in the current campaing.
    Only works if there's only one campaing data per calc_date
    '''
    i = 0
    m_ = pd.DataFrame()
    for cd in m0.CALC_DATE.drop_duplicates()[:]:
        if i == 0:
            mno = m0[m0.CALC_DATE == cd].iloc[:, :]
        if i > 0:
            cam = list(m0[m0.CALC_DATE == cd]['CAM_YEAR'].drop_duplicates())[0]
            cut_date = list(m0[m0.CALC_DATE == cd]['CUT_DATE'].drop_duplicates())[0]
            mn = m0[m0.CALC_DATE == cd].iloc[:, :]
            mo = m0[m0.CAM_YEAR == cam - 1].iloc[:, :]

            if del_anchor_prev_cam:
                mo['ANCHOR'] = np.nan

            mno = pd.concat([mn, mo.sort_values(['CALC_DATE']).drop_duplicates(subset=['CSF_LCID'], keep='last')],
                            axis=0)
            mno = mno.sort_values(['CALC_DATE']).drop_duplicates(subset=['CSF_LCID'], keep='last')
            mno['CALC_DATE'] = cd
            mno['CUT_DATE'] = cut_date
        m_ = pd.concat([m_, mno], axis=0)
        i += 1

    m_ = m_.reset_index(drop=True)
    return m_


def rollCorrOrBeta(a, b, window, corr=True):
    rollCorr = np.zeros((len(a)))
    window = window - 1
    for i in range(window, len(a)):
        rollCorr[i] = myCorrOrBeta(a[i - window:i + 1], b[i - window:i + 1], corr=corr)
    return rollCorr


def rollTE(a, b, window):
    rollTE = np.zeros((len(a)))
    window = window - 1
    for i in range(window, len(a)):
        rollTE[i] = np.std(a[i - window:i + 1] - b[i - window:i + 1], ddof=1) * np.sqrt(250)
    return rollTE


def rollMean(a, window):
    rollMean = np.zeros((len(a)))
    window = window - 1
    for i in range(window, len(a)):
        rollMean[i] = np.mean(a[i - window:i + 1])
    return rollMean


def rollSdev(a, window):
    rollSdev = np.zeros((len(a)))
    window = window - 1
    for i in range(window, len(a)):
        rollSdev[i] = np.std(a[i - window:i + 1], ddof=1)
    return rollSdev


def myCorrOrBeta(a, b, corr=True):
    l = len(a)
    cov = sum(a * b) / l - sum(a) / l * sum(b) / l
    vara = sum(a * a) / l - sum(a) / l * sum(a) / l
    varb = sum(b * b) / l - sum(b) / l * sum(b) / l

    if corr==True:
        res = float(cov / (np.sqrt(vara) * np.sqrt(varb)))
    else:
        res = float(cov / varb)

    return res


def flag_missing_prices(df, dfperf, bbgid_field_name, idmapcol):

    datelist = sorted(df.Date.drop_duplicates())

    dfdata = pd.DataFrame()

    for d in datelist[:]:
        try:
            nextfiledate = [x for x in datelist if x > d][0]
        except:
            nextfiledate = dt.date.today()
            nextfiledate = pd.to_datetime(nextfiledate.strftime('%Y-%m-%d'), format='%Y-%m-%d', dayfirst=True)

        dfport = df[df.Date == d].reset_index(drop=True).copy()
        dfret = dfperf[dfperf.columns[dfperf.columns.isin(list(dfport[bbgid_field_name]))]].copy()
        dfret = dfret[(dfret.index > d) & (dfret.index <= nextfiledate)].copy()
        dfret = dfret.T
        retcols = dfret.columns
        dfret[bbgid_field_name] = dfret.index

        dfdata_ = pd.merge(dfport[['Date', idmapcol, bbgid_field_name]], dfret, on=bbgid_field_name, how='left')
        dfdata_['MISSING'] = dfdata_[retcols].isnull().apply(lambda x: all(x), axis=1)
        dfdata_ = dfdata_[['Date', idmapcol, 'MISSING']]

        dfdata = pd.concat([dfdata, dfdata_], axis=0)

    df = pd.merge(df, dfdata, on=['Date', idmapcol], how='left')
    return df