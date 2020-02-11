import pandas as pd
import numpy as np
import datetime as dt
from quant.history import history_tools as ht
from quant.antonio import helpers as h


def get_ret_source(perf_source):
    ''' loads performance for 'msci_acwi', 'all_assessed', 'all_extended'
    '''
    perf_loc = 'S:/products and engineering - 0048187/Antonio/Python/quant/sources/data/'
    perf_file = perf_loc + h.find_last_file(perf_source, perf_loc)
    dfperf = pd.read_csv(perf_file, sep=';')
    dfperf = dfperf.T
    dfperf.columns = dfperf.iloc[0, :]
    dfperf = dfperf.iloc[1:]
    dfperf = dfperf / 100
    dfperf[dfperf > 50] = np.nan  # assumes >50x monthly returns are errors
    dfperf.columns.name = None
    #dfperf.columns = dfperf.columns.map(lambda x: x.replace(' BBGID', ''))
    dfperf.index = pd.to_datetime(dfperf.index, format='%Y%m%d', dayfirst=True)

    if len(dfperf.columns[dfperf.columns.isnull()]) > 0:
        dfperf = dfperf[dfperf.columns[~dfperf.columns.isnull()]]
    return dfperf

def combine_ret_source():
    dfaa = get_ret_source('all_assessed')
    dfacwi = get_ret_source('msci_acwi')
    dfadd = dfacwi[dfacwi.columns[~dfacwi.columns.isin(list(dfaa.columns))]]
    dfperf = pd.concat([dfaa, dfadd], axis=1)

    return dfperf

def calc_rets_pf(df, dfperf, nulls_2_zero, pf_name='mypf', bbgid_field_name='ID_BB_GLOBAL', attrfld=''):
    '''backtests portfolio using provided returns. it leaves components with unmapped return with their weight
    inputs:
    df - dataframe with column[0]: 'Date', format YYYY-MM-DD; column[1]: bbgid_field_name; column[2: ]: 'WEIGHT'
    dfperf - returns in the format returned by get_all_assessed_m_rets()
    pf_name - portfolio name
    bbgid_field_name - name of the BBG id that will map with the returns

    output:
    dataframe[0] - portfolio returns
    dataframe[1] - list of unmapped components
    dataframe[2] - turnover
    dataframe[3] - return attribution based attrfld
    dataframe[4] - return based attrfld (absolute)
    dataframe[5] - attrfld criteria weights
    '''

    datelist = sorted(df.Date.drop_duplicates())

    dfdata = pd.DataFrame()
    dfdatagrp = pd.DataFrame()
    dfdatagrpabs = pd.DataFrame()
    dfdatagrpwgt = pd.DataFrame()
    dfmiss = pd.DataFrame()


    dta = []
    dfweight = pd.DataFrame()
    dfweightold = pd.DataFrame()

    for d in datelist[:]:
        # define dates
        try:
            nextfiledate = [x for x in datelist if x > d][0]
        except:
            nextfiledate = dt.date.today()
            nextfiledate = pd.to_datetime(nextfiledate.strftime('%Y-%m-%d'), format='%Y-%m-%d',
                                          dayfirst=True)  # convert datetime to timestamp

        dfport = df[df.Date == d].reset_index(drop=True).copy()
        dfret = dfperf[dfperf.columns[dfperf.columns.isin(list(dfport[bbgid_field_name]))]].copy()  # select IDs
        dfret = dfret[(dfret.index > d) & (dfret.index <= nextfiledate)].copy()  # select dates

        dfret = dfret.T
        dfret[bbgid_field_name] = dfret.index

        if attrfld != '':
            column_lst = attrfld.copy()
            column_lst.append(bbgid_field_name)
            dfret_grp = pd.merge(dfret, dfport[column_lst].drop_duplicates(), on=bbgid_field_name, how='left')  # group

        if len(dfweight) > 0:
            dfweightold = dfweight.copy()
            dfweightold.columns = [bbgid_field_name, 'WEIGHT_OLD']

        dfweight = dfport[[bbgid_field_name, 'WEIGHT']].copy()

        if len(dfweightold) > 0:
            # calculate turnover
            x = pd.merge(dfweightold[-dfweightold[bbgid_field_name].isnull()], 
                         dfweight[-dfweight[bbgid_field_name].isnull()], how='outer', on=bbgid_field_name)
            count_in = len(x[x.WEIGHT_OLD.isnull()])
            count_out = len(x[x.WEIGHT.isnull()])
            count_total = len(x[-x.WEIGHT.isnull()])
            x.iloc[:, -2:] = x.iloc[:, -2:].fillna(0)
            to = (np.absolute(x['WEIGHT_OLD'] - x['WEIGHT'])).sum() / 2.
            dta.append([str(d)[:4], d, np.around(to, 6), count_out, count_in, count_total])

        try:
            #get misssing data
            dfmiss_ = pd.merge(dfweight, dfret[[bbgid_field_name, dfret.T.index[0]]].copy(), on=bbgid_field_name,
                               how='left')
            dfmiss_ = dfmiss_[dfmiss_[dfret.T.index[0]].isnull()][[bbgid_field_name]].reset_index(drop=True).copy()
            dfmiss_['Date'] = dfret.T.index[0]
            dfmiss = pd.concat([dfmiss, dfmiss_], axis=0).reset_index(drop=True)
        except:
            pass

        lstret = []
        lstretgrp = []
        lstretgrpabs = []
        lstretgrpwgt = []
        for col in dfret.columns[:-1]:
            df_ = pd.merge(dfweight, dfret[[bbgid_field_name, col]].copy(), on=bbgid_field_name, how='left')

            if nulls_2_zero:
                df_[col] = df_[col].fillna(0) # -------------------------------------------------------------------> here
                
            lstret.append((df_['WEIGHT'] * df_[col]).sum())
            df_['WEIGHT'] = df_['WEIGHT'] * (1 + df_[col])
            df_['WEIGHT'] = df_['WEIGHT'] / df_['WEIGHT'].sum()

            #grouped
            if attrfld != '':
                column_lst = attrfld.copy()
                column_lst.append(bbgid_field_name)
                column_lst.append(col)
                df_grp = pd.merge(dfweight, dfret_grp[column_lst].drop_duplicates().copy(), on=bbgid_field_name, how='left')

                df_grp[col] = df_grp[col].fillna(0)
                df_grp['wgtret'] = df_grp['WEIGHT'] * df_grp[col]
                lstretgrp.append(df_grp.groupby(attrfld)['wgtret'].sum())
                lstretgrpabs.append(df_grp.groupby(attrfld)['wgtret'].sum() / df_grp.groupby(attrfld)['WEIGHT'].sum())
                lstretgrpwgt.append(df_grp.groupby(attrfld)['WEIGHT'].sum())

            dfweight = df_[[bbgid_field_name, 'WEIGHT']]

        dfpfret = pd.DataFrame(lstret, index=dfret.T.index[:-1])  # put returns into df
        dfpfretgrp = pd.DataFrame(lstretgrp, index=dfret.T.index[:-1])
        dfpfretgrpabs = pd.DataFrame(lstretgrpabs, index=dfret.T.index[:-1])
        dfpfretgrpwgt = pd.DataFrame(lstretgrpwgt, index=dfret.T.index[:-1])

        dfdata = pd.concat([dfdata, dfpfret], axis=0)
        dfdatagrp = pd.concat([dfdatagrp, dfpfretgrp], axis=0)
        dfdatagrpabs = pd.concat([dfdatagrpabs, dfpfretgrpabs], axis=0)
        dfdatagrpwgt = pd.concat([dfdatagrpwgt, dfpfretgrpwgt], axis=0)

    dfto = pd.DataFrame(dta, columns=['year', 'date', '1way_turnover', 'count_out', 'count_in', 'count_total'])
    dfdata.columns = [pf_name]

    dfmiss[bbgid_field_name] = dfmiss[bbgid_field_name].fillna(0)

    return dfdata, dfmiss, dfto, dfdatagrp, dfdatagrpabs, dfdatagrpwgt


def rets_to_ts(df1, basedate, baseval=100):
    '''calculates time series based on returns (format: output file from calc_rets_pf)
    inputs: df1 - dataframe in format calc_rets_pf()
    output: dataframe with time series added to h_ column; base date and value are added
    '''

    df1 = pd.concat([pd.DataFrame(index=[basedate], columns=list(df1.columns)), df1], axis=0)
    df1 = df1.fillna(0)

    basevals = [baseval for i in df1.columns]
    vals = []

    for i in range(0, len(df1)):
        val = basevals * (1 + df1.iloc[i, :])
        vals.append(val)
        basevals = val

    df1.loc[:, df1.columns] = vals

    return df1


def backtest_pf(dfportfolio, basedate, wgting='ew', perf_source='acwi', baseval=100, pf_name='mypf',
                bbgid_field_name='ID_BB_GLOBAL', attrfld='', dfperf='', nulls_2_zero=True):
    '''
    dfportfolio: dataframe with column[0]: 'Date', format YYYYMMDD; column[1]: bbgid_field_name; column[2: ]: 'WEIGHT'
    basedate: date in format YYYYMMDD;
    wgting: column to be weighted
    perf_source: aa - All Assessed; acwi (or any else) - MSCI ACWI;
    return: backtest

    output:
    dataframe[0] - portfolio time series
    dataframe[1] - returns
    dataframe[2] - list of unmapped components
    dataframe[3] - turnover
    dataframe[4] - return attribution based attrfld
    dataframe[5] - return based attrfld (absolute)
    dataframe[6] - attrfld criteria weights
    '''

    dfportfolio = dfportfolio.reset_index(drop=True).copy()

    basedate = pd.to_datetime(basedate, format='%Y-%m-%d', dayfirst=True)

    if len(dfperf)==0:

        if perf_source=='aa':
            dfperf = get_ret_source('all_assessed')
        elif perf_source=='both':
            dfperf = combine_ret_source()
        elif perf_source=='ext':
            dfperf = get_ret_source('all_extended')
        else:
            dfperf = get_ret_source('msci_acwi')


    dfportfolio = ht.wgt_field(dfportfolio, wgting)

    dfcalc = calc_rets_pf(dfportfolio, dfperf, nulls_2_zero, pf_name, bbgid_field_name, attrfld)
    dfrets = dfcalc[0]
    dfmiss = dfcalc[1]
    dfto = dfcalc[2]
    dfdatagrp = dfcalc[3]
    dfdatagrpabs = dfcalc[4]
    dfdatagrpwgt = dfcalc[5]

    return rets_to_ts(dfrets, basedate, baseval), dfrets, dfmiss, dfto, dfdatagrp, dfdatagrpabs, dfdatagrpwgt


def backtest_pf_ntiles(df, ntile_fld, nbins, basedate, wgting, perf_source='acwi', pf_name='pf',
                       bbgid_field_name='ID_BB_GLOBAL', group_fld='Date', dfperf='', nulls_2_zero=True):
    '''backtests portfolio ntiles (1 is the largerst top bbgid_field_name numbers)
    df: dataframe with data
    ntile_fld: field to be used for ntil
    nbins: number of bins
    wgting: df column to be weighted. default - equal weight (ew)
    group_fld: grouping field (id multiple fields, use []); default 'Date'
    '''

    pd.options.mode.chained_assignment = None
    dfntiles = df.copy()
    dfntiles = ht.get_bins(dfntiles, ntile_fld, nbins, group_fld)

    dfts = pd.DataFrame()
    dfrets = pd.DataFrame()
    for bn in sorted(dfntiles[-dfntiles.bin.isnull()].bin.drop_duplicates()):
        dfntl = dfntiles[dfntiles.bin == bn]
        dfntl['ew'] = 1
        dfntl = ht.wgt_field(dfntl, wgting)
        ts = backtest_pf(dfntl, basedate, wgting=wgting, perf_source=perf_source, bbgid_field_name=bbgid_field_name,
                         pf_name=pf_name + '_' + ntile_fld + '_' + str(int(bn)), dfperf=dfperf, nulls_2_zero=nulls_2_zero)

        dfts = pd.concat([dfts, ts[0]], axis=1)
        dfrets = pd.concat([dfrets, ts[1]], axis=1)

    return dfts, dfrets