import pandas as pd
from scipy import stats
import numpy as np
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helpers as hlp

class AnalysisComponents:

    def plot_pdf(self, data, label=''):
        #data = data[~np.isnan(data)]
        kde = gaussian_kde(list(data))
        dist_space = linspace(min(data), max(data), 100)
        plt.plot(dist_space, kde(dist_space), label=label)
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.2), shadow=True, ncol=1)
        plt.xticks(rotation=45)


    def get_pie_chart(self, df1, grp=['NOD_INDUSTRY'], ct=['CSF_LONGNAME'], plotpie=True):
        dfct = pd.DataFrame(df1.groupby(grp).count()[ct])
        dfct.columns = ['count_all']
        dfct = dfct.sort_values('count_all', ascending=False)
        dfct['perc_all'] = dfct['count_all'] / dfct.count_all.sum()

        if plotpie:
            fracs = dfct.perc_all
            labels = dfct.index
            plt.subplot(gridspec.GridSpec(2, 2)[0, 0], aspect=1)
            plt.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True, radius=4.)

        return dfct

    def bias_test(self, df, gbias, sco, display=True):
        '''
        gbias: group bias, e.g. ['CAM_TYPE','SIZE','REGION4']
        sco: field where bias is present e.g. SZSCORE'

        kstest D stat: rate of convergence.
        at significance 0.05 reject H0 (eq. distr.) if D>0.043
        '''
        if display == True:
            print('\n', hlp.color.BOLD, hlp.color.CYAN, sco, hlp.color.END, '\n')

        sts = []
        for gb in gbias:

            if display == True:
                print(hlp.color.BOLD, hlp.color.RED, gb, 'BIAS', hlp.color.END, '\n')
            gbitems = list(df[-df[gb].isnull()][gb].drop_duplicates())

            dfstats = pd.DataFrame()
            full = np.array(df[(df.DIMENSION.isnull()) & (-df[sco].isnull())][sco])
            sts.append([gb, 'ALL', round(full.mean(), 4), round(full.std(), 4), len(full)])

            for gbi in gbitems:
                if display == True:
                    print(hlp.color.BOLD, gbi, hlp.color.END, '\n')
                gidx = df[(df.DIMENSION.isnull()) & (-df[sco].isnull()) & (df[gb] == gbi)].index
                g = np.array(df.loc[gidx][sco])
                fidx = df[(df.DIMENSION.isnull()) & (-df[sco].isnull()) & (df[gb] != gbi)].index
                f = np.array(df.loc[fidx][sco])
                sts.append([gb, gbi, round(g.mean(), 4), round(g.std(), 4), len(g)])

                if display == True:
                    ##normality check
                    # print('length:', gbi ,':', len(g), '/ rest:', len(f))
                    # print('norm test:', stats.kstest(g, 'norm'))
                    # stats.probplot(g, dist="norm", plot=plt)
                    # plt.show()

                    ##qq plot
                    print('two sample test:', stats.ks_2samp(g, f))
                    # print('', gb, gbi, ' - mean:', round(g.mean(), 4), '; std:', round(g.std(), 4),
                    #      '\n Full sample - mean:', round(full.mean(), 4), '; std:', round(full.std(), 4))

                    q = np.linspace(0, 100, 101)
                    k, ax = plt.subplots()
                    ax.scatter(np.percentile(f, q), np.percentile(g, q), color='b')
                    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
                    plt.ylabel(gb + ' ' + gbi + ' ' + sco)
                    plt.xlabel(gb + ' ' + 'rest ' + sco)
                    plt.title('qq plot - ' + gbi + ' ' + gb + ' vs rest' + '', fontsize=14)
                    plt.show()
                    print('')

            # sts.append([gb, 'ALL', round(full.mean(), 4), round(full.std(), 4), len(full)])

        dfstsall = pd.DataFrame(sts, columns=['group', 'item', 'mean', 'std', 'size'])

        if display == True:
            for gr in dfstsall.group.drop_duplicates():

                dfsts = dfstsall[dfstsall.group == gr]

                jet = plt.get_cmap('jet')
                colors = iter(jet(np.linspace(0, 1, 7)))
                div = dfsts['size'].max() / 1200
                for index, row in dfsts.iterrows():
                    plt.scatter(row['mean'], row['std'], label=row['item'], color=next(colors), s=row['size'] / div)

                plt.xticks(rotation=45)
                plt.xlabel('mean', fontsize=14)
                plt.ylabel('stdev', fontsize=14)
                plt.title(gr, fontsize=14)
                plt.xlim(-dfsts['mean'].abs().max() * 1.2, dfsts['mean'].abs().max() * 1.2)
                lgnd = plt.legend(bbox_to_anchor=(0, 1), loc=2, scatterpoints=1, fontsize=10)
                for handle in lgnd.legendHandles:
                    handle.set_sizes([10])

                plt.show()

        return dfstsall

    def plot_pdf_group(self, df, gbias, sco, plotall=False):

        gbitems = list(df[-df[gbias].isnull()][gbias].drop_duplicates())
        for gbi in gbitems:
            data = np.array(df[(df.DIMENSION.isnull()) & (-df[sco].isnull()) & (df[gbias]==gbi)][sco])
            self.plot_pdf(data, sco + '_' + gbias + '_' + gbi)
        if plotall==True:
            data = np.array(df[(df.DIMENSION.isnull()) & (-df[sco].isnull())][sco])
            self.plot_pdf(data, sco)


    def plot_cigar_chart_(self, dfmod, sco, mapfld='CSF_LCID', datefld='CALC_DATE', lastn=''):
        if lastn == '':
            lastn = len(dfmod)

        ct = 0
        for cd in list(sorted(list(dfmod[datefld].drop_duplicates())))[-lastn - 1:]:
            if ct > 0:
                dfn = dfmod[dfmod[datefld] == cd][[mapfld, sco]].reset_index(drop=True)
                dfn.columns = [mapfld, sco + '_' + str(cd)]

                dfb = pd.merge(dfn, dfo, on=mapfld)

                k, ax = plt.subplots()
                ax.scatter(dfb[dfb.columns[2]], dfb[dfb.columns[1]], color='b', s=1.5)
                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
                plt.xlabel(dfb.columns[2])
                plt.ylabel(dfb.columns[1])
                plt.title(sco + '_' + str(cd), fontsize=14)
                plt.show()

            dfo = dfmod[dfmod[datefld] == cd][[mapfld, sco]].reset_index(drop=True)
            dfo.columns = [mapfld, sco + '_' + str(cd)]

            ct += 1

    def plot_cigar_chart(self, dfmod, sco, mapfld=['CSF_LCID'], datefld='CALC_DATE', lastn=''):

        if lastn == '':
            lastn = len(dfmod)

        mapfld2 = mapfld.copy()
        mapfld2.append(sco)
        ct = 0
        for cd in list(sorted(list(dfmod[datefld].drop_duplicates())))[-lastn - 1:]:
            mapfld3 = mapfld.copy()
            mapfld3.append(sco + '_' + str(cd))
            if ct > 0:
                dfn = dfmod[dfmod[datefld] == cd][mapfld2].reset_index(drop=True)
                dfn.columns = mapfld3
                dfb = pd.merge(dfn, dfo, on=mapfld)
                self.plot_cigar_2cols(dfb, dfb.columns[-1], dfb.columns[-2], title=sco + '_' + str(cd))
                print(np.around(np.mean(abs(dfb[dfb.columns[-1]] - dfb[dfb.columns[-2]])), 4),
                      'avg abs difference')


            dfo = dfmod[dfmod[datefld] == cd][mapfld2].reset_index(drop=True)
            dfo.columns = mapfld3

            ct += 1


    def plot_cigar_2cols(self, dfb, fld1, fld2, title=''):
        k, ax = plt.subplots()
        ax.scatter(dfb[fld1], dfb[fld2], color='b', s=1.5)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
        plt.ylabel(fld2)
        plt.xlabel(fld1)
        plt.title(title)
        plt.show()


    def compare_scores_2files(self, m1, m2, fld1, fld2='', bydate=True, grpkey=['CALC_DATE', 'CSF_LCID']):

        m1 = m1.rename(columns={fld1: fld1 + '_1'})
        if len(fld2) == 0:
            fld2 = fld1 + '_2'
            m2 = m2.rename(columns={fld1: fld2})

        m3 = pd.merge(m1, m2, on=grpkey, how='right')

        if bydate:
            calc_dates = sorted(list(m1.CALC_DATE.drop_duplicates()))
            for cd_ in calc_dates:
                print(cd_)
                self.plot_cigar_2cols(m3[m3.CALC_DATE == cd_], fld1+'_1', fld2)
        else:
            self.plot_cigar_2cols(m3, fld1+'_1', fld2)

            return m3


    def plot_sco_casa_dates(self, dfc, sco1, sco2):

        for cd in list(dfc.CALC_DATE.drop_duplicates()):
            gidx = dfc[(dfc.CALC_DATE == cd) & (-dfc[sco1].isnull()) & (dfc['CAM_TYPE'] == 'CA')].index
            g = np.array(dfc.loc[gidx][sco1])
            g1 = np.array(dfc.loc[gidx][sco2])
            fidx = dfc[(dfc.CALC_DATE == cd) & (-dfc[sco1].isnull()) & (dfc['CAM_TYPE'] == 'SA')].index
            f = np.array(dfc.loc[fidx][sco1])
            f1 = np.array(dfc.loc[fidx][sco2])
            plt.scatter(g1, g, label='CA', color='b', s=0.7)
            plt.scatter(f1, f, label='SA', color='r', s=0.7)
            plt.xlabel(sco2)
            plt.ylabel(sco1)
            plt.title(cd)
            lgnd = plt.legend()
            for handle in lgnd.legendHandles:
                handle.set_sizes([30])
            plt.show()


    def get_casa_jumps(self, dfmod, sco, qs=['06'], jump_type=2, plot_cigar=False):
        '''
        sco: field with score to be evaluated
        qs: list of the months to evaluate teh jumps. e.g. ['03','09']
        jump_type: 1: percentage; 2: difference
        '''
        dfmod = dfmod[dfmod.CALC_DATE.map(lambda x: str(x)[-2:] in qs)]

        dta = []
        ct = 0
        for cd in dfmod['CALC_DATE'].drop_duplicates():
            if ct > 0:

                dfnew = dfmod[(dfmod.CALC_DATE == cd) & (dfmod.CAM_YEAR == int(str(cd - 9)[:4]))][
                    ['CSF_LCID', 'CALC_DATE', 'CAM_TYPE', sco]]
                dfj = pd.merge(dfold, dfnew, on='CSF_LCID', suffixes=['_old', '_new'])
                dfj = dfj[dfj.CAM_TYPE_old != dfj.CAM_TYPE_new]

                if jump_type == 1:
                    dfj['jump'] = (dfj[sco + '_new'] - dfj[sco + '_old']) / dfj[sco + '_old']
                else:
                    dfj['jump'] = dfj[sco + '_new'] - dfj[sco + '_old']

                dfsa2ca = dfj[dfj.CAM_TYPE_old == 'SA']
                dfca2sa = dfj[dfj.CAM_TYPE_old == 'CA']

                # metrics
                saca_ct = len(dfsa2ca)
                saca_ct_pos = len(dfsa2ca[dfsa2ca['jump'] > 0])
                saca_ct_neg = len(dfsa2ca[dfsa2ca['jump'] < 0])
                saca_avg = np.around(dfsa2ca['jump'].mean() * 100, 4)
                saca_avg_pos = np.around(dfsa2ca[dfsa2ca['jump'] > 0]['jump'].mean() * 100, 4)
                saca_avg_neg = np.around(dfsa2ca[dfsa2ca['jump'] < 0]['jump'].mean() * 100, 4)

                casa_ct = len(dfca2sa)
                casa_ct_pos = len(dfca2sa[dfca2sa['jump'] > 0])
                casa_ct_neg = len(dfca2sa[dfca2sa['jump'] < 0])
                casa_avg = np.around(dfca2sa['jump'].mean() * 100, 4)
                casa_avg_pos = np.around(dfca2sa[dfca2sa['jump'] > 0]['jump'].mean() * 100, 4)
                casa_avg_neg = np.around(dfca2sa[dfca2sa['jump'] < 0]['jump'].mean() * 100, 4)

                dta.append([cd, saca_ct, saca_ct_pos, saca_ct_neg, saca_avg, saca_avg_pos, saca_avg_neg,
                            casa_ct, casa_ct_pos, casa_ct_neg, casa_avg, casa_avg_pos, casa_avg_neg])

                if plot_cigar:
                    if len(dfsa2ca) > 0:
                        self.plot_cigar_2cols(dfsa2ca, sco + '_old', sco + '_new',
                                              title='SA2CA_' + str(cdo) + '-' + str(cd))
                    if len(dfca2sa) > 0:
                        self.plot_cigar_2cols(dfca2sa, sco + '_old', sco + '_new',
                                              title='CA2SA_' + str(cdo) + '-' + str(cd))

            dfold = dfmod[(dfmod.CALC_DATE == cd) & (dfmod.CAM_YEAR == int(str(cd - 9)[:4]))][
                ['CSF_LCID', 'CALC_DATE', 'CAM_TYPE', sco]]

            cdo = cd
            ct += 1

        jumps = pd.DataFrame(dta)
        jumps.columns = ['date', 'saca_ct', 'saca_ct_pos', 'saca_ct_neg', 'saca_avg', 'saca_avg_pos', 'saca_avg_neg',
                         'casa_ct', 'casa_ct_pos', 'casa_ct_neg', 'casa_avg', 'casa_avg_pos', 'casa_avg_neg']

        return jumps
    
    
#from sqlalchemy import create_engine
    
#    class Store:
#
#    def __init__(self):
#        pass
#
#    def insert_and_get_runid(self, runname, freq, aspect, scoretypes, num_records, status='', comments=''):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        userid = os.getlogin()
#        ts = dt.datetime.now()
#        scoretype = list(scoretypes.keys())
#        scotyp = str(scoretype).replace('[', '').replace(']', '').replace(' ','')
#
#        uprow = (scotyp, runname, freq, aspect, ts, userid, num_records, status, comments)
#        sql = '''insert into RUN_LOG (SCORETYPES, NAME, FREQ, ASPECT, RUNTIME, USERID, NUM_RECORDS, STATUS, COMMENTS)
#                values (:1, :2, :3, :4, :5, :6, :7, :8, :9)
#                '''
#        cur.execute(sql, uprow)
#        con.commit()
#
#        cur.execute('SELECT ID FROM RUN_LOG ORDER BY RUNTIME DESC')
#        runid = cur.fetchall()[0][0]
#
#        return runid
#
#
#    def get_run_log(self):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#        cur.execute('SELECT * FROM RUN_LOG ORDER BY RUNTIME DESC')
#
#        return pd.DataFrame(cur.fetchall(), columns=[des[0] for des in cur.description])
#
#
#    def store_results(self, df, scoretypes, runname='', freq = 'Q', aspect='TOTAL', runid='', status='', comments=''):
#
#        '''
#        input:
#        df - campaign year
#        scoretypes - dictionary of structure:{'name_of_value_to_store': 'column_name_in_df', ...}
#        runname - batch name
#        freq = 'Q': quarterly; 'M': monthly
#        aspect  - 'TOTAL': top level scores; 'ALL': all questions, themes, dimensions and total scores
#        comments - optional argument
#        '''
#
#        temp_tbl = 'SCORES_TEMP'
#
#        if aspect == 'TOTAL':
#            df = df[df.ASP_LCID==4307611].copy().reset_index(drop=True)
#
#        num_records = len(df)
#        if runid == '':
#            runid = self.insert_and_get_runid(runname, freq, aspect, scoretypes, num_records, status, comments)
#
#        result_tbl = 'RESULTS_' + freq + '_' + aspect
#
#        t1x = time.time()
#
#        ###create table
#        sco_cols = list(scoretypes.values())
#        df1 = df[['CALC_DATE', 'CAM_YEAR', 'CSF_LCID', 'ASP_LCID'] + sco_cols].copy().reset_index(drop=True)
#        df1 = df1.where(pd.notnull(df1), None)
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        sql = '''CREATE TABLE ''' + temp_tbl + '''
#                (CALC_DATE number(10) NOT NULL,
#                  CAM_YEAR number(10) NOT NULL,
#                  CSF_LCID number(30) NOT NULL,
#                  ASP_LCID number(30),
#                  S''' + str(runid) + ''' number(30)
#                )
#                '''
#        cur.execute(sql)
#
#        sql = '''ALTER TABLE ''' + temp_tbl + '''
#                ADD (CONSTRAINT ''' + temp_tbl + '''_pk PRIMARY KEY (CALC_DATE, CAM_YEAR, CSF_LCID, ASP_LCID))
#                '''
#        cur.execute(sql)
#
#        argct = 5
#        kwargs = ':1, :2, :3, :4, :5,'
#        cols = 'S' + str(runid)
#        colsup = 'A.S' + str(runid) + '=B.S' + str(runid)
#        colsins = 'B.S' + str(runid)
#
#        #add marker column to identify records where the score applies
#        cur.execute('ALTER TABLE ' + result_tbl + ' ADD ' + cols + ' BINARY_FLOAT')
#
#        for scotyp in scoretypes.keys():
#            argct += 1
#            kwargs = kwargs + ' :' + str(argct) + ','
#            col = 'S' + str(runid) + '_' + scotyp
#            colup = 'A.' + col + '=B.' + col
#            colins = 'B.' + col
#            cols = cols + ', ' + col
#            colsup = colsup + ', ' + colup
#            colsins = colsins + ', ' + colins
#            df1.rename(columns={scoretypes[scotyp]: col}, inplace=True)
#            cur.execute('ALTER TABLE ' + temp_tbl + ' ADD S' + str(runid) + '_' + scotyp + ' BINARY_FLOAT')
#
#            try:
#                cur.execute('ALTER TABLE ' + result_tbl + ' ADD ' + col + ' BINARY_FLOAT')
#            except BaseException as e:
#                if str(e)[:9] == 'ORA-01430':
#                    print('score', col, 'already existed and will be updated')
#
#        kwargs = kwargs[:-1]
#
#        ###insert data into table
#        maxiter = 0
#        uprows = []
#
#        print(len(df1), 'records to be stored')
#
#        for index, row in df1.iterrows():
#            uprow = [int(row.CALC_DATE), int(row.CAM_YEAR), row.CSF_LCID, row.ASP_LCID, 1]
#            for scotyp in scoretypes.keys():
#                col = 'S' + str(runid) + '_' + scotyp
#                uprow.append(row[col])
#
#            uprows.append(uprow)
#            if not (index + 1) % 300000 or index == len(df1) - 1:
#
#                sql_ = '''insert into ''' + temp_tbl + ''' (CALC_DATE, CAM_YEAR, CSF_LCID, ASP_LCID,  ''' + cols + ''')
#                            values (''' + kwargs + ''')'''
#
#                while maxiter < 30:
#                    try:
#                        cur.executemany(sql_, uprows)
#                        print(index + 1, 'records stored')
#                        maxiter = 0
#                        break
#                    except:
#                        con = cx_Oracle.connect(c.constr)
#                        cur = con.cursor()
#                        maxiter += 1
#                        print('error', maxiter, 'iterations')
#                else:
#                    print('max iteration limit reached, you cannot connect to the database')
#                    cur.execute('DROP TABLE ' + temp_tbl)
#                    self.delete_runid(runid)
#                    exit()
#
#                con.commit()
#                uprows = []
#
#        sql = '''MERGE INTO ''' + result_tbl + ''' A USING ''' + temp_tbl + ''' B
#                ON (A.CALC_DATE=b.CALC_DATE and A.CSF_LCID=B.CSF_LCID and A.ASP_LCID=B.ASP_LCID)
#                WHEN MATCHED THEN UPDATE SET ''' + colsup + '''
#                WHEN NOT MATCHED THEN INSERT (CALC_DATE, CAM_YEAR, CSF_LCID, ASP_LCID, ''' + cols + ''') VALUES
#                (B.CALC_DATE, B.CAM_YEAR, B.CSF_LCID, B.ASP_LCID, ''' + colsins + ''')'''
#
#        cur.execute(sql)
#        con.commit()
#
#        cur.execute('DROP TABLE ' + temp_tbl)
#
#        con.close()
#        hlp.time_it(t1x)
#        print('results stored under', runid)
#
#
#    def delete_results(self, freq, aspect, colnames):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        for colname in colnames:
#            try:
#                sql = 'ALTER TABLE RESULTS_' + freq + '_' + aspect + ' DROP COLUMN ' + colname
#                cur.execute(sql)
#            except:
#                print('no column', colname, ' in table RESULTS_' + freq + '_' + aspect)
#                pass
#
#
#    def delete_runid(self, runid):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        sql = 'SELECT SCORETYPES, FREQ, ASPECT FROM RUN_LOG WHERE ID =:1'
#        cur.execute(sql, [runid])
#        run_log = cur.fetchall()
#
#        if len(run_log) == 0:
#            print('no score run with ID', runid)
#        else:
#            for sco in run_log[0][0].split(','):
#                sco = sco.replace("'", '')
#
#                self.delete_results(run_log[0][1], run_log[0][2], ['S' + str(runid) + '_' + sco])
#            self.delete_results(run_log[0][1], run_log[0][2], ['S' + str(runid)])
#
#        cur.execute('DELETE FROM RUN_LOG WHERE ID=' + str(runid))
#        con.commit()



class storeOther:

    def create_config_dates_2(self, create=False, update_date_scores=False, sync_acc=True):

        from dateutil.rrule import rrule, MONTHLY
        from dateutil.relativedelta import relativedelta
        from datetime import date

        con = cx_Oracle.connect(c.constr)
        cur = con.cursor()

        if create == True:

            y = 2001;
            m = 3
            strt_dt = dt.date(y, m, 1)

            cur.execute('DROP TABLE CONFIG_DATES_2')

            sql = '''CREATE TABLE CONFIG_DATES_2
                        (CALC_DATE NUMBER NOT NULL,
                         CAM_YEAR NUMBER NOT NULL,
                         BATCH VARCHAR2(4),
                         DATE_SCORES DATE,
                         DATE_ANCHOR DATE,
                         DATE_MCAP DATE,
                         DATE_RETURN DATE)
                        '''
            cur.execute(sql)

            sql = '''ALTER TABLE CONFIG_DATES_2
                        ADD (CONSTRAINT pk2 PRIMARY KEY (CALC_DATE))
                        '''
            cur.execute(sql)

        else:

            dfcd = q.get_table('CONFIG_DATES_2')
            dfcd = dfcd.sort_values('CALC_DATE')
            calc_date = dfcd.tail(1).iloc[0, 0]
            y = int(str(calc_date)[:4])
            m = int(str(calc_date)[4:6])

            strt_dt = dt.date(y, m, 1) + relativedelta(months=1)

        end_dt = dt.date.today()

        for dat in rrule(MONTHLY, dtstart=strt_dt, until=end_dt):

            calc_date = str(dat)[:4] + str(dat)[5:7]
            cam_year = (dat.year - 1) if int(str(dat)[5:7]) < 9 else dat.year
            batch = '2a' if dat.month == 3 else ('2b' if dat.month == 6 else ('1a' if dat.month == 9 else ('1b' if dat.month == 12 else '')))
            #date_scores = str(dat + relativedelta(months=1) - dt.timedelta(days=1))[:10]
            date_scores = str(dt.date(dat.year, dat.month, 15))[:10]
            date_anchor = str(dat - dt.timedelta(days=1))[:10]
            date_mcap = str(dt.date(cam_year - 1, 12, 31))[:10]
            date_return = ''

            vals = (calc_date, cam_year, batch, date_scores, date_anchor, date_mcap, date_return)
            print('values inserted', vals)

            sql = '''INSERT INTO CONFIG_DATES_2 (CALC_DATE, CAM_YEAR, BATCH, DATE_SCORES, DATE_ANCHOR, DATE_MCAP, DATE_RETURN) 
                        VALUES (:1, :2, :3, to_date(:4, 'RRRR-MM-DD'), to_date(:5, 'RRRR-MM-DD'), 
                        to_date(:6, 'RRRR-MM-DD'), to_date(:7, 'RRRR-MM-DD'))
                    '''
            print(sql)
            cur.execute(sql, vals)
            con.commit()

        if update_date_scores == True:

            sql = '''MERGE INTO CONFIG_DATES_2 A USING CONFIG_DATES B ON (A.CALC_DATE=B.CALC_DATE)
                        WHEN MATCHED THEN UPDATE SET A.DATE_SCORES=B.DATE_SCORES
                    '''
            cur.execute(sql)
            con.commit()

        if sync_acc:
            conac = create_engine(c.constralcha)
            df = q.get_table('CONFIG_DATES_2')
            df.to_sql('CONFIG_DATES_2', conac, if_exists='replace')
            self.sinc_acc_score_tables(calc_date)

#
#    def insert_returns(self, dfret, tablename='RETURNS_ANNUAL'):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        sql1 = '''DROP TABLE ''' + tablename
#
#        sql2 = '''CREATE TABLE ''' + tablename + '''
#                (CAM_YEAR number(10) NOT NULL,
#                  CSF_LCID number(30),
#                  BBGID varchar2(50),
#                  CSF_BB_COMPANY varchar2(50),
#                  TRA BINARY_DOUBLE,
#                  EFFECTIVE_DATE DATE
#                )
#                '''
#        try:
#            cur.execute(sql1)
#        except:
#            pass
#
#        cur.execute(sql2)
#
#        dfret = dfret.reset_index(drop=True)
#
#        uprows = []
#        for index, row in dfret.iterrows():
#            uprow = (int(row.CAM_YEAR), int(row.CSF_LCID), str(row.BBGID), str(row.CSF_BB_COMPANY), row.TRA, row.EFFECTIVE_DATE)
#            uprows.append(uprow)
#
#        sql_ = '''insert into ''' + tablename + ''' (CAM_YEAR, CSF_LCID, BBGID, CSF_BB_COMPANY, TRA, EFFECTIVE_DATE)
#        values (:1, :2, :3, :4, :5, to_date(:6, 'RRRR-MM-DD'))'''
#
#        cur.executemany(sql_, uprows)
#
#        con.commit()
#
#
#    def insert_mcaps(self, dfret, tablename='MCAPS_ANNUAL'):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        sql1 = '''DROP TABLE ''' + tablename
#
#        sql2 = '''CREATE TABLE ''' + tablename + '''
#                (CAM_YEAR number(10) NOT NULL,
#                  CSF_LCID number(30),
#                  BBGID varchar2(50),
#                  CSF_BB_COMPANY varchar2(50),
#                  FULL_MCAP_USD BINARY_DOUBLE,
#                  MCAP_DATE DATE
#                )
#                '''
#        try:
#            cur.execute(sql1)
#        except:
#            pass
#
#        cur.execute(sql2)
#
#        dfret = dfret.reset_index(drop=True)
#
#        uprows = []
#        for index, row in dfret.iterrows():
#            uprow = (int(row.CAM_YEAR), int(row.CSF_LCID), str(row.BBGID), str(row.CSF_BB_COMPANY), row.FULL_MCAP_USD, row.MCAP_DATE)
#            uprows.append(uprow)
#
#        sql_ = '''insert into ''' + tablename + ''' (CAM_YEAR, CSF_LCID, BBGID, CSF_BB_COMPANY, FULL_MCAP_USD, MCAP_DATE)
#        values (:1, :2, :3, :4, :5, to_date(:6, 'RRRR-MM-DD'))'''
#
#        cur.executemany(sql_, uprows)
#
#        con.commit()
#
#        #alter table mcaps_annual rename column cam_year to year;
#        #ALTER TABLE mcaps_annual ADD cam_year NUMBER(10);
#        #update mcaps_annual set cam_year = year + 1;
#
#
#    def update_missing_mcaps(self):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        lc = 'S:/products and engineering - 0048187/Quantitative Analysis/Research/program_files/ret/'
#        dfmis = pd.read_excel(lc + 'missing_mcaps_upload.xlsx')
#
#        tablename = 'MCAPS_ANNUAL_COPY' #change to do on the original table
#
#        uprows = []
#        for index, row in dfmis.iterrows():
#            uprow = (int(row.YEAR), int(row.CSF_LCID), row.MCAP_UNION, str(int(row.YEAR)) + '-12-31', int(row.CAM_YEAR))
#            uprows.append(uprow)
#
#        sql_ = '''insert into ''' + tablename + ''' (YEAR, CSF_LCID, FULL_MCAP_USD, MCAP_DATE, CAM_YEAR)
#                values (:1, :2, :3, to_date(:4, 'RRRR-MM-DD'), :5)'''
#
#        cur.executemany(sql_, uprows)
#        con.commit()
#        print('done')
#
#
#    def store_anchor_universe(self, yr):
#
#        con = create_engine(c.constralch)
#
#        col_list = ['INDEX NAME', 'INDEX KEY', 'COMPANY', 'RIC', 'BLOOMBERG TICKER', 'ISIN', 'CSF ID', 'SEDOL',
#                    'GICS CODE', 'MIC', 'COUNTRY OF DOMICILE', 'COUNTRY OF LISTING', 'CURRENCY CODE']
#
#        df = pd.DataFrame()
#
#        for file in os.listdir(aloc + str(yr)):
#            df_ = pd.read_excel(aloc + str(yr) + '/' + file)
#            df_ = df_[-df_['CSF ID'].isnull()][col_list]
#            df = pd.concat([df, df_], axis=0).reset_index(drop=True)
#
#        df['SEDOL'] = df['SEDOL'].map(lambda x: str(x))
#        df.to_sql('anchor_universe_' + str(yr), con, if_exists='replace')
#
#
#    def create_results_table(self, freq='M', score_level='TOTAL', drop_old=True):
#
#        con = cx_Oracle.connect(c.constr)
#        cur = con.cursor()
#
#        results_tb_name = 'results_' + freq + '_' + score_level
#
#        dfdts = q.get_table('CONFIG_DATES_2')
#        if freq == 'M':
#            calc_dates = list(dfdts.CALC_DATE)
#        elif freq == 'Q':
#            calc_dates = list(dfdts.CALC_DATE[dfdts.CALC_DATE.map(lambda x: str(x)[-2:] in ['03', '06', '09', '12'])])
#
#        if score_level == 'TOTAL':
#            str_total = 'and dimension is null'
#        elif score_level == 'ALL':
#            str_total = ''
#
#        if drop_old == True:
#            sql1 = '''DROP TABLE ''' + results_tb_name
#            cur.execute(sql1)
#
#        for i in range(len(calc_dates)):
#            if i == 0:
#                sql_create = ('create table ' + results_tb_name +
#                              ' as (select calc_date, cam_year, csf_lcid, asp_lcid ' +
#                              'from SCORE_DATA_' + str(calc_dates[i]) +
#                              ' where asp_lcid is not null ' + str_total + ')')
#                cur.execute(sql_create)
#                con.commit()
#            else:
#                pass
#                sql_insert = ('INSERT INTO ' + results_tb_name + ' (calc_date, cam_year, csf_lcid, asp_lcid) ' +
#                              'SELECT calc_date, cam_year, csf_lcid, asp_lcid FROM score_data_' + str(calc_dates[i]) +
#                              ' where asp_lcid is not null ' + str_total + '')
#
#                cur.execute(sql_insert)
#                con.commit()
#
#        sql_cons = ('alter table  ' + results_tb_name + ' add (constraint res_' + freq + '_' + score_level +
#                    ' primary key (calc_date, csf_lcid, asp_lcid))')
#        cur.execute(sql_cons)
#        con.commit()
#
#        print('table', results_tb_name, 'created')
#
