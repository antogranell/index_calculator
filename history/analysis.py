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