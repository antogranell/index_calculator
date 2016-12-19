#run index selection
################## this code below should populate throught the interface
import sys
sys.path.append('D:/Python/AntoTradingSystem/indexifyLib/')
from idxfy import yafin, idxfy as ix

dfmaster = yafin.getmaster('Stock') #this can be added to the universe class
uni1 = ix.universe(dfmaster, 'equity', country='usa', industry='industrial')
cal = ix.calendar([3,9], startdate='2016/06/01')

for cd in cal.reviewCutDates:
    p1 = ix.portfolio(uni1, cd)
################## this code below that the user writes
    p1.getMcap() #get mcap
    p1.getAdtv(1) #get 1 month adtv
    p1.getItem('yield') #get dividend yield
    p1.rank(by=['adtv1m','ffmcap'], ascending=[False, False], prc=True) #rank by adtv1m and by mcap as second criterion
    p1.table = p1.table[p1.table.rk>0.5] #screen 50% best
    p1.selectTop('Dividend Yield', 20) #from those, select the top 20 by dividend yield 
    
print(p1.selection)