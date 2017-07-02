import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader

def get_timeseries_gross_return(symbols, startdate, enddate):
    """Return price return timeseries
    
    Keyword arguments:
    symbols -- Symbol or list of Symbols (string / [string])
    startdate -- timeseries start date (datetime.date(year, month, day))
    enddate -- timeseries end date (datetime.date(year, month, day))
    """
    if type(symbols) == str:
        symbols = [symbols]
    data = []
    for symbol in symbols:
        try:
            df = DataReader(symbol, 'yahoo', startdate, enddate)[['Adj Close']]
            df.columns = [symbol]
        except:
            df = pd.DataFrame(np.nan, index=pd.bdate_range(startdate, enddate), columns=[symbol])
        data.append(df)
    return pd.concat(data, axis=1, join='outer')