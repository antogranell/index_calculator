import calendar
from pandas.tseries.offsets import BDay

def get_datelist(startdate, enddate, months, implementation=(3,5)):
    """Return effective dates
    
    Keyword arguments:
    startdate -- earliest possible date (datetime.date(year, month, day))
    enddate -- last possible date (datetime.date(year, month, day))
    months -- review/rebalancing months ([int])
    implementation -- implementation days ((int,int))
        default: 3rd Friday (3,5)
    """
    datelist = []
    for year in range(startdate.year, enddate.year+1):
        for month in [x for x in range(1,13) if x in months]:
            implementationdate = calendar.Calendar(implementation[1]-1).monthdatescalendar(year, month)[implementation[0]][0]
            effectivedate = (implementationdate + BDay(1)).date()
            datelist.append(effectivedate)
    return [x for x in datelist if x >= startdate if x <= enddate]