'''
    Billboard .csv Set-Up

    This script will pull Billboard Data using the billboard.py webscraper.
    GOAL: Create a 1 year, 3 year, and 5 year log of the billboard 'top-100' charts.

    @ TODO Complete date sectioning
    @ TODO Pull Data for 1 year charts. Save to csv
    @ TODO Pull Data for 3 year charts. Save to csv
    @ TODO Pull Data for 5 year charts. Save to csv
'''

# Imports

import setup
import pandas as pd
import billboard
import datetime

## Code ##

# Date Sectioning

'''
The Dates on Billboard 'Hot-100' are recorded in a YYYY-MM-DD format
The chart is updated weekly, from Sunday - Saturday.
    EX: Today is 2019 October 4th, in the week of 2019 September 29 - 2019 October 5

Per Year, there are 52 Charts
'''

# All charts start at the end of August 2019, or 2019 August 25 or 2019-08-25
# To get date out of timedate object, use date.isoformat()

dates_1 = [] # 1 Year Dates
dates_3 = [] # 3 Year Dates
dates_5 = [] # 5 Year Dates

initial_date = datetime.date(2019,8,25)
one_week = datetime.timedelta(days=7)

curr_date = initial_date

for i in range(52*5):
    curr_date_fmt = curr_date.isoformat()
    if (i < 52):
        dates_1.append(curr_date_fmt)
        dates_3.append(curr_date_fmt)
        dates_5.append(curr_date_fmt)
    elif (i >= 52 and i < 52*3):
        dates_3.append(curr_date_fmt)
        dates_5.append(curr_date_fmt)
    elif (i >= 52*3):
        dates_5.append(curr_date_fmt)

    curr_date = curr_date - one_week
