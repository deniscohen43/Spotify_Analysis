'''
    Billboard .csv Set-Up

    This script will pull Billboard Data using the billboard.py webscraper.
    GOAL: Create a 1 year, 3 year, and 5 year log of the billboard 'top-100' charts.

    @ DONE Complete date sectioning
    @ DONE Pull Data for 1 year charts. Save to csv
    @ DONE Pull Data for 3 year charts. Save to csv
    @ DONE Pull Data for 5 year charts. Save to csv
'''

# Sys Path to Set Up
import sys

sys.path.append('../')

# Imports

import setup # Path is '../'
import pandas as pd
import datetime
from time import sleep

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

dates_5 = [] # 5 Year Dates

initial_date = datetime.date(2019,8,25)
one_week = datetime.timedelta(days=7)

curr_date = initial_date

for i in range(52*5):
    curr_date_fmt = curr_date.isoformat()
    dates_5.append(curr_date_fmt)

    curr_date = curr_date - one_week

# Charts

charts_1 = []
charts_3 = []
charts_5 = []

for i in range(52*5):
    date = dates_5[i]

    # Logic for all Charts
    chart = setup.import_chart(date=date)

    chart_df = setup.chart_parser(chart)

    # Add chart_df to matching years

    if (i < 52):
        # Add first year to all charts
        charts_1.append(chart_df)
        charts_3.append(chart_df)
        charts_5.append(chart_df)

    elif (i >=52 and i < 52*3):
        # Add years 2 and 3 to charts_3 and charts_5
        charts_3.append(chart_df)
        charts_5.append(chart_df)

    elif (i >= 52*3):
        # Add years 3,4,5 to charts_5
        charts_5.append(chart_df)

    # Sleep for 5 seconds
    sleep(5)

# Combine charts list to single dataframe
year1 = pd.concat(charts_1)
year3 = pd.concat(charts_3)
year5 = pd.concat(charts_5)

year1.to_csv("../Resources/billboard_1_year.csv")
year3.to_csv("../Resources/billboard_3_year.csv")
year5.to_csv("../Resources/billboard_5_year.csv")

