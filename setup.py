import billboard
import json
import pandas as pd

def import_chart(name='hot-100', date=None, fetch=None, timeout=None):
    '''
    Chart importing using the billboard.py HTML parser

    '''

    chart = billboard.ChartData(name=name, date=date, fetch=fetch, timeout=timeout)

    return chart
    # chart = billboard.ChartData(name=name, date=date)
    #
    # chart_json = json.loads(chart.json())
    #
    # chart_df = pd.DataFrame(chart_json)
    #
    # chart_df = pd.DataFrame(chart_df['entries'].array)
    #
    # return chart_df

def chart_parser(chart):
    '''
    Chart Parser. Returns Pandas DataFrame object

    Returns:
        chart_df - Pandas DataFrame
            .date - YYYY-MM-DD
            .artist - String
            .title - String
            .lastPos - Int64
            .rank - Int64
            .weeks - Int64
    '''

    chart_entries = json.loads(today.json()) # Load from JSON object
    chart_entries = pd.DataFrame.from_dict(chart_entires['entries'])

    # Clean Entries
    chart_entries = chart_entries[['artist', 'title', 'lastPos', 'rank', 'weeks']]
    chart_entries['date'] = chart.date # Add Date to all rows

    return chart_entries

    
