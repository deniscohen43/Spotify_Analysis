import billboard
import json
import pandas as pd

def import_chart(date=None):

    chart = billboard.ChartData('hot-100', date=date)

    chart_json = json.loads(chart.json())

    chart_df = pd.DataFrame(chart_json)

    chart_df = pd.DataFrame(chart_df['entries'].array)

    return chart_df