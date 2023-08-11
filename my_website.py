from flask import Flask,  render_template, request
from polygon import RESTClient
import pandas as pd
import numpy as np

website = Flask(__name__) 

@website.route('/', methods=['GET'])
def input_form():
    # Render the input form when the user visits the website (GET request)
    return render_template("index.html")

@website.route('/trade', methods=['POST'])
def trade():
    
    api_key="WVUSvKCnEd6LhmVcunpZUXfeTq39md5p"

    client = RESTClient(api_key=api_key)
    
    ticker = request.form['ticker']
    end_date = request.form['end_date']
    start_date = request.form['start_date']
    
    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=3, timespan="hour", from_=f"{start_date}", to=f"{end_date}", limit=50000):
        aggs.append(a)

    data = pd.DataFrame(aggs)

    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

    period = 55
    data['average'] = data['close'].rolling(window=period).mean()

    # Calculate the Low1 that meet the conditions
    data['Prev_Lows'] = data['low'].shift(1)
    data['Next_Lows'] = data['low'].shift(-1)

    # Filter the Low1 that are lower than the three previous Low1 and the three following Low1
    LOP_conditions = (
        (data['low'] < data['Prev_Lows']) &
        (data['low'] < data['Prev_Lows'].shift(1)) &
        (data['low'] < data['Prev_Lows'].shift(2)) &
        (data['low'] < data['Next_Lows']) &
        (data['low'] < data['Next_Lows'].shift(-1)) &
        (data['low'] < data['Next_Lows'].shift(-2))
    )

    # Calculate the Low1 that meet the conditions
    data['Prev_High'] = data['high'].shift(1)
    data['Next_High'] = data['high'].shift(-1)

    # Filter the Low1 that are lower than the three previous Low1 and the three following Low1
    HIP_conditions = (
        (data['high'] > data['Prev_High']) &
        (data['high'] > data['Prev_High'].shift(1)) &
        (data['high'] > data['Prev_High'].shift(2)) &
        (data['high'] > data['Next_High']) &
        (data['high'] > data['Next_High'].shift(-1)) &
        (data['high'] > data['Next_High'].shift(-2))
    )
    # Create a new column 'Filtered_Lows' with filtered low values and set others to NaN
    data['LOP'] = np.where(LOP_conditions, data['low'], np.nan)
    data['HIP'] = np.where(HIP_conditions, data['high'], np.nan)

    High1 = data[data['HIP'].notna()]
    High1 = High1.drop('low', axis = 1)

    Low1 = data[data['LOP'].notna()]
    Low1 = Low1.drop('high', axis = 1)

    Low2_conditions = (
        (Low1['LOP'] < Low1['LOP'].shift(1)) & (Low1['LOP'] < Low1['LOP'].shift(-1))
    )

    High2_conditions = ( 
        (High1['HIP'] > High1['HIP'].shift(1)) & (High1['HIP'] > High1['HIP'].shift(-1))
    )

    # Create a new column 'Filtered_Lows' with filtered low values and set others to NaN
    Low1['Low2'] = np.where(Low2_conditions, Low1['low'], np.nan)
    High1['High2'] = np.where(High2_conditions, High1['high'], np.nan)

    low1_columns_to_select = ['date', 'low', 'LOP', 'Low2']  # Replace 'column1' and 'column2' with actual column names from Low1 DataFrame
    high1_columns_to_select = ['date', 'high', 'HIP', 'High2']  # Replace 'column3' and 'column4' with actual column names from High1 DataFrame

    Lows = Low1[low1_columns_to_select]
    Highs = High1[high1_columns_to_select]

    low_high = pd.merge(Lows, Highs, on = 'date', how = 'outer')
    low_high = low_high[['date', 'LOP', 'HIP']]

    filtered_low_high = low_high[low_high['HIP'].notna() | low_high['LOP'].notna()]

    # Merge the 'average' column back into the filtered dataframe
    filtered_low_high = pd.merge(filtered_low_high, data[['date', 'average']], on='date', how='left')

    # Create a new DataFrame with selected columns
    selected_columns = ['date', 'average', 'HIP', 'LOP']

    filtered_low_high_selected = filtered_low_high[selected_columns]


    Low2s = Lows[Lows['Low2'].notna()]
    High2s = Highs[Highs['High2'].notna()]

    Low3_conditions = (
        (Low2s['Low2'] < Low2s['Low2'].shift(1)) & (Low2s['Low2'] < Low2s['Low2'].shift(-1))
    )

    High3_conditions = ( 
        (High2s['High2'] > High2s['High2'].shift(1)) & (High2s['High2'] > High2s['High2'].shift(-1))
    )

    # Create a new column 'Filtered_Lows' with filtered low values and set others to NaN
    Low2s['Low3'] = np.where(Low3_conditions, Low2s['LOP'], np.nan)
    High2s['High3'] = np.where(High3_conditions, High2s['HIP'], np.nan)

    High3A_conditions = (
        (High2s['High2'].shift(1) < High2s['High2']) &
        (High2s['High2'].shift(-1) < High2s['High2']) &
        (High2s['High2'].shift(1) > High2s['High2'].shift(-1))
    )

    Low3A_conditions = (
        (Low2s['Low2'].shift(1) > Low2s['Low2']) &
        (Low2s['Low2'].shift(-1) > Low2s['Low2']) &
        (Low2s['Low2'].shift(1) < Low2s['Low2'].shift(-1))
    )
        
        # Create a new column 'Filtered_Lows' with filtered low values and set others to NaN
    Low2s['Low3A'] = np.where(Low3A_conditions, Low2s['LOP'], np.nan)
    High2s['High3A'] = np.where(High3A_conditions, High2s['HIP'], np.nan)
        
    # Concatenate the DataFrames along the rows
    merged_df = pd.concat([Low2s, High2s])

    # Sort the merged DataFrame based on the 'date' column
    merged_df.sort_values(by='date', inplace=True)

    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)

    df_final = merged_df[['date', 'High2', 'High3', 'Low2', 'Low3', 'High3A', 'Low3A']]

    # Merge the DataFrames based on the 'date' column using an outer join
    merged_df2 = pd.merge(df_final, filtered_low_high_selected, on='date', how='outer')

    # Sort the merged DataFrame based on the 'date' column
    merged_df2.sort_values(by='date', inplace=True)

    # Reset the index of the merged DataFrame
    merged_df2.reset_index(drop=True, inplace=True)

    df = merged_df2[['date', 'HIP', 'LOP', 'High2', 'Low2', 'High3', 'Low3', 'High3A','Low3A']]

    df_html = df.to_html()

    return render_template("output.html", ticker=ticker, dataframe=df_html)


if __name__ == '__main__':
    # Run the Flask app
    website.run()




