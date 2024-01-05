from flask import Flask,  render_template, request
from polygon import RESTClient
import pandas as pd
import numpy as np
import requests
import math

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
    timespan = request.form['timespan']
    multiplier = request.form['multiplier']
    period = int(request.form['period'])

    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=f"{start_date}", to=f"{end_date}", limit=50000):
        aggs.append(a)

    data = pd.DataFrame(aggs)

    data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

    
    # Function to calculate logarithmic return
    def calculate_log_return(price, initial_price):
        return math.log(price / initial_price)

    # Function to calculate simple return
    def calculate_simple_return(price, initial_price):
        return ((price - initial_price) / initial_price) * 100

    data['return'] = ((data['close']-data['close'].shift(1))/ (data['close'].shift(1))) * 100

    data['average'] = data['close'].rolling(window=period).mean()

    # MAD, like range, variation and standard deviations, is another measure of how spread out the data is.
    # Comparing the MAD value at a specific date with historical MAD values can provide context. If the current MAD is significantly higher than historical values, it may suggest increased volatility.

    data['deviation'] = data['close'] - data[ 'average']

    data['MAD'] = data['deviation'].rolling(window=period).mean()
    # Calculate the volatility
    data['RollingVolatility'] = data['return'].rolling(window=period).std()
    # Calculate 1-month rolling window of daily volatility
    data['vol'] = data['RollingVolatility'].rolling(window=period).sum()

    data['SumMAD'] = data['MAD'].rolling(window=period).sum()

    # Calculate 1-month rolling window of daily volatility
    data['vol'] = data['RollingVolatility'].rolling(window=period).sum()

    # Calculate the "Price"
    data['Price'] = data['SumMAD'] / data['vol']

    data['Price'] = data['Price'].round(2)
    data['MAD'] = data['MAD'].round(2)

    squeeze = data[['date','close','MAD','Price']]

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
    filtered_low_high = pd.merge(filtered_low_high, data[['date', 'average', 'MAD']], on='date', how='left')

    # Create a new DataFrame with selected columns
    selected_columns = ['date', 'average', 'HIP', 'LOP', 'MAD']

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

    # Concatenate the DataFrames along the rows
    merged_df = pd.concat([Low2s, High2s])

    # Sort the merged DataFrame based on the 'date' column
    merged_df.sort_values(by='date', inplace=True)

    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)

    df_final = merged_df[['date', 'High2', 'High3', 'Low2', 'Low3']]

    # Merge the DataFrames based on the 'date' column using an outer join
    merged_df2 = pd.merge(df_final, filtered_low_high_selected, on='date', how='outer')

    # Sort the merged DataFrame based on the 'date' column
    merged_df2.sort_values(by='date', inplace=True)

    # Reset the index of the merged DataFrame
    merged_df2.reset_index(drop=True, inplace=True)

    df = merged_df2[['date', 'average', 'HIP', 'LOP', 'High2', 'Low2', 'High3', 'Low3', 'MAD']]
    df[['average', 'HIP', 'LOP', 'High2', 'Low2', 'High3', 'Low3', 'MAD']] = df[['average', 'HIP', 'LOP', 'High2', 'Low2', 'High3', 'Low3', 'MAD']].round(2)

    # Merge columns and highlight based on common or not
    def merge_and_highlight(row, col1, col2):
        if pd.notna(row[col1]) and pd.notna(row[col2]):
            if row[col1] == row[col2]:
                return f'<span class="common">{row[col1]}</span>'
            else:
                return f'<span class="different">{row[col1]} ({row[col2]})</span>'
        elif pd.notna(row[col1]):
            return row[col1]
        elif pd.notna(row[col2]):
            return row[col2]
        else:
            return ''
            
    high2 = 'High2'
    high3 = 'High3'
    low2 = 'Low2'
    low3 = 'Low3'

    df['Highs'] = df.apply(merge_and_highlight, args = (high2, high3), axis=1)
    df['Lows'] = df.apply(merge_and_highlight, args = (low2, low3), axis=1)

    df['Highs'] = pd.to_numeric(df['Highs'], errors = 'coerce')
    df['Lows'] = pd.to_numeric(df['Lows'], errors = 'coerce')

    Low_High = df[df['Highs'].notna()]
    High_Low = df[df['Lows'].notna()]

    # Low_High['Highs'] = pd.to_numeric(Low_High['Highs'], errors = 'coerce')
    # High_Low['Lows'] = pd.to_numeric(High_Low['Lows'], errors = 'coerce')

    high_low_condition = (
        (High_Low['Lows'] > High_Low['Lows'].shift(1)) & (High_Low['Lows'] > High_Low['Lows'].shift(-1))
    )
    low_high_condition = (
            (Low_High['Highs'] < Low_High['Highs'].shift(1)) & (Low_High['Highs'] < Low_High['Highs'].shift(-1))
        )

    High_Low['High_Low'] = np.where(high_low_condition, High_Low['Lows'], np.nan)


    Low_High['Low_High'] = np.where(low_high_condition, Low_High['Highs'], np.nan)

    df['Highs'] = df['Highs'].fillna('')
    df['Lows'] = df['Lows'].fillna('')

    df.drop(['High2', 'High3', 'Low2', 'Low3'], axis=1, inplace=True)
    df['Highs'] = df['Highs'].apply(lambda x: f'<span class="common">{x}</span>' if '<span class="common">' in str(x) else f'<span class="different">{x}</span>')
    df['Lows'] = df['Lows'].apply(lambda x: f'<span class="common">{x}</span>' if '<span class="common">' in str(x) else f'<span class="different">{x}</span>')

    df2 = pd.concat([High_Low,Low_High])
    df2.sort_values(by='date', inplace=True)
    df2.reset_index(drop=True, inplace=True)

    df2['Low_High'] = df2['Low_High'].round(2)
    df2['High_Low'] = df2['High_Low'].round(2)

    df2 = df2[['date', 'Low_High','High_Low']]

    df = df.merge(df2, how='left', on = 'date')

    df = df.drop_duplicates()
    df.fillna('', inplace=True)
    df.replace(np.nan, '', inplace=True)

    squeeze.fillna('', inplace=True)

    df_html = df.to_html(escape=False, classes='styled-table', index=False)

# Set up the API endpoint and parameters
    symbol = request.form['ticker']
    api_key = "WVUSvKCnEd6LhmVcunpZUXfeTq39md5p"

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={api_key}"

    # Make the API request
    response = requests.get(url)
    prev = response.json()
    prev = prev['results']

    prev_data = prev[0]  # Accessing the first dictionary in the list

    open_price = prev_data['o']
    close_price = prev_data['c']
    low_price = prev_data['l']
    high_price = prev_data['h']

    # Create a DataFrame
    data_dict = {
        "Previous Open": [open_price],
        "Previous High": [high_price],
        "Previous Low": [low_price],
        "Previous Close": [close_price]
    }
    prevs = pd.DataFrame(data_dict)

    # Convert the DataFrames to HTML tables
    # Replace with your function to get the existing table HTML
    prev_html = prevs.to_html(index=False, escape=False, classes=['styled-table2','text-with-colour'])

    squeeze_html = squeeze.to_html(index=False, classes='styled-table', escape=False)
    # Extract the values
    return render_template("output.html", previous=prev_html, squeeze = squeeze_html, ticker=ticker, dataframe= df_html)

if __name__ == '__main__':
    # Run the Flask app
    website.run()




