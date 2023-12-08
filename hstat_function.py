# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:18:55 2023

@author: townsend.lansburg
"""


import pandas as pd
import numpy as np
import boto3
from arcticdb import Arctic, QueryBuilder
from datetime import date, timedelta


ac = Arctic(f's3://s3.us-east-2.amazonaws.com:lansburg-aws?region=us-east-2&access=AKIA3ISFLTSLLKWRNMKN&secret=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')
available_libraries = ac.list_libraries()

#create a library
# ac.create_library('hstat_volume')



commodity_lists = [
    'corn',
    'wheat',
    'kansas_wheat',
    'minnie_wheat',
    'soy',
    'soy_meal',
    'soy_oil',
    'sugar',
    'white_sugar',
    'cotton',
    'coffee',
    'ethanol',
    'crude',
    'brent',
    'heating_oil',
    'gasoline',
    'cocoa',
    'feeder_cattle',
    'lean_cattle',
    'lean_hogs',
    'gasoil',
    'canola',
    'rapeseed',
    'palm_oil',
    'china_cotton',
    'china_sugar',
    'china_palm',
    'china_rapeseed_oil',
    'china_rapeseed_meal', 
    'china_corn',
    'china_soy_meal' ,
    'china_bean_oil',

    ]



# library = ac['hstat_open_interest']
# library.read('ethanol').data





def calculate_commodity_stats(commodity, start_date, end_date):
    # Fetch the price data
    library = ac['commodity_prices']
    px_data = library.read(commodity).data
    px_data = px_data[px_data.index >= start_date ]
    
    library = ac['commodity_volume']
    volume_data = library.read(commodity).data
    volume_data = volume_data[volume_data.index >= start_date ]

    library = ac['commodity_open_interest']
    oi_data = library.read(commodity).data
    oi_data = oi_data[oi_data.index >= start_date ]
    
    # Combine the data into a single DataFrame
    max_values_per_date = oi_data.groupby('date')['value'].transform('max')
    max_oi = oi_data[oi_data['value'] == max_values_per_date]
    
    max_values_per_date = volume_data.groupby('date')['value'].transform('max')
    max_volume = volume_data[volume_data['value'] == max_values_per_date]
    
    
    
    px_data_sorted = px_data.sort_index().sort_values(by=['contract', 'date'])
    # Then apply the diff function within each group of 'contract'
    px_data_sorted['value_diff'] = px_data_sorted.groupby('contract')['value'].diff()
    
    
    px_data_sorted = px_data_sorted.reset_index() if 'date' not in px_data_sorted.columns else px_data_sorted
    max_oi = max_oi.reset_index() if 'date' not in max_oi.columns else max_oi
    max_volume = max_volume.reset_index() if 'date' not in max_volume.columns else max_volume
    # Merge the max_oi DataFrame with the px_data DataFrame to get 'value' and 'value_diff'
    # We perform a left join to keep all rows from max_oi and match with px_data
    
    price_oi = pd.merge(max_oi, px_data_sorted, on=['date', 'contract'], how='left')
    price_volume = pd.merge(max_volume, px_data_sorted, on=['date', 'contract'], how='left')
    
    price_oi= price_oi.set_index('date')
    price_volume= price_volume.set_index('date')
    
    daily_price_change_volume = price_volume['value_diff'].fillna(0)
    daily_price_change_oi =  price_oi['value_diff'].fillna(0)
    
    price_volume_cumulative = price_volume['value_diff'].fillna(0).cumsum() + price_volume['value_y'][0]
    price_oi_cumulative = price_oi['value_diff'].fillna(0).cumsum() + price_oi['value_y'][0]
    
    # Calculate Sharpe Ratio, Variability and Longer Signal for price volume
    price_volume_stats = pd.DataFrame(price_volume_cumulative).rename(columns={'value_diff': 'PX'})
    price_volume_stats['Sharpe'] = ((daily_price_change_volume.rolling(20).mean())/(daily_price_change_volume.rolling(20).std()))*20**.5
    price_volume_stats['Var'] = ((price_volume_stats['PX'].rolling(4).mean() - price_volume_stats['PX'].rolling(39).mean())/ (((price_volume_stats['PX'].rolling(4).var()*4 + price_volume_stats['PX'].rolling(39).var()*39)/43)**0.5*(1/5+1/40)**0.5))
    price_volume_stats['Longer_Signal'] = (price_volume_stats['PX'] - price_volume_stats['PX'].rolling(100).mean())/price_volume_stats['PX'].rolling(100).std()
    price_volume_stats['HSTAT'] = price_volume_stats['Sharpe'] + price_volume_stats['Var'] + price_volume_stats['Longer_Signal']
    price_volume_stats['Spot'] = price_volume['value_y']
    price_volume_stats = price_volume_stats.fillna(0)
    
    price_oi_stats = pd.DataFrame(price_oi_cumulative).rename(columns={'value_diff': 'PX'})
    price_oi_stats['Sharpe'] = ((daily_price_change_oi.rolling(20).mean())/(daily_price_change_oi.rolling(20).std()))*20**.5
    price_oi_stats['Var'] = ((price_oi_stats['PX'].rolling(4).mean() - price_oi_stats['PX'].rolling(39).mean())/ (((price_oi_stats['PX'].rolling(4).var()*4 + price_oi_stats['PX'].rolling(39).var()*39)/43)**0.5*(1/5+1/40)**0.5))
    price_oi_stats['Longer_Signal'] = (price_oi_stats['PX'] - price_oi_stats['PX'].rolling(100).mean())/price_oi_stats['PX'].rolling(100).std()
    price_oi_stats['HSTAT'] = price_oi_stats['Sharpe'] + price_oi_stats['Var'] + price_oi_stats['Longer_Signal']
    price_oi_stats['Spot'] = price_oi['value_y']
    price_oi_stats = price_oi_stats.fillna(0)
    
    library = ac['hstat_volume']
    library.write(commodity, price_volume_stats)
    library = ac['hstat_open_interest']
    library.write(commodity, price_oi_stats)
    
    # ... Repeat the same steps for open interest ...

    return price_volume_stats, price_oi_stats


def main():
# Example usage:
# Usage
    start_date = '20070910'
    end_date = '20251231'
    for commodity in commodity_lists:
        print(commodity)
    
    #connection = None  # Replace with your actual connection object
    
        price_volume_stats, price_oi_stats = calculate_commodity_stats(commodity, start_date, end_date)
        # library = ac['hstat_volume']
        # library.write(commodity, price_volume_stats)
        # library = ac['hstat_open_interest']
        # library.write(commodity, price_oi_stats)
        
    


   
if __name__ == "__main__":
    data = main() 


# library = ac['hstat_open_interest']
# library.read('cocoa').data


