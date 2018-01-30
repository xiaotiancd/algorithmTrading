import feather
import pandas as pd

# read stock data
path = './SH600000.feather'
df = feather.read_dataframe(path)
# feather.write_dataframe(df, output_path)

# extract data from "start_date" to "end_date"
start_date = "2016-09-30 14:30"
end_date = "2016-09-30 15:00"
data = df
valid_data = data.loc[(data['date']>start_date)
         & (data['date']<=end_date)]
print valid_data.shape
# filtering those data with nan or null
valid_data_filter = valid_data.dropna(how='any')

# calculate Smart metric "S"
# R_t = abs(close-preClose)/preClose*100%
# V_t = amount
import numpy as np
R_elem1 = 'close'
R_elem2 = 'preClose'
V_elem = 'amount'
R = abs(valid_data_filter[R_elem1]-valid_data_filter[R_elem2])/(valid_data_filter[R_elem2]+np.finfo(float).eps)*100
S = R/(np.sqrt(valid_data_filter[V_elem])+np.finfo(float).eps)
print S.shape

# sort data by "S" -- descending
S_sort_ids = np.argsort(S.values)
S_arr = S.values[S_sort_ids[::-1]]
sort_valid_data = valid_data_filter.iloc[S_sort_ids[::-1]]

#成交量的累积占比
trade_cum_ratio = 0.2
amount_cumsum = sort_valid_data['amount'].cumsum()
all_amount_cumsum = amount_cumsum.values[-1]
smart_ids = np.where((amount_cumsum.values/all_amount_cumsum) <= trade_cum_ratio)[0]

# 聪明钱的情绪因子Q
# Q = VWAP_smart/VWAP_all
trade_mon = sort_valid_data['amount'].values * sort_valid_data['vwap'].values
VWAP_smart = np.sum(trade_mon[smart_ids])/np.sum(sort_valid_data['amount'].values[smart_ids])
VWAP_all = np.sum(trade_mon)/np.sum(sort_valid_data['amount'].values)
Q = VWAP_smart/VWAP_all
print "VWAP_smart:", VWAP_smart, ", VWAP_all:", VWAP_all
print "Q:", Q
