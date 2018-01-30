import feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""extract data from "start_date" to "end_date"
    -----------------------------------
    Inputs:
    @df: dataframe defined in pandas
    @start_date: "2016-09-30 14:30"
    @end_date: "2016-09-30 15:00"
    @filter: True or False
    -----------------------------------
    Outputs:
    @valid_data:
        Type: dataframe
        columns: date, open, high, low, close, preClose, preClose2, amount, vwap
        shape: (validate_sample_num, 9)
"""
def extract_valid_data_range(df, start_date, end_date, filter = True):
    data = df
    valid_data = data.loc[(data['date']>start_date)
             & (data['date']<=end_date)]
    if filter:
        # filtering those data with nan or null
        valid_data = valid_data.dropna(how='any')
    return valid_data


"""calculate Smart metric "S"
    R_t = abs(close-preClose)/preClose*100%
    V_t = amount
    S = |R_t|/sqrt(V_t)
    -----------------------------------
    Inputs:
    @valid_data_filter:
        Type: dataframe
        columns: date, open, high, low, close, preClose, preClose2, amount, vwap
        shape: (sample_num, 9)
    -----------------------------------
    Outputs:
    @S:
        Type: series
        shape: (sample_num, 1)
"""
def calculate_S(valid_data_filter):
    R_elem1 = 'close'
    R_elem2 = 'preClose'
    V_elem = 'amount'
    R = abs(valid_data_filter[R_elem1]-valid_data_filter[R_elem2])/(valid_data_filter[R_elem2]+np.finfo(float).eps)*100
    S = R/(np.sqrt(valid_data_filter[V_elem])+np.finfo(float).eps)
    return S

"""calculate emotional factor "Q"
    -----------------------------------
    @valid_data_filter:
        Type: dataframe
        columns: date, open, high, low, close, preClose, preClose2, amount, vwap
        shape: (sample_num, 9)
    @S:
        Type: dataframe
        shape: (sample_num, 1)
    @trade_cum_ratio: 20%
    -----------------------------------
    Outputs:
    @Q:
        Type: float
"""
def calculate_Q(valid_data_filter, S, trade_cum_ratio = 0.2):
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
    return Q

def visualize_S(sort_valid_data, trade_cum_ratio = 0.2):
    fig = plt.figure()
    show_valid_data = pd.DataFrame(sort_valid_data['amount'].values, columns = ['amount'], index = sort_valid_data['date'])

    # ax = sort_valid_data[['amount']].unstack('date').plot(kind='bar', use_index=True)
    ax = show_valid_data.plot(kind='bar')
    ax2 = ax.twinx()
    ax2.scatter(ax.get_xticks(), S_arr, marker='o', c='red', s=10)
    ax2.set_ylim((np.min(S_arr)-np.min(S_arr)*0.1, np.max(S_arr)+np.max(S_arr)*0.1))
    ax.grid(True)
    ax2.grid(True)

    fig = plt.figure()
    ax = show_valid_data.plot(kind='bar')
    ax2 = ax.twinx()
    amount_cumsum = sort_valid_data['amount'].cumsum()
    all_amount_cumsum = amount_cumsum.values[-1]
    ax2.plot(ax.get_xticks(), amount_cumsum.values/all_amount_cumsum*100, c = 'g')

    # ax4 = ax3.twinx()
    ax2.plot(ax2.get_xticks(), [trade_cum_ratio*100]*len(ax.get_xticks()), c = 'y')
    x_ids = np.where((amount_cumsum.values/all_amount_cumsum) <= trade_cum_ratio)[0]
    ax2.plot([ax2.get_xticks()[x_ids[-1]]]*2, [0,100], 'm--')
    ax.grid(True)
    ax2.grid(True)
    plt.show()

if __name__ == '__main__':
    # read stock data
    path = './SH600000.feather'
    df = feather.read_dataframe(path)
    # feather.write_dataframe(df, output_path)

    start_date = "2016-09-30 14:30"
    end_date = "2016-09-30 15:00"
    valid_data = extract_valid_data_range(df, start_date, end_date)
    S = calculate_S(valid_data)

    # sort data by "S" -- descending
    S_sort_ids = np.argsort(S.values)
    S_arr = S.values[S_sort_ids[::-1]]
    sort_valid_data = valid_data.iloc[S_sort_ids[::-1]]
    # visualize Q
    visualize_S(sort_valid_data)

    Q = calculate_Q(valid_data, S)
