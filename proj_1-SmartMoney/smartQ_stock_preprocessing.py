import feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility as u_q
import time
import cPickle as pickle
import os

__DEBUG = True

"""generate year-month index from 2013-04 to 2016-06"""
def generate_year_month_ids(begin_date, end_date):
    _begin_date = begin_date.split('-')
    _end_date = end_date.split('-')
    begin_year = int(_begin_date[0])
    end_year = int(_end_date[0])
    begin_mon = int(_begin_date[1])
    end_mon = int(_end_date[1])
    mon_num = (end_year - begin_year)*12+(end_mon-begin_mon)
    year_month_set = []

    for month_ in xrange(mon_num):
        year_step = (month_+begin_mon)//12
        year_step = year_step if (month_+begin_mon)%12!=0 else year_step-1
        month_curr = (month_+begin_mon)%12 if (month_+begin_mon)%12!=0 else 12
        year_month_str = '{:d}-{:02d}'.format(begin_year+year_step, month_curr)
        year_month_set.append(year_month_str)
    print "[generate_year_month_ids] -- (begin_date, end_date):", (begin_date, end_date)
    print len(year_month_set)
    print year_month_set
    return year_month_set

"""split the data by their month"""
def data_split_by_month(valid_data, year_month_set, flag=False):
    _data_mon_group = []
    valid_data_dateids = valid_data.set_index('date')
    for ids, _begin in enumerate(year_month_set):
        if ids < len(year_month_set)-1:
            _data_mon_group.append(valid_data_dateids[_begin:year_month_set[ids+1]])
        else:
            _data_mon_group.append(valid_data_dateids[_begin:])
    print "[data_split_by_month]-----------------------------------"
    print "data(month) before filtering:", len(_data_mon_group)
    # filter empty element
    if flag:
        del_ids = []
        for _ids,_ in enumerate(_data_mon_group):
            if _.empty:
                _data_mon_group.pop(_ids)
        print "data(months) after filtering:", len(_data_mon_group)
    return _data_mon_group

"""extract last 10 days in each month"""
def _extract_last_ndays_month(mon_data, offset_days = 10):
    mon_date_ids = pd.to_datetime(mon_data.index)
    last_days = mon_date_ids.day.unique()[-offset_days:]
    print "[_extract_last_ndays_month]-----------------------------"
    print "last {:d} days:".format(offset_days), last_days
    year = mon_date_ids.year.unique()[0]
    month = mon_date_ids.month.unique()[0]
    start_date = "{:d}-{:02d}-{:02d}".format(year, month, last_days[0])
    end_date = "{:d}-{:02d}-{:02d}".format(year, month, last_days[-1])
    print "date: ", (start_date, end_date)
    return (start_date, end_date)

"""filter 10 days data in each month"""
def filter_ndays_data_in_month(_data_mon_group, offset_days = 10):
    data_mon_group = []
    for mon_data in _data_mon_group:
        if mon_data.empty:
            data_mon_group.append(mon_data)
            continue
        date_start_end = _extract_last_ndays_month(mon_data, offset_days)
        days = int(date_start_end[1].split('-')[2])-int(date_start_end[0].split('-')[2])
        if days<offset_days:
            data_mon_group.append(pd.DataFrame({"open":"", "high":"", "low":"", "close":"", "preClose":"", "preClose2":"", "amount":"", "vwap":""}, index=[]))
        else:
            data_mon_group.append(mon_data[date_start_end[0]:])
    print "[filter_ndays_data_in_month]----------------------------"
    print len(data_mon_group)
    return data_mon_group

def calculate_smartQ_by_stock(data_mon_group, trade_cum_ratio = 0.2):
    print "[calculate_smartQ_by_stock]-----------------------------"
    trade_num_per_day = 60*4*10 # minutes*hours*days
    months_Q = []
    for mon_data in data_mon_group:
        if mon_data.empty:
            Q = 0
        else:
            if __DEBUG and mon_data.shape[0]!=trade_num_per_day:
                print "[calculate_smartQ_by_stock] mon_data.shape:", mon_data.shape
                print mon_data.tail(3)
            Q = u_q.smart_money_Q(mon_data, trade_cum_ratio)
        months_Q.append(Q)
    print len(months_Q)
    print "Q:", months_Q
    return months_Q

# statistic profit in each month
def calculate_profits(_data_mon_group):
    last_price = 'open'
    cur_price = 'close'
    month_profits = []
    for mon_data in _data_mon_group:
        if mon_data.empty:
            month_profits.append(0)
        else:
            profit = mon_data.iloc[-1][cur_price] / (mon_data.iloc[0][last_price]+np.finfo(float).eps) - 1.0
            month_profits.append(profit)
    """
    back-up for profits
    last_price = 0
    for mon_iter in xrange(len(_data_mon_group)-1):
        mon_data = _data_mon_group[mon_iter]
        print "mon_iter:", mon_iter
        if mon_data.empty or last_price==0:
            month_profits.append(0)
        else:
            profit = mon_data.iloc[0]['open'] / last_price - 1.0
            month_profits.append(profit)
        last_price = 0 if mon_data.empty else mon_data.iloc[0]['open']
    """
    print "[calculate_profits]------------------------------------"
    print "len(month_profits):",len(month_profits)
    print "month_profits:", month_profits
    return month_profits

def end2end_smartQ_by_stock(df, begin_date, end_date, offset_days = 10, trade_cum_ratio = 0.2):
    """extract data"""
    year_month_set = generate_year_month_ids(begin_date, end_date)
    _data_mon_group = data_split_by_month(df, year_month_set)
    data_mon_group = filter_ndays_data_in_month(_data_mon_group, offset_days)

    months_Q = []
    for mon_data in data_mon_group:
        Q = u_q.smart_money_Q(mon_data, trade_cum_ratio)
        months_Q.append(Q)
    print "[end2end_smartQ_by_stock]------------------------------"
    print len(months_Q)
    print "Q:", months_Q
    return months_Q

def statis_smartQ_and_profits_by_months(df, begin_date, end_date, offset_days = 10, trade_cum_ratio = 0.2):
    """extract data"""
    year_month_set = generate_year_month_ids(begin_date, end_date)
    _data_mon_group = data_split_by_month(df, year_month_set)
    data_mon_group = filter_ndays_data_in_month(_data_mon_group, offset_days)

    months_Q = calculate_smartQ_by_stock(data_mon_group, trade_cum_ratio)
    months_profits = calculate_profits(_data_mon_group)
    return months_Q, months_profits, data_mon_group

def calculate_Q_profit_of_feather(path):
    # path = './SH600000.feather'
    time_tag = time.time()
    df = feather.read_dataframe(path)

    """Form 2013-04-01 to 2016-5-31"""
    start_date = "2013-04-01 09:30"
    end_date = "2016-05-31 15:00"
    valid_data = u_q.extract_valid_data_range(df, start_date, end_date)
    begin_date_tmp = "2013-04"
    end_date_tmp = "2016-06"
    offset_days = 10
    trade_cum_ratio = 0.2
    (months_Q, months_profits, data_mon_group) = statis_smartQ_and_profits_by_months(valid_data, begin_date_tmp, end_date_tmp, offset_days, trade_cum_ratio)

    print "[calculate_Q_profit_of_feather]------------------------"
    print "months_Q:", months_Q
    print "months_profits", months_profits

    print "It takes ", time.time()-time_tag, "s"
    """ save Q and Profit using pickle package"""
    name_inds = path.rfind('/')
    save_path = os.path.join(path[:name_inds], 'Q_and_Profits', path[name_inds+1:-len(".feather")])
    print save_path
    with open(save_path, 'w') as f:
        pickle.dump(months_Q, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(months_profits, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(data_mon_group, f, pickle.HIGHEST_PROTOCOL)
    print "pickle dump over..."

    return months_Q, months_profits

if __name__ == '__main__':
    """query data path"""
    feather_path = r'/Volumes/Seagate Backup Plus Drive/workspace/1m data/1'
    feathers = os.listdir(feather_path)
    feat_names = []
    for i in feathers:
        if i.endswith('.feather'):
            filename = os.path.join(feather_path, i)
            feat_names.append(filename)
    for feat_ids, feat_path in enumerate(feat_names):
        # if feat_ids < 2796:
        #     continue
        print "No:", feat_ids, "/", len(feat_names), ", path:", feat_path
        (months_Q, months_profits) = calculate_Q_profit_of_feather(feat_path)
