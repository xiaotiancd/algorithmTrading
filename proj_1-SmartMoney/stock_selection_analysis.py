#!/usr/bin/python
# -*- coding: latin-1 -*-

u"""筛选给定时间段数据"""
import feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility as u_q
import os
import cPickle as pickle
import scipy.stats as spst

"""
initialize preprocessing data from path "feather_path"
"""
def init_preprocessing_data_path(feather_path):
    # feather_path = r'/Volumes/Seagate Backup Plus Drive/workspace/1m data/1/Q_and_Profits'
    feathers = os.listdir(feather_path)
    feat_names = []
    for i in feathers:
        filename = os.path.join(feather_path, i)
        feat_names.append(filename)
    print len(feat_names)
    return feat_names

"""
load Qs and Profits of all A stocks,
and remove the 'ST' stocks and published less than 60 days
"""
def load_Astock_Q_profits(feat_names, load_stocks_mons_flag=False):
    # load st_symbol
    base_path = '/Volumes/Seagate Backup Plus Drive/workspace/1m data'
    with open(os.path.join(base_path, 'st_symbol'), 'rb') as f:
        st_symbol = pickle.load(f)
        print st_symbol.shape
    # load publicated less than 60 days
    with open(os.path.join(base_path, 'publicated_less_60'), 'rb') as f:
        publicated_lq_60 = pickle.load(f)
        print publicated_lq_60.shape

    Astock_Q = np.array([])
    Astock_profits = np.array([])
    data_mon_group = []
    filter_num = 0
    for feat_ids, feat_path in enumerate(feat_names):
        print "No:", feat_ids, "/", len(feat_names), ", path:", feat_path
        nameids = feat_path.rfind('/')
        stock_symbol = feat_path[nameids+1+2:nameids+1+8]
        """filter stocks which is ST or publicated less than 60 days"""
        if stock_symbol in st_symbol or stock_symbol in publicated_lq_60:
            print stock_symbol, " in numpy-'st_symbol' or numpy-'publicated_lq_60'\n",
            filter_num = filter_num + 1
            continue
        with open(feat_path, "rb") as f:
            try:
                stock_iter_Q = pickle.load(f)
                stock_iter_profits = pickle.load(f)
                if load_stocks_mons_flag:
                    stock_iter_group_data = pickle.load(f)
                print "stock_iter_Q.len:", len(stock_iter_Q)
                print "stock_iter_profits.len:", len(stock_iter_profits)
                if load_stocks_mons_flag:
                    print "stock_iter_group_data.len:", len(stock_iter_group_data)
                if Astock_Q.size == 0:
                    Astock_Q = np.array(stock_iter_Q)[np.newaxis,:]
                    Astock_profits = np.array(stock_iter_profits)[np.newaxis,:]
                else:
                    Astock_Q = np.vstack((Astock_Q, stock_iter_Q))
                    Astock_profits = np.vstack((Astock_profits, stock_iter_profits))
                    if load_stocks_mons_flag:
                        data_mon_group.append(stock_iter_group_data)
            except:
                print "catching exception!"
                pass
        print "Astock_Q.shape:", Astock_Q.shape
        print "Astock_profits.shape:", Astock_profits.shape
        if load_stocks_mons_flag:
            print "data_mon_group.len:", len(data_mon_group)
    print "filter stock num:", filter_num
    if load_stocks_mons_flag:
        return Astock_Q, Astock_profits, data_mon_group
    return Astock_Q, Astock_profits

def filter_Q_zeros(Astock_Q, Astock_profits):
    AstockQ_tmp = []
    Astock_profits_tmp = []
    """filter zero-Q one by one
    for mon_item in xrange(Astock_Q.shape[1]):
        Q_nonzeros_ids = np.where(Astock_Q[:, mon_item]!=0)[0]
        AstockQ_tmp.append(Astock_Q[Q_nonzeros_ids, mon_item])
        Astock_profits_tmp.append(Astock_profits[sel, :][Q_nonzeros_ids, mon_item])
        print Q_nonzeros_ids.shape
    """

    print "Astock_Q.shape:", Astock_Q.shape
    print "Astock_profits.shape:", Astock_profits.shape
    """summarize Q zeros at stocks in all months"""
    Q_all_zeros_ids = np.array([])
    for mon_item in xrange(Astock_Q.shape[1]):
        Q_zeros_ids = np.where(Astock_Q[:, mon_item]==0)[0]
    #     print type(Q_zeros_ids)
    #     print Q_zeros_ids.shape
        if Q_all_zeros_ids.size == 0:
            Q_all_zeros_ids = Q_zeros_ids
        else:
            Q_all_zeros_ids = np.concatenate((Q_all_zeros_ids, Q_zeros_ids))
    print Q_all_zeros_ids.shape
    Q_zeros_ids = np.unique(Q_all_zeros_ids)
    print Q_zeros_ids.shape

    """filter Astocks whose Q is zero"""
    sel = np.ones(Astock_Q.shape[0], dtype = np.int32)
    sel[Q_zeros_ids] = 0
    sel = sel.astype(bool)

    print "before filter: ", Astock_Q.shape
    AstockQ_tmp = Astock_Q[sel, :]
    Astock_profits_tmp = Astock_profits[sel, :]
    print "after filter: ", AstockQ_tmp.shape
    return AstockQ_tmp, Astock_profits_tmp


"""
 compute RankIC by months, using Q-(current month) & Profits-(next month)
 Astock_Q: (stocks_num, months_num)
"""
def compute_RankICs(Astock_Q, Astock_profits):
    print "Astock_Q.shape:", Astock_Q.shape
    print "Astock_profits.shape:", Astock_profits.shape
    """sort Astock_Q & Astock_profits respectively"""
    Astock_Q_ids = np.argsort(Astock_Q, axis=0)
    Astock_profits_ids = np.argsort(Astock_profits, axis=0)

    RankICs = []
    pvals = []
    for mon_i in xrange(Astock_Q_ids.shape[1]-1):
        RankIC, pval = spst.spearmanr(Astock_Q_ids[:,mon_i], Astock_profits_ids[:,mon_i+1], axis=0)
        RankICs.append(RankIC)
        pvals.append(pval)
    print len(RankICs)
    print(RankICs)
    return RankICs, pvals

"""
stock_date_path = '/Users/bowen/workspace/Quant/algorithmTrading/proj_1-SmartMoney/stock_date'
"""
def generate_date_set(stock_date_path, data_mon_group=None):
    if data_mon_group!=None:
        item = data_mon_group[1]
        for mon_itm in item[1:]:
            date_set.append(mon_itm.iloc[-1].name[:10])
        with open(stock_date_path, 'rb') as f:
            pickle.dump(date_set, f, pickle.HIGHEST_PROTOCOL)
    else:
        date_set = []
        with open(stock_date_path, 'rb') as f:
            date_set = pickle.load(f)
    print date_set
    print len(date_set)
    return date_set

def visualize_RankICs(RankICs, date_set):
    RankICs = np.array(RankICs)
    show_valid_data = pd.DataFrame(RankICs, index=date_set[:len(RankICs)], columns=['Rankic'])
    ax = show_valid_data.plot(kind='bar')
    ax.grid(True)
    plt.show()

    RankICs_pos = RankICs[RankICs>0.03]
    RankICs_neg = RankICs[RankICs<-0.03]
    # RankICs_pos =
    print u"RankIC num:", len(RankICs) if type(RankICs)==np.ndarray else 1
    print u"RankIC positive num:", len(RankICs_pos) if type(RankICs_pos)==np.ndarray else RankICs_pos
    print u"RankIC negative num:", len(RankICs_neg) if type(RankICs_neg)==np.ndarray else RankICs_neg
    print u"RankIC non-salient num:", len(RankICs) - len(RankICs_pos) - len(RankICs_neg)

u"""分组表现与多空对冲净值"""
def statics_group_return(Astock_Q, Astock_profits, group_num = 5):
    print "group num:", group_num
    print "Astock_Q.shape:", Astock_Q.shape
    print "Astock_Q_ids.shape:", Astock_Q_ids.shape
    Astock_Q_ids = np.argsort(Astock_Q, axis=0)

    group_profit = []
    for group_item in xrange(group_num):
        begin_ids = int(group_item * (Astock_Q_ids.shape[0] * 1.0/group_num))
        end_ids = int(begin_ids + (Astock_Q_ids.shape[0] * 1.0/group_num))
        profit_group_item = np.array([])
        for mon_item in xrange(Astock_Q_ids.shape[1]-1):
            if profit_group_item.size == 0:
                profit_group_item = Astock_profits[Astock_Q_ids[begin_ids:end_ids, mon_item], mon_item+1][:,np.newaxis]
            else:
                profit_group_item = np.hstack((profit_group_item
                                               , Astock_profits[Astock_Q_ids[begin_ids:end_ids,mon_item], mon_item+1][:,np.newaxis]))
    group_profit.append(profit_group_item)
    print group_profit[group_item].shape
    return group_profit

u"""分组表现与多空对冲净值曲线"""
def visualize_group_return(group_profit, date_set):
    group_curve = np.array([])
    date_indexes = pd.to_datetime(date_set[:-1])
    for item in xrange(group_num):
        curve = np.mean(group_profit[item], axis=0)[:, np.newaxis]
    #     print curve.shape
        if group_curve.size == 0:
            group_curve = curve
        else:
            group_curve = np.hstack((group_curve, curve))
    print "group_curve.shape:", group_curve.shape
    curve_data = pd.DataFrame(group_curve, index=date_indexes, columns=['group 1','group 2'
                                                                         ,'group 3','group 4','group 5'])
    curve_data = curve_data.cumsum()

    fig = plt.figure()
    ax = curve_data.plot()

    ax2 = ax.twinx()
    g1_g5_diff = curve_data['group 1'] - curve_data['group 5']
    print ax.get_xticks()
    ax2.plot(date_indexes, g1_g5_diff, 'k--')

    ax.grid(True)
    plt.show()
    return group_curve


u"""分组相对于市场平均收益的超额收益（月度均值）"""
def visualize_group_excess_return(group_curve, Astock_profits):
    print "group_curve.shape:", group_curve.shape
    print "Astock_profits.shape:", Astock_profits[:,1:].shape
    all_mkt_mon_mean = np.mean(Astock_profits[:,1:], axis = 0)
    print "mean_group_curve.shape:", group_curve.shape
    print "all_mkt_mon_mean.shape:", all_mkt_mon_mean.shape
    group_profits_rt = np.array([])
    for group_item in xrange(group_num):
        group_profits_item = np.divide(group_curve[:, group_item] - all_mkt_mon_mean, abs(all_mkt_mon_mean))*100.0
        if group_profits_rt.size == 0:
            group_profits_rt = group_profits_item[:,np.newaxis]
        else:
            group_profits_rt = np.hstack((group_profits_rt, group_profits_item[:,np.newaxis]))
    print group_profits_rt.shape
    group_profits_rt = np.mean(group_profits_rt, axis=0)

    show_group_profits_rt = pd.DataFrame(group_profits_rt, index=['group 1','group 2'
                                                   ,'group 3','group 4','group 5'], columns=['Excess returns'])
    ax = show_group_profits_rt.plot(kind='bar')
    ax.grid(True)
    plt.show()
    return group_profits_rt

if __name__ == '__main__':
    feather_path = r'/Volumes/Seagate Backup Plus Drive/workspace/1m data/1/Q_and_Profits_2'
    feat_names = init_preprocessing_data_path(feather_path)
    load_stocks_mons_flag=False
    (Astock_Q, Astock_profits) = load_Astock_Q_profits(feat_names, load_stocks_mons_flag)
    Astock_Q = Astock_Q[:,1:]
    Astock_profits = Astock_profits[:,1:]
    (Astock_Q, Astock_profits) = filter_Q_zeros(Astock_Q, Astock_profits)
    (rankICs, pvals) = compute_RankICs(Astock_Q, Astock_profits)
    data_mon_group=None
    stock_date_path = '/Users/bowen/workspace/Quant/algorithmTrading/proj_1-SmartMoney/stock_date'
    date_set = generate_date_set(stock_date_path, data_mon_group)
    visualize_RankICs(rankICs, date_set)

    u"""分组表现与多空对冲净值曲线 & 分组相对于市场平均收益的超额收益（月度均值）"""
    group_num = 5
    group_profit = statics_group_return(Astock_Q, Astock_profits, group_num)
    group_curve = visualize_group_return(group_profit, date_set)
    group_profits_rt = visualize_group_excess_return(group_curve, Astock_profits)
