'''
date: 2021/1/23
@author: 流氓兔23333
content: 训练xgboost选股模型  在综合发分选股的基础上进行选股
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

import os, warnings, pickle
warnings.filterwarnings('ignore')

import tushare as ts
pro = ts.pro_api('44e2ca5912fe54773b542c2259135a84c05ff75e75e60bc486da5de6')


save_path = './data_path/'

values_col = ['q_profit_yoy', 'q_sales_yoy', 'roa', 'roe', 'profit_dedt', 'debt_to_assets', 
              'assets_turn', 'Amplitude_20', 'Amplitude_60', 'pct_chg_20', 'pct_chg_60', 
              'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'free_circ_mv', 'turnover_rate_f_mean20', 
              'turnover_rate_f_mean60', 'total_liab']



# 数据去重
def fun_dropsame(df1):
    '''
    去除重复数据
    '''
    return df1.drop_duplicates()

# 极值推压法
def fun_Extremum_push(x):
    '''
    将前5%的因子值以第5%处的因子值重新赋值,
    将后5%的因子值以第95%处的因子值重新赋值,
    '''
    new_x = x.copy()
    length_x = len(x)
    # 前5%
    idx_0 = x.sort_values().iloc[:np.int(length_x*0.05)].index
    value_0 = x.sort_values().iloc[np.int(length_x*0.05)]
    # 后5%
    idx_1 = x.sort_values(ascending=False)[:np.int(length_x*0.05)].index
    value_1 = x.sort_values(ascending=False).iloc[np.int(length_x*0.05)]

    new_x.loc[idx_0] = value_0
    new_x.loc[idx_1] = value_1
    
    return new_x

# 非等权标准化
def fun_scale(df):
    '''
    return 非等权标准化
    不删除缺失数据
    '''
    
    # first step 剔除含有NA的行
    df_dropna = fun_dropsame(df)

    for v in values_col:
        df_dropna[v] = fun_Extremum_push(x=df_dropna[v])
    
    # fif 用均值代替
    df_dropna['fif'] = df_dropna['fif'].fillna(df_dropna['fif'].mean())

    # 权重求解
    weight = (df_dropna['free_share']*df_dropna['close']*df_dropna['fif']) / np.sum((df_dropna['free_share']*df_dropna['close']*df_dropna['fif']))
    # print(np.sum(weight))
    
    for v in values_col:
        # 处理inf值
        idx_ = np.where(df_dropna[v] == np.inf)[0]
        df_dropna[v].iloc[idx_] = None
        u_v = (weight*df_dropna[v]).sum()
        sigma_v = np.sqrt((weight*((df_dropna[v] - u_v)**2)).sum())
        df_dropna[v] = (df_dropna[v] - u_v) / sigma_v

    return df_dropna

# K近邻 缺失值插补 
def fun_fillna(df1, k=30):
    ''' 
    基于无缺失列数据的K近邻插补
    返回填补后的非等权标准化数据
    '''
    from sklearn.neighbors import KDTree

    df1 = fun_dropsame(df1)

    df_dropna = df1.copy()
    df_dropna = df_dropna.drop_duplicates() 

    # df = df_dropna
    df_scale = fun_scale(df_dropna)

    # 需要做插补的数据
    df_dist = df_scale.loc[:, values_col]
    
    df_dropna = df_dist.dropna(axis=1) 
    code_na = df_dist.index[np.where(df_dist.isnull().any(axis=1) == True)[0]]
    
    tree = KDTree(df_dropna, leaf_size=30, metric='euclidean')
    for c in code_na:
        # print(c)
        # 一只股票存在多行的问题
        _, ind = tree.query(np.array(df_dropna.loc[c,:]).reshape(1, -1), k=k) 
        code_ind = df_dropna.index[ind][0]
        df = df_dist.loc[code_ind,:]
        value_na = df_dist.loc[c,:].index[np.where(df_dist.loc[c,:].isnull() == True)[0]]
        for v in value_na:
            df_scale.loc[c,v] = df.loc[:, v].mean()

    return df_scale




# 训练选股模型
def fun_train_clf(y, data_tagging):
    ''' 
    返回选股模型
    '''
    import xgboost as xgb
    import sklearn.metrics as ms
    from sklearn.model_selection import train_test_split

    y = eval(y)

    year_ = []
    for i in range(len(data_tagging)):
        year_.append(data_tagging['m'].iloc[i][:4])
    data_tagging['year_'] = [eval(i) for i in year_]

    data_tagging_new = data_tagging.iloc[np.where(data_tagging['year_'] >= y)[0], :]
    data_tagging_train = data_tagging.iloc[np.where(data_tagging_new['year_'] < y+3)[0], :]
    data_tagging_train['year_'] = data_tagging_train['year_'].astype(str)
    
    X, y = data_tagging.loc[:,values_col], data_tagging.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.fit(X_train, y_train)
    train_pred = clf_xgb.predict(X_train)
    test_pred = clf_xgb.predict(X_test)

    # 预测准确率
    train_acc = ms.accuracy_score(y_train, train_pred)
    test_acc = ms.accuracy_score(y_test, test_pred)
    print('train_acc: ', train_acc, '\ntest_acc: ', test_acc)
    return clf_xgb


''' 综合打分模块 '''
def fun_syn_select(df_financial):
    '''
    综合打分法选股模型
    正向因子: (x - min(x)) / (max(x) - min(x))
    反向因子: (max(x) - x) / (max(x) - min(x))
    等权相加, 取前三百
    '''

    factor2positive_dict = pickle.load(open(save_path + 'factor2positive_dict.pkl', 'rb'))
    factor2weight_dict = pickle.load(open(save_path + 'factor2weight_dict.pkl', 'rb'))
    weight = [1,1,1,1,0.5,0.5, 0.5, 0.5, 1,0.5,1,1,1,1,1,1,1,1,0.5]

    sacle_postive = lambda x: (x - np.nanmin(x)) /  (np.nanmax(x) - np.nanmin(x))
    sacle_negtive = lambda x: (np.nanmax(x) - x) /  (np.nanmax(x) - np.nanmin(x))

    df_postive = df_financial.copy()    
    df_fillna = fun_fillna(df_postive)

    for v in values_col:
        if df_fillna[v].dtype == 'O':
            df_fillna[v] = df_fillna[v].astype(np.float64)

        if factor2positive_dict[v] == 1:
            df_fillna[v] = sacle_postive(df_fillna[v])
        else:
            df_fillna[v] = sacle_negtive(df_fillna[v])
    

    df_scale = fun_scale(df_fillna)
    df_dropna = df_scale.dropna(axis=0)

    # 等权相加
    syn_scores = pd.Series([(df_dropna.loc[:,values_col].iloc[i,:] * weight)\
                            .sum() for i in range(len(df_dropna))], 
                            index=df_dropna['ts_code_all']).sort_values(ascending=False)

    code_buy_list = syn_scores.index[select_range]

    return code_buy_list





''' 滚动预测(模型不发生变化) '''
def ma_back_test(start, end, code_buy):
    '''
    输入 持仓期， 持仓股票
    '''
    code = code_buy
    
    df_cal = pro.trade_cal(start_date=start, end_date=end)
    df_cal = df_cal.query('(exchange=="SSE") & (is_open==1)')
    date = df_cal.cal_date

    idx_all = []
    idx_base_all = []
    for t in date:
        idx_ = list(np.where(daily_data['trade_date'] == eval(t))[0])
        idx_all = idx_all + idx_
        idx_base_ = list(np.where(base_hs300['trade_date'] == eval(t))[0])
        idx_base_all = idx_base_all + idx_base_

    # 持仓期 所有股票/hs300 数据库
    df_t_all = daily_data.iloc[idx_all,:]
    df_t_base = base_hs300.iloc[idx_base_all,:]
    
    # hs300收益率
    phg_base_series = df_t_base['pct_chg']
    total_return_all = (np.prod((phg_base_series/100 + 1).values) - 1 ) * 100

    idx_code = []
    for c in code:
        idx_ = list(np.where(df_t_all['ts_code'] == c)[0])
        idx_code = idx_code + idx_

    df_t_300 = df_t_all.iloc[idx_code]
    # 持仓期 选股组合累计收益率
    df_t_300.index = df_t_300['trade_date']
    select_return = pd.Series(index=date)
    for t in date:
        # 计算每日平均
        select_return.loc[t] = df_t_300.loc[eval(t),'pct_chg'].mean()
    # 计算累计
    total_return_300 = (np.prod((select_return/100 + 1).values) - 1 ) * 100

    df_t_base.index = date
    daily_return = pd.DataFrame({
        'select_return': select_return,
        'base_return': df_t_base['pct_chg']
    },index=date.values)

    # 胜率
    win_rate, win_t, total = compute_win(date, df_t_300, df_t_all)
    
    return total_return_300, total_return_all, win_rate, win_t, total, daily_return 

def compute_win(date, df_t_300, df_t_base):
    win = 0
    for t in date:
        idx_1 = np.where(df_t_300['trade_date'] == eval(t))[0]
        df_1 = df_t_300.iloc[idx_1,:]
        idx_0 = np.where(df_t_base['trade_date'] == eval(t))[0]
        df_0 = df_t_base.iloc[idx_0,:]
        if df_1['pct_chg'].mean() > df_0['pct_chg'].mean():
            win += 1
    return win/len(date), win, len(date)

def back_test_all(year_test, clf_xgb, select_sum=range(100)):
    d = ['0501', '0831', '0901', '1031', '1101', '0430']

    back_test_result_all = pd.DataFrame(
            columns=['m_','return_300', 'return_base', 'win_rate', 'win_t', 'all_t'])
    back_test_daily_return = pd.DataFrame(columns=['select_return', 'base_return', 'season'])
    
    for y in year_test:
        print(y, '已回测')
        if y == '2009':
            season_ = ['_season2', '_season3', '_season4']
        elif y == '2018':
            season_ = ['_season1', '_season2', '_season3']
        else:
            season_ = ['_season1', '_season2', '_season3', '_season4']
        
        back_test_result = pd.DataFrame(index=range(len(season_)-1), 
            columns=['m_','return_300', 'return_base', 'win_rate', 'win_t', 'all_t'])
        
        for i in range((len(season_)-1)):
            season_y = y+season_[i]
            df = pd.read_csv(str(new_path+'season_data_0/data_{}.csv').format(season_y),index_col=0)
            df_copy = df.copy().rename(columns={'ts_code_all.1':'ts_code_all'})

            code_buy_list = fun_syn_select(df_copy)
            df_copy = df_copy[df_copy['ts_code_all'].isin(code_buy_list)]

            df_scale = fun_fillna(df_copy)
            df_scale = df_scale.dropna(axis=0)

            df_test = df_scale.loc[:, values_col]

            # 概率预测
            prob_postive = fun_predict_prob(clf_xgb, df_test)
            idx_buy = prob_postive.sort_values(ascending=False).index[xgb_select]
            
            code_buy = df_scale['ts_code_all'].loc[idx_buy]
            
            start, end = y + d[2*i], y + d[2*i+1]
            if season_[i] == '_season3':
                start, end = y + d[2*i], str(eval(y)+1) + d[2*i+1]

            total_return_300, total_return_all, win_rate, win_t, all_t, daily_return = ma_back_test(start, end, code_buy)
            
            back_test_result.loc[i,'m_'] = season_y
            back_test_result.loc[i,'return_300'] = total_return_300
            back_test_result.loc[i,'return_base'] = total_return_all
            back_test_result.loc[i,'win_rate'] = win_rate
            back_test_result.loc[i,'win_t'] = win_t
            back_test_result.loc[i,'all_t'] = all_t

            daily_return['season'] = season_y
            back_test_daily_return = pd.concat((back_test_daily_return,daily_return),axis=0)

        back_test_result_all = pd.concat((back_test_result_all, back_test_result))

    return back_test_result_all, back_test_daily_return

def fun_predict_prob(clf, x_test):
    '''
    返回预测概率
    '''
    prob = clf.predict_proba(x_test)
    pred = clf.predict(x_test)
    prob_postive = np.array([i[1] for i in prob])
    prob_postive = pd.Series(prob_postive, index=x_test.index)
    return prob_postive

def back_test_all_season(year_test, clf_xgb, select_sum=range(100)):
    d = ['0501', '0831', '0901', '1031', '1101', '0430']

    back_test_result_all = pd.DataFrame(
            columns=['m_','return_300', 'return_base', 'win_rate', 'win_t', 'all_t'])
    back_test_daily_return = pd.DataFrame(columns=['select_return', 'base_return', 'season'])
    
    for y in year_test:
        print(y, '已回测')
        
        back_test_result = pd.DataFrame(index=range(1), 
            columns=['m_','return_300', 'return_base', 'win_rate', 'win_t', 'all_t'])
        
        season_y = y
        df = pd.read_csv(str(new_path+'season_data_0/data_{}.csv').format(season_y),index_col=0)
        
        df_scale = fun_fillna(df)
        df_scale = df_scale.dropna(axis=0)

        df_test = df_scale.loc[:, values_col]

        # 概率预测
        prob_postive = fun_predict_prob(clf_xgb, df_test)
        idx_buy = prob_postive.sort_values(ascending=False).index[select_sum]
        # idx_buy = df_scale['pct_chg_sum'].sort_values(ascending=False).index[select_sum]
        
        code_buy = df_scale['ts_code_all'].loc[idx_buy]
        
        year, i = y[:4], eval(y[-1])-1
        start, end = year + d[2*i], year + d[2*i+1]
        if season_[i] == '_season3':
            start, end = y + d[2*i], str(eval(y)+1) + d[2*i+1]

        total_return_300, total_return_all, win_rate, win_t, all_t, daily_return = ma_back_test(start, end, code_buy)
        
        back_test_result.loc[0,'m_'] = season_y
        back_test_result.loc[0,'return_300'] = total_return_300
        back_test_result.loc[0,'return_base'] = total_return_all
        back_test_result.loc[0,'win_rate'] = win_rate
        back_test_result.loc[0,'win_t'] = win_t
        back_test_result.loc[0,'all_t'] = all_t

        daily_return['season'] = season_y
        back_test_daily_return = pd.concat((back_test_daily_return, daily_return),axis=0)

        back_test_result_all = pd.concat((back_test_result_all, back_test_result))

    return back_test_result_all, back_test_daily_return

def r_plot(daily_return, v):
    '''
    回测累计收益图
    '''
    try:
        daily_return_0 = data_change_date(daily_return)
    except:
        pass
    plt.plot(np.cumsum(daily_return_0['select_return']),'r-', linewidth = '2')
    plt.plot(np.cumsum(daily_return_0['base_return']), linestyle='-', color='#000000', linewidth = '2')
    plt.legend(fontsize=20)
    plt.xlabel('时间', size=20)
    plt.ylabel('累计收益率（%）', size=20)
    plt.tick_params(labelsize=20)
    plt.title('factor：{} 回测收益图'.format(v), fontsize=20)
    # plt.savefig(str(path+'{}.png').format(v))
    plt.show()

def data_change_date(df):
    # 将 trade_date 转化为可识别的datetime
    df_copy = df.copy()
    import datetime
    plot_axis = list(map(lambda x: datetime.datetime.strptime(x, "%Y%m%d"), df_copy.index))
    df_copy.index = plot_axis
    return df_copy




if __name__ == "__main__":   
    
    # 综合打分股票候选池
    select_range = range(300)
    select_range = range(int(input('选择前几的股票为候选池：')))

    # xgb选股数量
    xgb_select = range(10)
    xgb_select = range(int(input('选股数量：')))

    YEAR = np.array(range(2009,2019))
    YEAR = [str(y) for y in YEAR]
    path = 'D:/VSCode/pyStudy/ZJGS/project_research/'
    new_path = str(path+'data_require/')
    data_tagging = pd.read_csv(str(path+'data_tagging/data_tagging_yearall.csv'), index_col=0)
    data_tagging = data_tagging.rename(columns={'ts_code_all.1':'ts_code_all'})
    clf_xgb = fun_train_clf(YEAR[0], data_tagging)

    daily_data = pd.read_csv(str(new_path+'daily_data/data_not_st_all.csv'),index_col=0)
    base_hs300 = pd.read_csv(str(new_path+'base_hs300.csv'), index_col=0)

    year_test = [str(i) for i in range(2012,2019)]
    back_test_result_0, back_test_daily_return_0 = back_test_all(year_test, clf_xgb, select_range)
    r_plot(back_test_daily_return_0, 'xgb')

    # 基于当前数据进行选股
    df_financial =  pd.read_csv(save_path+'df_financial.csv',  index_col=0)
    df_test = df_financial.copy()
    df_test = df_financial[df_financial['ts_code_all'].isin(code_buy_list)]
    
    df_test_fillna = fun_fillna(df_test)
    df_test_scale = fun_scale(df_test_fillna)
    df_test_scale = df_test_scale.dropna(axis=0)
    code_prob = fun_predict_prob(clf_xgb, df_test_scale.loc[:, values_col])
    idx_buy = code_prob.sort_values(ascending=False).index[xgb_select]
    code_buy = df_test_scale['ts_code_all'].loc[idx_buy]
    print(code_buy)

