'''
date: 2021/1/22
@author: 流氓兔23333
content: 综合打分法(输入当期财务数据 --> 数据插补/非等全标准化/综合打分法多因子选股)
'''


# 财务指标
# 市现率(cash_ratio)对于不同的行业有不同的标准，不能说高好，也不能说低好
# 剔除市现率(total_liab)

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


# data = pro.index_classify(ts_code='881106.SI')
# df = pro.index_classify(level='L1', src='SW')
# data.head()

save_path = './data_path/'

values_col = ['q_profit_yoy', 'q_sales_yoy', 'roa', 'roe', 'profit_dedt', 'debt_to_assets', 
              'assets_turn', 'Amplitude_20', 'Amplitude_60', 'pct_chg_20', 'pct_chg_60', 
              'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'free_circ_mv', 'turnover_rate_f_mean20', 
              'turnover_rate_f_mean60', 'total_liab']

# 因子名称
values_name = ['净利润同比增长率(%)(单季度)', '营业收入同比增长率(%)(单季度)', '总资产报酬率', 
                '净资产收益率',  '扣除非经常性损益后的净利润', '资产负债率',
                '总资产周转率', '近20日日均振幅', '近60日日均振幅', '近20日日均收益率','近60日日均收益率',
                '市盈率TTM', '市净率（总市值/净资产）', '市销率TTM', '股息率TTM',
                '流通市值的对数', '进20日日均换手率', '进60日日均换手率', '负债合计']


# 因子方向
factor_positive = [1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
# 因子权重
factor_weight = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 2, 0.5, 0.5, 1]

# 因子->name/因子方向  dict
# factor2name_dict = {values_col[i]: values_name[i] for i in range(len(values_col))}
# pickle.dump(factor2name_dict, open(save_path + 'factor2name_dict.pkl', 'wb'))

# factor2positive_dict = {values_col[i]: factor_positive[i] for i in range(len(values_col))}
# pickle.dump(factor2positive_dict, open(save_path + 'factor2positive_dict.pkl', 'wb'))

# factor2weight_dict = {values_col[i]: factor_weight[i] for i in range(len(values_col))}
# pickle.dump(factor2weight_dict, open(save_path + 'factor2weight_dict.pkl', 'wb'))

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


# 综合打分模块 
def fun_syn_select(df_financial):
    '''
    综合打分法选股模型
    正向因子: (x - min(x)) / (max(x) - min(x))
    反向因子: (max(x) - x) / (max(x) - min(x))
    等权相加, 取前三百
    '''
    values_col = ['q_profit_yoy', 'q_sales_yoy', 'roa', 'roe', 'profit_dedt', 'debt_to_assets', 
              'assets_turn', 'free_circ_mv', 'total_liab']

    factor2positive_dict = pickle.load(open(save_path + 'factor2positive_dict.pkl', 'rb'))
    factor2weight_dict = pickle.load(open(save_path + 'factor2weight_dict.pkl', 'rb'))
    weight = [1, 1, 1, 1, 1, 1, 1, 1, 1]

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
    pickle.dump(code_buy_list, open(save_path + 'code_buy_list.pkl', 'wb'))
    return code_buy_list



# 回测模块
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

def back_test(v, start, end, df_scale, select_num=range(300), negtive=False):
    '''
    返回当期回测结果
    '''
    
    # 选股
    idx_ = df_scale[v].sort_values(ascending=negtive).index[select_num]
    code = df_scale['ts_code_all'].loc[idx_]
    
    df_cal = pro.trade_cal(start_date=start, end_date=end)
    df_cal = df_cal.query('(exchange=="SSE") & (is_open==1)')
    date = df_cal.cal_date

    idx_all = []
    idx_base_all = []
    for t in date:
        # 获取持仓期交易日在 数据库 的绝对位置索引
        idx_ = list(np.where(daily_data['trade_date'] == eval(t))[0])
        idx_all = idx_all + idx_
        idx_base_ = list(np.where(base_hs300['trade_date'] == eval(t))[0])
        idx_base_all = idx_base_all + idx_base_

    # 持仓期所有股票/hs300 数据
    df_t_all = daily_data.iloc[idx_all,:]
    df_t_base = base_hs300.iloc[idx_base_all,:]

    # hs300持仓期的累计收益率
    phg_base_series = df_t_base['pct_chg']
    total_return_all = (np.prod((phg_base_series/100 + 1).values) - 1 ) * 100
    
    idx_code = []
    for c in code:
        idx_ = list(np.where(df_t_all['ts_code'] == c)[0])
        idx_code = idx_code + idx_

    df_t_300 = df_t_all.iloc[idx_code,:]
    
    # 持仓期 选股组合 累计收益率
    df_t_300.index = df_t_300['trade_date']
    select_return = pd.Series(index=date)
    for t in date:
        # print(t,': ', len(df_t_300.loc[eval(t),'pct_chg']))
        try:
            select_return.loc[t] = df_t_300.loc[eval(t),'pct_chg'].mean()
        except:
            select_return.loc[t] = None
    total_return_300 = (np.prod((select_return/100 + 1).values) - 1 ) * 100

    df_t_base.index = date
    daily_return = pd.DataFrame({
        'select_return': select_return,
        'base_return': df_t_base['pct_chg']
    },index=date.values)

    # 胜率
    win_rate, win_t, total = compute_win(date, df_t_300, df_t_base)
    return total_return_300, total_return_all, win_rate, win_t, total, daily_return


def fun_back_test_yaer(y, select_num=range(100)):
    ''' 
    y 表示输入年份
    '''
    factor2positive_dict = pickle.load(open(save_path + 'factor2positive_dict.pkl', 'rb'))
    sacle_postive = lambda x: (x - np.nanmin(x)) /  (np.nanmax(x) - np.nanmin(x))
    sacle_negtive = lambda x: (np.nanmax(x) - x) /  (np.nanmax(x) - np.nanmin(x))


    if y == '2009':
        season_ = ['_season2', '_season3', '_season4']
    elif y == '2018':
        season_ = ['_season1', '_season2', '_season3']
    else:
        season_ = ['_season1', '_season2', '_season3', '_season4']
    d = ['0501', '0831', '0901', '1031', '1101', '0430']
    
    back_test_result = pd.DataFrame(index=range(len(season_)-1), 
            columns=['m_','return_300', 'return_base', 'win_rate', 'win_t', 'all_t'])
    back_daily_return = pd.DataFrame(columns=['select_return','base_return','season'])

    for i in range((len(season_)-1)):
        
        season_y = y+season_[i]
        df = pd.read_csv(str(new_path+'season_data_0/data_{}.csv').format(season_y),index_col=0)
        df = df.rename(columns={'ts_code_all.1':'ts_code_all'})

        df_value = df.copy()
        df_fillna = fun_fillna(df_value)
        for v in values_col:
            # 处理反向因子
            if df_fillna[v].dtype == 'O':
                df_fillna[v] = df_fillna[v].astype(np.float64)

            if factor2positive_dict[v] == 1:
                df_fillna[v] = sacle_postive(df_fillna[v])
            else:
                df_fillna[v] = sacle_negtive(df_fillna[v])
    
        df_scale = fun_scale(df_fillna)
        df_scale = df_scale.dropna(axis=0)
                
        # 等权求和
        factor_vote = (df_scale.loc[:, values_col]*weight).mean(axis=1)
        df_scale['factor_vote']  = factor_vote
        negtive = False
        v = 'factor_vote'

        if y == '2009':
            start, end = y + d[2*(i+1)], y + d[2*(i+1)+1]
            if season_[i] == '_season3':
                start, end = y + d[2*(i+1)], str(eval(y)+1) + d[2*(i+1)+1]
        else:
            start, end = y + d[2*i], y + d[2*i+1]
            if season_[i] == '_season3':
                start, end = y + d[2*i], str(eval(y)+1) + d[2*i+1]

        total_return_300, total_return_all, win_rate, win_t, all_t, daily_return = back_test(v=v, start=start, end=end, df_scale=df_scale, select_num=select_num, negtive=negtive)
        
        back_test_result.loc[i,'m_'] = season_y
        back_test_result.loc[i,'return_300'] = total_return_300
        back_test_result.loc[i,'return_base'] = total_return_all
        back_test_result.loc[i,'win_rate'] = win_rate
        back_test_result.loc[i,'win_t'] = win_t
        back_test_result.loc[i,'all_t'] = all_t

        daily_return['season'] = season_y
        back_daily_return = pd.concat((back_daily_return,daily_return),axis=0)
    return back_test_result, back_daily_return

def fun_back_test_yaer_all(select_num=range(100)):
    '''
    fill_na=False 表示正向因子
    '''
    col = ['m_', 'return_300', 'return_base', 'win_rate', 'win_t', 'all_t']
    back_test_result_all = pd.DataFrame(columns=col)
    back_daily_return_all = pd.DataFrame(columns=['select_return', 'base_return', 'season'])
    for y in YEAR[:]:
        print(y, '   已test')
        data_back_test, back_daily_return = fun_back_test_yaer(y, select_num)
        back_test_result_all = pd.concat((back_test_result_all, data_back_test), axis=0)
        back_daily_return_all = pd.concat((back_daily_return_all, back_daily_return), axis=0)

    return back_test_result_all, back_daily_return_all

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
    # 综合打分法选股数量
    select_range = range(300)
    select_range = range(int(input('选择前几的股票为候选池：')))

    if_testback = 'N'
    if_testback = input('综合打分法是否进行回测(Y/N)：')
    if if_testback == 'Y':
        # 数据回测
        
		# values_col = ['q_profit_yoy', 'q_sales_yoy', 'roa', 'roe', 'profit_dedt', 'debt_to_assets', 
        #       'assets_turn', 'free_circ_mv', 'total_liab']

        # 因子方向
        # factor_positive = [1, 1, 1, 1, 1, -1, 1, -1, -1]
        # 因子权重
        # weight = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        YEAR = np.array(range(2009,2019))
        YEAR = [str(y) for y in YEAR]

        path = 'D:/VSCode/pyStudy/ZJGS/project_research/'
        new_path = str(path+'data_require/')
        daily_data = pd.read_csv(str(new_path+'daily_data/data_not_st_all.csv'),index_col=0)
        base_hs300 = pd.read_csv(str(new_path+'base_hs300.csv'), index_col=0)

        back_fillna, return_fillna = fun_back_test_yaer_all(select_num=select_range)
        print(back_fillna)
        r_plot(return_fillna, 'syn')
    
    df_financial =  pd.read_csv(save_path+'df_financial.csv',  index_col=0)
    code_buy_list = fun_syn_select(df_financial)
    print(list(code_buy_list))





