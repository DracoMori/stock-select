'''
date: 2021/1/19
@author: 流氓兔23333
content: 数据获取(return 最近的财务数据)
'''

import tushare as ts
pro = ts.pro_api('44e2ca5912fe54773b542c2259135a84c05ff75e75e60bc486da5de6')

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm 
import json
import os, warnings, pickle
warnings.filterwarnings('ignore')

save_path = './data_path/'




# load js文件
def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic


#获取当前股票最近3个月的财务数据
# return ['ts_code', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'circ_mv', 
# 'free_share_ratio', 'turnover_rate_f_mean20', 'turnover_rate_f_mean60']
def fun_value_df():
    date = DATE.tolist()
    field = ['ts_code','trade_date','pe_ttm','pb','ps_ttm','dv_ttm','circ_mv','free_share_ratio',
            'turnover_rate_f', 'float_share']
    value_df = pd.DataFrame(columns=field)
    for t in tqdm(date):
        while(True):
            try:
                df = pro.daily_basic(trade_date=t, 
                    fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ttm,circ_mv,\
                            float_share,total_share,turnover_rate_f') 
                break
            except:
                continue
        df = df[df['ts_code'].isin(code_list)]
        df['free_share_ratio'] = df['float_share'] / df['total_share']
        value_df = pd.concat((value_df, df.drop(['total_share'], axis=1)), axis=0)
    
    mean_20 = lambda x: x.sort_values('turnover_rate_f', ascending=False)['turnover_rate_f']\
        .iloc[:20].mean()
    mean_60 = lambda x: x.sort_values('turnover_rate_f', ascending=False)['turnover_rate_f']\
        .iloc[:60].mean()
    
    turnover_rate_f_mean20 = value_df.groupby('ts_code')['trade_date', 'turnover_rate_f'].apply(mean_20)
    turnover_rate_f_mean60 = value_df.groupby('ts_code')['trade_date', 'turnover_rate_f'].apply(mean_60)

    value_df_adj = value_df.groupby('ts_code')['pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'circ_mv', 
                'free_share_ratio', 'float_share']\
                .apply(lambda x: x.mean())
    value_df_adj['turnover_rate_f_mean20'] = turnover_rate_f_mean20
    value_df_adj['turnover_rate_f_mean60'] = turnover_rate_f_mean60
    value_df = value_df_adj.reset_index()

    return value_df


# 离今日最近的每股现金流量净额(财务指标) [cfps]
# [q_profit_yoy, q_sales_yoy, roa, roe, profit_dedt, debt_to_assets, assets_turn, cash_ratio ]
# ['Amplitude_20','Amplitude_60', 'pct_chg_20', 'pct_chg_60']
# [净利润同比增长率, 营业收入同比增长率, 总资产报酬率, 净资产收益率, 扣除非经常性损益后利润, 
# 资产负债率, 资产周转率, 市现率]
def fun_cfps_df():
    cfps_df = pd.DataFrame(columns=['ts_code', 'cfps', 'q_profit_yoy', 'q_sales_yoy', 
        'roa', 'roe', 'profit_dedt', 'debt_to_assets', 'assets_turn', 'Amplitude_20',
        'Amplitude_60', 'pct_chg_20', 'pct_chg_60', 'cash_ratio', 'close'])
    i = 0
    for c in tqdm(code_list):
        while(True):    
            try:
                df = pro.fina_indicator(ts_code=c, start_date=START, end_date=END, 
                    fields='ts_code,end_date,cfps,q_profit_yoy,q_sales_yoy,roa,roe,\
                    profit_dedt,debt_to_assets, assets_turn')
                if len(df) != 0:
                    df = df.sort_values('end_date',ascending=False).drop(['end_date'],axis=1).iloc[0]
                else:
                    # print('数据缺失')
                    df = pd.Series(index=cfps_df.columns.tolist())
                    df.loc[['ts_code','cfps','q_profit_yoy','q_sales_yoy','roa','roe',\
                            'profit_dedt','debt_to_assets', 'assets_turn']] = [c] + [None]*8
                df_ptc_chg = pro.daily(ts_code=c, start_date=START, end_date=END)\
                    .loc[:,['ts_code', 'trade_date','high', 'low', 'pre_close','pct_chg', 'close']]\
                        .sort_values('trade_date', ascending=False)
                
                df.loc['close'] = df_ptc_chg['close'].iloc[0]
                # 计算市现率 = 最新日收盘价/cfps
                df['cash_ratio'] = df_ptc_chg['close'].iloc[0] / df.loc['cfps']
                # 计算 近20/60日日均 振幅和收益率
                Amplitude = (df_ptc_chg['high'] - df_ptc_chg['low']) / df_ptc_chg['pre_close']
                df['Amplitude_20'], df['Amplitude_60'] = Amplitude[:20].mean(), Amplitude[:60].mean()
                df['pct_chg_20'], df['pct_chg_60'] = df_ptc_chg['pct_chg'][:20].mean(), df_ptc_chg['pct_chg'][:60].mean()
                break
            except:
                continue
        
        df = pd.DataFrame({i: df.loc[i] for i in cfps_df.columns.tolist()}, index=[i])
        cfps_df = pd.concat((cfps_df, df), axis=0)    
        i += 1  
    
    return cfps_df


# 总负债 total_liab
def fun_total_liab_df():
    total_liab_df = pd.DataFrame(columns=['ts_code', 'total_liab'])
    i = 0
    for c in tqdm(code_list):
        while(True):
            try:
                df = pro.balancesheet(ts_code=c, start_date=START, end_date=END, 
                    fields='ts_code,end_date,total_liab')
                if len(df) != 0:
                    df = df.sort_values('end_date',ascending=False).drop(['end_date'],axis=1).iloc[0]  
                else:
                    print(c, '数据缺失')
                    df = pd.Series([c] + [None], index=total_liab_df.columns.tolist())        
                break
            except:
                continue
        df = pd.DataFrame({i: df.loc[i] for i in total_liab_df.columns.tolist()}, index=[i])
        total_liab_df = pd.concat((total_liab_df, df), axis=0)    
        i += 1  
    
    return total_liab_df


# 近3个月daily数据
def fun_daily_df():
    daily_df = pd.DataFrame(columns=['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
            'pre_close', 'change', 'pct_chg', 'vol', 'amount'])
    for c in tqdm(code_list):
        try:
            df = pro.daily(ts_code=c, start_date=START, end_date=END)
        except:
            df = pd.Series([c] + ['None']*10, index=daily_df.columns.tolist())
        daily_df = pd.concat((daily_df, df), axis=0)    
    
    return daily_df

# 数据合并
def fun_merge():
    cfps_df = fun_cfps_df()
    value_df = fun_value_df()
    total_liab_df = fun_total_liab_df()

    df_transition_0 = pd.DataFrame({'ts_code_all':code_list})
    df_transition_1 = pd.merge(df_transition_0, cfps_df, 'left', left_on='ts_code_all', 
                        right_on='ts_code').drop(['ts_code'],axis=1)
    
    df_transition_2 = pd.merge(df_transition_1, value_df, 'left', left_on='ts_code_all', 
                        right_on='ts_code').drop(['ts_code'],axis=1)
    df_financial = pd.merge(df_transition_2, total_liab_df, 'left', left_on='ts_code_all', 
                        right_on='ts_code').drop(['ts_code'],axis=1)
    
    df_financial = df_financial.rename(columns={'circ_mv': 'free_circ_mv', 
                    'float_share': 'free_share'})

    return df_financial

# 计算流通比率
def fif_transform(df_financial):
    fif = pd.Series(index=df_financial['ts_code_all'])
    free_share_ratio = df_financial['free_share_ratio']
    slice_ = [0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    idx_0 = np.where(free_share_ratio <= 0.15)[0]
    fif.iloc[idx_0] = 0.15
    for i in range((len(slice_)-1)):
        idx = np.where((free_share_ratio <= slice_[i+1]) & (free_share_ratio > slice_[i]))[0]
        fif.iloc[idx] = slice_[i+1]
    idx_1 = np.where(free_share_ratio > 0.8)[0]
    fif.iloc[idx_1] = 1.0
    df_financial = df_financial.set_index('ts_code_all')
    df_financial['fif'] = fif
    df_financial = df_financial.reset_index().drop(['free_share_ratio'], axis=1)

    return df_financial



if __name__ == "__main__":
    # 获取当前日期
    Day_now = datetime.datetime.now().strftime('%Y%m%d')

    ''' 股票候选池参数 '''
    # get_code_list  if_GEM  if_allstock
    # hs300成分股字典 {code: name}
    # 是否重新获取股票候选池
    get_code_list = input('是否重新获取hs300成分股(股票候选池, Y/N): ')
    if get_code_list == 'Y':
        hs300_com = ts.get_hs300s().set_index('name')
        f = lambda x: x+'.SH' if x[0] == '6' else x+'.SZ'
        hs300_com_dict = {f(hs300_com['code'].loc[name]): name  for name in hs300_com.index.tolist()}
        pickle.dump(hs300_com_dict, open(save_path + 'hs300_com_dict.pkl', 'wb'))
    else:
        hs300_com_dict = pickle.load(open(save_path + 'hs300_com_dict.pkl', 'rb'))

    if_allstock = input('是否将所有股票纳入股票候选池(Y/N): ')
    if if_allstock == 'Y':
        # 无st股，无当年新上市的股票
        while(1):
            try:
                y = input('选择某一年（2009-2018）提出st股和新上市股票的股票池: ')
                Stock_candidate_pool = load_dict(str(save_path+'Stock_candidate_pool_noNew'))
                code_list = Stock_candidate_pool[y]
                break
            except:
                continue

    # 是否选择创业板股票
    if_GEM = input('是否将创业板股票纳入股票候选池(Y/N): ')
    if if_GEM == 'N':
        code_list = [code for code in code_list if code[0] != '3']
    else:
        code_list = hs300_com_dict.keys()
    
    # 获取的数据日期范围（近3个月数据） 一个月算22个交易日 3个月 = 3*22 = 66 个交易日
    START, END = str(int(Day_now[:4])-1)+Day_now[4:], Day_now
    df_cal = pro.trade_cal(start_date=START, end_date=END)
    df_cal = df_cal.query('(exchange=="SSE") & (is_open==1)')
    DATE = df_cal.cal_date[-66:]

    df_financial = fun_merge()
    df_financial = fif_transform(df_financial)
    # df_financial = df_financial.rename(columns={'circ_mv':'free_circ_mv'})
    # df_financial['free_circ_mv'] = np.log(df_financial['free_circ_mv'])
    df_financial.to_csv(save_path + 'df_financial.csv')

