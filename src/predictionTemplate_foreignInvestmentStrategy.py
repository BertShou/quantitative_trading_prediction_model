import datetime

import talib
from jqdata import *
from kuanke.wizard import *

'''
这是一个量化交易策略模板

代码来源：http://study-quant.s3-website-ap-southeast-2.amazonaws.com/3-1/，需要在聚宽量化投资平台上运行。
因为脱离了平台之后，没有了数据集，所以无法在本地直接运行。

代码详解：http://study-quant.s3-website-ap-southeast-2.amazonaws.com/3-1/, http://study-quant.s3-website-ap-southeast-2.amazonaws.com/3-4/

'''

## 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 设定滑点
    set_slippage(FixedSlippage(0.01))
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True)
    # 设定成交量比例
    set_option('order_volume_ratio', 1)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    # 容器初始化
    check_container_initialize()
    # 动态仓位、频率、计数初始化函数
    check_dynamic_initialize()
    # 股票筛选初始化函数
    check_stocks_initialize()
    # 出场初始化函数
    sell_initialize()
    # 入场初始化函数
    buy_initialize()

    # 关闭提示
    log.set_level('order', 'info')

    # 运行函数
    run_daily(check_stocks, '9:15')  # 选股
    run_daily(main_stock_pick, '9:16')  # 买入卖出列表
    run_daily(sell_every_day, 'open')  # 卖出未卖出成功的股票
    run_daily(trade, 'open')  # 交易
    run_daily(selled_security_list_count, 'after_close')  # 卖出股票日期计数


#######################！！！新手需要使用的地方！！！###################################################
##动态仓位、频率、计数初始化函数(持仓比例，选股频率，买入频率，卖出频率在这里设置)
def check_dynamic_initialize():
    # 个股最大持仓比重
    g.security_max_proportion = 1
    # 选股和买卖频率
    g.check_stocks_refresh_rate = 30
    # 最大建仓数量
    g.max_hold_stocknum = 1

    # 下面这几项不用管
    # 买入频率
    g.buy_refresh_rate = 1
    # 卖出频率
    g.sell_refresh_rate = 1

    # 选股频率计数器
    g.check_stocks_days = 0
    # 机器学习选股频率计数器
    g.days = 0
    # 买卖交易频率计数器
    g.buy_trade_days = 0
    g.sell_trade_days = 0


## 股票池初筛设置函数(股票初筛在这里设置)
def check_stocks_initialize():
    # 是否过滤停盘
    g.filter_paused = True
    # 是否过滤退市
    g.filter_delisted = True
    # 是否只有ST
    g.only_st = False
    # 是否过滤ST
    g.filter_st = True
    # 股票池(填指数)
    g.security_universe_index = [
        'all_a_securities']  # 这里填写指数，全部股票就填['all_a_securities']，沪深300股票就填['000300.XSHG'],中证500就填['000905.XSHG'],沪深300+中证500就填['000300.XSHG','000905.XSHG']
    # 填成分股(填成分股)
    g.security_universe_user_securities = []
    # 行业列表
    g.industry_list = ["801010", "801020", "801030", "801040", "801050", "801080", "801110", "801120", "801130", "801140",
                       "801150", "801160", "801170", "801180", "801200", "801210", "801230", "801710", "801720", "801730",
                       "801740", "801750", "801760", "801770", "801780", "801790", "801880", "801890"]
    # 概念列表
    g.concept_list = []
    # 黑名单
    g.blacklist = ['300268.XSHE', '600035.XSHG', '300028.XSHE']


## 买入股票，卖出股票筛选函数
def main_stock_pick(context):
    if g.days % g.check_stocks_refresh_rate != 0:
        g.days += 1
        return
    g.sell_stock_list = []
    g.buy_stock_list = []

    ####自定义编辑范围#####

    # 外资策略

    date = context.current_dt.strftime("%Y-%m-%d")
    today = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    yesterday = shifttradingday(today, shift=-1)
    print('前一个交易日:', yesterday)

    q = query(finance.STK_EL_TOP_ACTIVATE).filter(finance.STK_EL_TOP_ACTIVATE.day == yesterday,
                                                  finance.STK_EL_TOP_ACTIVATE.code.in_(g.check_out_lists))
    df = finance.run_query(q)

    df['net'] = df.buy - df.sell
    df = df.sort_values('net', ascending=False)
    df = df[(df.link_id != 310003) & (df.link_id != 310004)]
    a = 0
    while df.empty == True:
        a = a + 1
        yesterday = shifttradingday(today, shift=-a)
        print('前一个交易日:', yesterday)
        q = query(finance.STK_EL_TOP_ACTIVATE).filter(finance.STK_EL_TOP_ACTIVATE.day == yesterday)
        df = finance.run_query(q)
        df['net'] = df.buy - df.sell
        df = df.sort_values('net', ascending=False)
        df = df[(df.link_id != 310003) & (df.link_id != 310004)]

    stockset = list(df['code'])

    g.sell_stock_list1 = list(context.portfolio.positions.keys())

    current_data = get_current_data()

    paused_list = [stock for stock in g.sell_stock_list1 if current_data[stock].paused]

    for stock in g.sell_stock_list1:
        if stock in paused_list:
            continue
        elif stock not in stockset[:g.max_hold_stocknum]:
            g.sell_stock_list.append(stock)

    for stock in stockset[:g.max_hold_stocknum]:
        if stock in g.sell_stock_list:
            pass
        else:
            g.buy_stock_list.append(stock)

    ####自定义编辑范围#####

    log.info('卖出列表:', g.sell_stock_list)
    log.info('购买列表:', g.buy_stock_list)
    g.days = 1
    return g.sell_stock_list, g.buy_stock_list


#######################！！！新手需要使用的地方！！！###################################################


##容器初始化(有新的全局容器可以加到这里)(新手忽略这里)
def check_container_initialize():
    # 卖出股票列表
    g.sell_stock_list = []
    # 买入股票列表
    g.buy_stock_list = []
    # 获取未卖出的股票
    g.open_sell_securities = []
    # 卖出股票的dict
    g.selled_security_list = {}
    # 涨停股票列表
    g.ZT = []


## 出场初始化函数(新手忽略这里)
def sell_initialize():
    # 设定是否卖出buy_lists中的股票
    g.sell_will_buy = True

    # 固定出仓的数量或者百分比
    g.sell_by_amount = None
    g.sell_by_percent = None


## 入场初始化函数(新手忽略这里)
def buy_initialize():
    # 是否可重复买入
    g.filter_holded = False

    # 委托类型
    g.order_style_str = 'by_cap_mean'
    g.order_style_value = 100


## 股票初筛(新手忽略这里)
def check_stocks(context):
    if g.check_stocks_days % g.check_stocks_refresh_rate != 0:
        # 计数器加一
        g.check_stocks_days += 1
        return
    # 股票池赋值
    g.check_out_lists = get_security_universe(context, g.security_universe_index, g.security_universe_user_securities)
    # 行业过滤
    g.check_out_lists = industry_filter(context, g.check_out_lists, g.industry_list)
    # 概念过滤
    g.check_out_lists = concept_filter(context, g.check_out_lists, g.concept_list)
    # 过滤ST股票
    g.check_out_lists = st_filter(context, g.check_out_lists)
    # 过滤停牌股票
    g.check_out_lists = paused_filter(context, g.check_out_lists)
    # 过滤退市股票
    g.check_out_lists = delisted_filter(context, g.check_out_lists)
    # 过滤黑名单股票
    g.check_out_lists = [s for s in g.check_out_lists if s not in g.blacklist]
    # 计数器归一
    g.check_stocks_days = 1
    return


## 卖出未卖出成功的股票(新手忽略这里)
def sell_every_day(context):
    g.open_sell_securities = list(set(g.open_sell_securities))
    open_sell_securities = [s for s in context.portfolio.positions.keys() if s in g.open_sell_securities]
    if len(open_sell_securities) > 0:
        for stock in open_sell_securities:
            order_target_value(stock, 0)
    g.open_sell_securities = [s for s in g.open_sell_securities if s in context.portfolio.positions.keys()]
    return


## 交易函数(新手忽略这里)
def trade(context):
    # 初始化买入列表
    buy_lists = []

    # 买入股票筛选
    if g.buy_trade_days % g.buy_refresh_rate == 0:
        # 获取 buy_lists 列表
        buy_lists = g.buy_stock_list
        # 过滤涨停股票
        buy_lists = high_limit_filter(context, buy_lists)
        log.info('购买列表最终', buy_lists)

    # 卖出操作
    if g.sell_trade_days % g.sell_refresh_rate != 0:
        # 计数器加一
        g.sell_trade_days += 1
    else:
        # 卖出股票
        sell(context, buy_lists)
        # 计数器归一
        g.sell_trade_days = 1

    # 买入操作
    if g.buy_trade_days % g.buy_refresh_rate != 0:
        # 计数器加一
        g.buy_trade_days += 1
    else:
        # 卖出股票
        buy(context, buy_lists)
        # 计数器归一
        g.buy_trade_days = 1


##################################  交易函数群 ##################################(新手忽略)

# 交易函数 - 出场
def sell(context, buy_lists):
    # 获取 sell_lists 列表
    init_sl = context.portfolio.positions.keys()
    sell_lists = context.portfolio.positions.keys()

    # 判断是否卖出buy_lists中的股票
    if not g.sell_will_buy:
        sell_lists = [security for security in sell_lists if security not in buy_lists]

    ### _出场函数筛选-开始 ###
    sell_lists = g.sell_stock_list
    ### _出场函数筛选-结束 ###

    # 卖出股票
    if len(sell_lists) > 0:
        for stock in sell_lists:
            sell_by_amount_or_percent_or_none(context, stock, g.sell_by_amount, g.sell_by_percent, g.open_sell_securities)

    # 获取卖出的股票, 并加入到 g.selled_security_list中
    selled_security_list_dict(context, init_sl)

    return


# 交易函数 - 入场
def buy(context, buy_lists):
    # 判断是否可重复买入
    buy_lists = holded_filter(context, buy_lists)

    # 获取最终的 buy_lists 列表
    Num = g.max_hold_stocknum - len(context.portfolio.positions)
    buy_lists = buy_lists[:Num]

    # 买入股票
    if len(buy_lists) > 0:
        # 分配资金
        for stock in buy_lists:
            position_count = len(context.portfolio.positions)
            if g.max_hold_stocknum > position_count:
                value = context.portfolio.cash / (g.max_hold_stocknum - position_count)
                if context.portfolio.positions[stock].total_amount == 0:
                    order_target_value(stock, value)
    return


###################################  公用函数群 ##################################(新手忽略)


## 过滤同一标的继上次卖出N天不再买入
def filter_n_tradeday_not_buy(security, n=0):
    try:
        if (security in g.selled_security_list.keys()) and (g.selled_security_list[security] < n):
            return False
        return True
    except:
        return True


## 是否可重复买入
def holded_filter(context, security_list):
    if not g.filter_holded:
        security_list = [stock for stock in security_list if stock not in context.portfolio.positions.keys()]
    # 返回结果
    return security_list


## 卖出股票加入dict
def selled_security_list_dict(context, security_list):
    selled_sl = [s for s in security_list if s not in context.portfolio.positions.keys()]
    if len(selled_sl) > 0:
        for stock in selled_sl:
            g.selled_security_list[stock] = 0


## 过滤停牌股票
def paused_filter(context, security_list):
    if g.filter_paused:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if not current_data[stock].paused]
    # 返回结果
    return security_list


## 过滤退市股票
def delisted_filter(context, security_list):
    if g.filter_delisted:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if
                         not (('退' in current_data[stock].name) or ('*' in current_data[stock].name))]
    # 返回结果
    return security_list


## 过滤ST股票
def st_filter(context, security_list):
    if g.only_st:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if current_data[stock].is_st]
    else:
        if g.filter_st:
            current_data = get_current_data()
            security_list = [stock for stock in security_list if not current_data[stock].is_st]
    # 返回结果
    return security_list


# 过滤涨停股票
def high_limit_filter(context, security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not (current_data[stock].last_price >= current_data[stock].high_limit)]
    # 返回结果
    return security_list


# 获取股票股票池
def get_security_universe(context, security_universe_index, security_universe_user_securities):
    temp_index = []
    for s in security_universe_index:
        if s == 'all_a_securities':
            temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
        else:
            temp_index += get_index_stocks(s)
    for x in security_universe_user_securities:
        temp_index += x
    return sorted(list(set(temp_index)))


# 行业过滤
def industry_filter(context, security_list, industry_list):
    if len(industry_list) == 0:
        # 返回股票列表
        return security_list
    else:
        securities = []
        for s in industry_list:
            temp_securities = get_industry_stocks(s)
            securities += temp_securities
        security_list = [stock for stock in security_list if stock in securities]
        # 返回股票列表
        return security_list


# 概念过滤
def concept_filter(context, security_list, concept_list):
    if len(concept_list) == 0:
        return security_list
    else:
        securities = []
        for s in concept_list:
            temp_securities = get_concept_stocks(s)
            securities += temp_securities
        security_list = [stock for stock in security_list if stock in securities]
        # 返回股票列表
        return security_list


## 卖出股票日期计数
def selled_security_list_count(context):
    # g.daily_risk_management = True
    if len(g.selled_security_list) > 0:
        for stock in g.selled_security_list.keys():
            g.selled_security_list[stock] += 1


# 获取交易日
def shifttradingday(date, shift):
    # 获取N天前的交易日日期
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) + shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]
