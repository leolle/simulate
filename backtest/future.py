# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from lib.gftTools import gsConst, gftIO, gsUtils


"""
1) 策略初始化函数

可以通过给 futures_account 添加新的属性的方法，自定义各种指标变量等等；

策略回测、模拟交易中的账户，策略初始化之前，会建立一个交易账户futures_account，在这个账户会存储上述全局变量参数信息，并在整个策略 执行期间更新并维护可用现金、期货头寸、每日交易订单委托明细等。futures_account会在策略整个回测期间存续。

2). 循环futures_account.universe里面的连续合约的交易日历.

如果有交易信号发生：

如果没有相关合约头寸：

建立头寸

如果有相关合约头寸：

关闭头寸

如果连续合约背后的具体合约发生变化：

如果有头寸：

移仓
"""

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
if not handler:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug('start')

# process input continious data files 
#连续合约价格
#合约序列标志
#具体合约价格
#目标连续合约名称

import abupy
