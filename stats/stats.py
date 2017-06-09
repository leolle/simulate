# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from lib.gftTools import gftIO
from lib.gftTools import gsConst


def cal_max_dd(df_single_return):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    df_single_return :
        Daily returns of the strategy, noncumulative.

    Returns
    ----------
    float
        Maximum drawdown.
    """
    if len(df_single_return) < 1:
        return np.nan

    df_perform_equity_curve = (1. + df_single_return).cumprod()
    df_perform_cum_max = df_perform_equity_curve.cummax()
    # drawdown series
    df_perform_drawdown = df_perform_equity_curve / df_perform_cum_max - 1
    max_dd = df_perform_drawdown.min()
    val = max_dd.values[0].astype(np.float)
    return val


def cum_returns(df_single_return):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    df_single_return : np.ndarray
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    float
        Series of cumulative returns, starting value from 0.

    """

    if len(df_single_return) < 1:
        return type(df_single_return)([])

    if np.any(np.isnan(df_single_return)):
        df_single_return = df_single_return.copy()
        df_single_return[np.isnan(df_single_return)] = 0.

    df_cum = (df_single_return + 1).cumprod(axis=0) - 1

    cum_val = np.array(df_cum)
    #print(type(cum_val))

    return cum_val[-1][-1]


def aggregate_returns(df_single_return, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    df_single_return : pd.DataFrame
        Daily returns of the strategy, noncumulative.
    convert_to : int
        Can be '1 day', '1 month', '3 months', or '3 months', '6 months', 
        '1 year', '3 years'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == 7:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == 21:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == 252:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be in format(days, months, years)'
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def annual_return(df_single_return, period=gsConst.Const.DAILY):
    """Determines the mean annual growth rate of returns.

    Parameters
    ----------
    df_single_return : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    float
    """

    if len(df_single_return) < 1:
        return np.nan

    num_years = float(len(df_single_return)) / period

    # Pass array to ensure index -1 looks up successfully.
    cum_ret = cum_returns(np.asanyarray(df_single_return))
    f_annual_return = (1. + cum_ret) ** (1. / num_years) - 1

    return f_annual_return


def sharpe_ratio(df_single_returns, f_risk_free_rate):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
    f_risk_free_rate : int, float
        Constant risk-free return throughout the period.

    Returns
    -------
    float
        Sharpe ratio.

        np.nan
            If insufficient length of returns or if if adjusted returns are 0.

    """

    if len(df_single_returns) < 2:
        return np.nan

    annual_ret = annual_return(df_single_returns)
    annual_vol = annual_volatility(df_single_returns)
    return (annual_ret - f_risk_free_rate) / annual_vol


def sortino_ratio(df_single_returns, required_return=0,
                  _downside_risk=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    float
        Annualized Sortino ratio.

    """

    if len(df_single_returns) < 2:
        return np.nan

    f_mu = annual_return(df_single_returns)

    dsr = (_downside_risk if _downside_risk is not None
           else annual_downside_risk(df_single_returns))
    sortino = (f_mu - required_return) / dsr

    return sortino


def annual_downside_risk(df_single_returns, required_return=0, period=gsConst.Const.DAILY):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    float, pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        Annualized downside deviation

    """

    if len(df_single_returns) < 1:
        return np.nan

    downside_diff = (df_single_returns - df_single_returns.mean()).copy()
    mask = downside_diff > 0
    downside_diff[mask] = 0.0

    squares = np.square(downside_diff)
    mean_squares = np.mean(squares)

    dside_risk = np.sqrt(mean_squares) * np.sqrt(period)

    return dside_risk.values[0].astype(np.float)


def downside_std(df_single_returns):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    float, pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        downside deviation

    """

    if len(df_single_returns) < 1:
        return np.nan

    downside_diff = (df_single_returns - df_single_returns.mean()).copy()
    mask = downside_diff > 0
    downside_diff[mask] = 0.0

    squares = np.square(downside_diff)
    mean_squares = np.mean(squares)
    downside_std = np.sqrt(mean_squares)

    return downside_std.values.astype(np.float)


def int_trading_days(df_single_returns):
    """
    Determines the number of trading days for a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    int
       Trading days.

    """

    if len(df_single_returns) < 1:
        return np.nan

    trading_days = len(df_single_returns.index)

    return trading_days


def annual_volatility(df_single_returns, period=gsConst.Const.DAILY):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.

    Returns
    -------
    float, np.ndarray
        Annual volatility.
    """

    if len(df_single_returns) < 2:
        return np.nan

    std = df_single_returns.std(ddof=1)

    volatility = std * (period ** (1.0 / 2))

    return volatility.values[0].astype(np.float)


def return_std(df_single_returns):
    """
    Determines the standard deviation of returns for a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.

    Returns
    -------
    float, np.ndarray
        standard deviation of returns.
    """

    if len(df_single_returns) < 2:
        return np.nan

    std = df_single_returns.std(ddof=1)

    return std.values[0].astype(np.float)


def max_holding_num(df_holding):
    """
    Determines the maximum asset holding number for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.

    Returns
    -------
    int
        Maximum assets holding number.
    """

    if len(df_holding) < 1:
        return np.nan

    max_holding_number = df_holding.count(axis=1).max()

    return max_holding_number


def min_holding_num(df_holding):
    """
    Determines the minimum asset holding number for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.

    Returns
    -------
    int
        Minimum assets holding number.
    """

    if len(df_holding) < 1:
        return np.nan

    max_holding_number = df_holding.count(axis=1).min()

    return min_holding_number


def average_holding_num(df_holding):
    """
    Determines the average asset holding number for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.

    Returns
    -------
    float
        Average assets holding number.
    """

    if len(df_holding) < 1:
        return np.nan

    average_holding_number = df_holding.count(axis=1).mean()

    return average_holding_number


def latest_holding_num(df_holding):
    """
    Determines the latest asset holding number for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.

    Returns
    -------
    int
        Latest assets holding number.
    """

    if len(df_holding) < 1:
        return np.nan

    latest_holding_number = df_holding.count(axis=1).ix[-1]

    return latest_holding_number


def average_holding_num_percentage(df_holding, df_universe):
    """
    Determines the average asset holding percentage for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    df_universe : pd.DataFrame or np.ndarray
        Historical universe of the strategy.

    Returns
    -------
    float
        Mean assets holding number percentage.
    """

    if len(df_holding) < 1:
        return np.nan

    avg_holding_num_pct = (df_holding.count(axis=1) / df_universe.count(axis=1)).mean()

    return avg_holding_num_pct


def latest_holding_num_percentage(df_holding, df_universe):
    """
    Determines the latest asset holding percentage for a strategy.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    df_universe : pd.DataFrame or np.ndarray
        Historical universe of the strategy.

    Returns
    -------
    float
        Latest assets holding number percentage.
    """

    if len(df_holding) < 1:
        return np.nan

    latest_holding_num_pct = (df_holding.count(axis=1) / df_universe.count(axis=1)).ix[-1]

    return latest_holding_num_pct


def PNLFitness(df_single_period_return, f_risk_free_rate, periods, benchmark_ret, holding, closing_price, market_capital):
    """
    calculate pnl fitness for a strategy.

    Parameters
    ----------
    df_single_returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.

    Returns
    -------
    result, dictionary
        fitness of returns.
    """
    df_single_period_return = df_single_period_return.asMatrix()
    result = {}
    result[gsConst.Const.AnnualReturn] = annual_return(df_single_period_return, period=periods)
    result[gsConst.Const.AnnualVolatility] = annual_volatility(df_single_period_return, period=periods)
    result[gsConst.Const.AnnualDownVolatility] = annual_downside_risk(df_single_period_return, period=periods)
    result[gsConst.Const.CumulativeReturn] = cum_returns(df_single_period_return)
    result[gsConst.Const.DownStdReturn] = downside_std(df_single_period_return)
    result[gsConst.Const.StartDate] = df_single_period_return.index[0]
    result[gsConst.Const.EndDate] = df_single_period_return.index[-1]
    result[gsConst.Const.MaxDrawdownRate] = cal_max_dd(df_single_period_return)
    result[gsConst.Const.StdReturn] = return_std(df_single_period_return)
    result[gsConst.Const.SharpeRatio] = sharpe_ratio(df_single_period_return, f_risk_free_rate)
    result[gsConst.Const.SortinoRatio] = sortino_ratio(df_single_period_return,f_risk_free_rate)
    result[gsConst.Const.TotalTradingDays] = int_trading_days(df_single_period_return)
    if len(benchmark_ret)>1:
        result[gsConst.Const.BenchmarAnnualReturn] = annual_return(benchmark_ret)
        result[gsConst.Const.Benchmar]
        
    return result

# if __name__ == '__main__':
#     data = [100, 101, 100, 101, 102, 103, 104, 105, 106, 107]
#     dates = pd.date_range('1/1/2000', periods=10)
#     df_price = pd.DataFrame(data, index=dates, columns=['price'])
#     df_single_period_return = df_price / df_price.shift(1) - 1
#     f_risk_free_rate = 0.015
#     result = PNLFitness(df_single_period_return, f_risk_free_rate)

result = {}
import pickle

path = '~/projects/simulate/data/stats/'
x0 = pickle.load(path + 'x4.pkl')

# result['annualized_return'] = annual_return(df_single_period_return)
# result['annualized_volatility'] = annual_volatility(df_single_period_return)
# result['annualized_downrisk_vol'] = annual_downside_risk(
#     df_single_period_return)
# result['cumlative_return'] = cum_returns(df_single_period_return)
# result['downside_std'] = downside_std(df_single_period_return)
# result['start_date'] = df_single_period_return.index[0]
# result['end_date'] = df_single_period_return.index[-1]
# result['max_dd'] = cal_max_dd(df_single_period_return)
# result['return_std'] = return_std(df_single_period_return)
# result['sharpe_ratio'] = sharpe_ratio(df_single_period_return,
#                                       f_risk_free_rate)
# result['sornito_ratio'] = sortino_ratio(df_single_period_return,
#                                         f_risk_free_rate)
# result['trading_days'] = int_trading_days(df_single_period_return)
# print (result)

# if __name__ == '__main__':
#     path = '../data/'
#     df_single_period_return = pd.read_csv(path + 'single_return.csv', index_col=0)
#     df_single_period_return.index = pd.to_datetime(df_single_period_return.index)
#     df_single_period_return = df_single_period_return.ix['2015-06-02':]
#     f_risk_free_rate = 0.015
#     result = U_PNL_FITNESS(df_single_period_return, f_risk_free_rate)

#     print(result)
