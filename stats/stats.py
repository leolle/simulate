# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
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


#    return cum_val[-1][-1]
    if df_cum.shape[1] == 1:
        return df_cum.ix[-1].values[-1].astype(np.float)
    else:
        return df_cum.iloc[-1,:].values.astype(np.float)

def annual_return(df_single_return, period):
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
    cum_ret = cum_returns(df_single_return)
    f_annual_return = (1. + cum_ret) ** (1. / num_years) - 1

    return f_annual_return


def sharpe_ratio(df_single_returns, f_risk_free_rate, period):
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

    annual_ret = annual_return(df_single_returns, period)
    annual_vol = annual_volatility(df_single_returns, period)
    return (annual_ret - f_risk_free_rate) / annual_vol


def sortino_ratio(df_single_returns, required_return, period):
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

    f_mu = annual_return(df_single_returns, period)

    dsr = annual_downside_risk(df_single_returns, period)
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

    return downside_std.values[-1].astype(np.float)


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


def annual_volatility(df_single_returns, period):
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

    min_holding_number = df_holding.count(axis=1).min()

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

    latest_holding_number = df_holding.count(axis=1).iloc[-1]

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

    df_count = df_universe.loc[df_holding.index,:].count(axis=1)
    df_count = df_count[df_count > 0]
    avg_holding_num_pct = (df_holding.count(axis=1) / df_count).mean()

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


    latest_holding_num_pct = (df_holding.count(axis=1).ix[-1] / df_universe.ix[df_holding.index].count(axis=1)).ix[-1]

    return latest_holding_num_pct


def excess_annual_return(df_single_return, benchmark_ret, period):
    """
    Determines the excess annual return for a strategy vs benchmark.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    benchmark_ret : pd.DataFrame or np.ndarray
        Benchmark returns.

    Returns
    -------
    float
        excess return.
    """

    if len(df_single_return) < 1:
        return np.nan

    ret_diff = (df_single_return - benchmark_ret).dropna()
    if len(ret_diff) < 1:
        raise ValueError("check length of single return and benchmark return")
    ex_ret = annual_return(ret_diff, period)

    return ex_ret


def excess_single_period_return(df_single_return, benchmark_ret):
    """
    Determines the excess single period return for a strategy vs benchmark.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    benchmark_ret : pd.DataFrame or np.ndarray
        Benchmark returns.

    Returns
    -------
    float
        excess return.
    """

    if len(df_single_return) < 1:
        return np.nan

    ret_diff = (df_single_return - benchmark_ret).dropna()
    if len(ret_diff) < 1:
        raise ValueError("check length of single return and benchmark return")

    return ret_diff


def excess_cumulative_return(df_single_return, benchmark_ret):
    """
    Determines the excess cumulative returns for a strategy vs benchmark.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    benchmark_ret : pd.DataFrame or np.ndarray
        Benchmark returns.

    Returns
    -------
    float
        excess return.
    """

    if len(df_single_return) < 1:
        return np.nan

    ret_diff = (df_single_return - benchmark_ret).dropna()
    if len(ret_diff) < 1:
        raise ValueError("check length of single return and benchmark return")
    df_cum = (ret_diff + 1).cumprod(axis=0) - 1

    return df_cum


def aggregate_returns(df_single_return, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    df_single_return : pd.DataFrame
        Daily returns of the strategy, noncumulative.
    convert_to : int
        Can be '1 day', '1 month': 30days, '3 months': 90days, '6 months': 180days,
        '1 year': 365days, '3 years': 1095days.

    Returns
    -------
    float
        Aggregated returns.
    """

    def cumulate_returns(x):
        return cum_returns(x)

    last_day = df_single_return.index[-1]

    return cumulate_returns(df_single_return.loc[(last_day- pd.to_timedelta("%sday"%convert_to)):,:])


def portfolio_market_ratio(df_holding, df_market_price, df_market_capital):
    """
    To calculate the holding value to total market capital value.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    df_market_price: pd.DataFrame
        Historical market close price from 1990.
    df_market_capital:
        Historical market capital from 1990

    Returns
    -------
    float
        Portfolio Value Market Capital Ratio.
    """
    if len(df_market_price) < 1 or len(df_market_price) < 1:
        return np.nan

    date_range = df_holding.index

    df_holding_value = (df_holding * df_market_price.loc[date_range,:])
    df_weight = df_holding_value.divide(df_holding_value.sum(axis=1), axis=0)
    df_portfolio_market_ratio = (df_weight * df_market_capital.loc[date_range,:]).\
                                sum(axis=1)/df_market_capital.loc[date_range,:].sum(axis=1)

    return df_portfolio_market_ratio.mean()


def holding_dispersion_std(df_holding, df_market_price, period=gsConst.Const.DAILY):
    """
    To calculate the holding annual return standard deviation.

    Parameters
    ----------
    df_holding : pd.DataFrame or np.ndarray
        Historical holding of the strategy.
    df_market_price: pd.DataFrame
        Historical market close price from 1990.

    Returns
    -------
    float
        standard deviation.
    """
    if len(df_market_price) < 1:
        return np.nan

    date_range = df_holding.index

    df_holding_value = (df_holding * df_market_price.loc[date_range,:])
    df_holding_value = df_holding_value.fillna(method='ffill')
    df_holding_ret = df_holding_value / df_holding_value.shift(1) - 1
    df_holding_ret = df_holding_ret

    if np.any(np.isnan(df_holding_ret)):
        df_single_return = df_holding_ret.copy()
        df_single_return[np.isnan(df_single_return)] = 0.
    cum_ret = (df_single_return + 1).cumprod(axis=0) - 1
    num_years = float(len(df_holding_ret)) / period
    df_annual_return = (1. + cum_ret) ** (1. / num_years) - 1
    df_annual_return_std = df_annual_return.std().mean()

    return df_annual_return_std


def PNLFitness(df_single_period_return, f_risk_free_rate, benchmark_ret, holding, closing_price, market_capital):
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
    try:
        df_single_period_return = df_single_period_return.asMatrix()
    except AttributeError:
        df_single_period_return = None
    try:
        benchmark_ret = benchmark_ret.asMatrix()
    except AttributeError:
        benchmark_ret = None
    try:
        holding = holding.asMatrix()
    except AttributeError:
        holding = None
    try:
        closing_price = closing_price.asMatrix()
    except AttributeError:
        closing_price = None
    try:
        market_capital = market_capital.asMatrix()
    except AttributeError:
        market_capital = None

    dt_diff = df_single_period_return.index.to_series().diff().mean()
    if dt_diff < pd.Timedelta('3 days'):
        periods = gsConst.Const.DAILY
    elif dt_diff > pd.Timedelta('3 days') and dt_diff < pd.Timedelta('10 days'):
        periods = gsConst.Const.WEEKLY
    else:
        periods = gsConst.Const.MONTHLY

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
    result[gsConst.Const.SharpeRatio] = sharpe_ratio(df_single_period_return, f_risk_free_rate, periods)
    result[gsConst.Const.SortinoRatio] = sortino_ratio(df_single_period_return, required_return=f_risk_free_rate, period=periods)
    result[gsConst.Const.TotalTradingDays] = int_trading_days(df_single_period_return)
    result[gsConst.Const.MaxHoldingNum] = max_holding_num(holding)
    result[gsConst.Const.MinHoldingNum] = min_holding_num(holding)
    result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)
    result[gsConst.Const.LatestHoldingNum] = latest_holding_num(holding)
    result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)
    if closing_price is not None:
        result[gsConst.Const.AverageHoldingNumPercentage] = average_holding_num_percentage(holding, closing_price)
        result[gsConst.Const.LatestHoldingNumPercentage] = latest_holding_num_percentage(holding, closing_price)
        result[gsConst.Const.PortfolioValueMarketCapitalRatio] = portfolio_market_ratio(holding, closing_price, market_capital)
        result[gsConst.Const.AnnualReturnDispersionAverage] = holding_dispersion_std(holding, closing_price, period=periods)
    result[gsConst.Const.AggregateReturnOneDay] = aggregate_returns(df_single_period_return, 0)
    result[gsConst.Const.AggregateReturnOneMonth] = aggregate_returns(df_single_period_return, 30)
    result[gsConst.Const.AggregateReturnThreeMonth] = aggregate_returns(df_single_period_return, 90)
    result[gsConst.Const.AggregateReturnSixMonth] = aggregate_returns(df_single_period_return, 180)
    result[gsConst.Const.AggregateReturnOneYear] = aggregate_returns(df_single_period_return, 365)
    result[gsConst.Const.AggregateReturnThreeYear] = aggregate_returns(df_single_period_return, 1095)

    if len(benchmark_ret) > 1:
        result[gsConst.Const.BenchmarkAnnualReturn] = annual_return(benchmark_ret, period=periods)
        result[gsConst.Const.BenchmarkSharpeRatio] = sharpe_ratio(benchmark_ret, f_risk_free_rate, periods)
        result[gsConst.Const.BenchmarkAnnualVolatility] = annual_volatility(benchmark_ret, period=periods)
        result[gsConst.Const.BenchmarStdReturn] = return_std(benchmark_ret)
        result[gsConst.Const.BenchmarkMaxDrawdownRate] = cal_max_dd(benchmark_ret)
        result[gsConst.Const.BenchmarkCumulativeReturn] = cum_returns(benchmark_ret)
        result[gsConst.Const.ExcessAnnualReturn] = excess_annual_return(df_single_period_return, benchmark_ret, period=periods)
 
    return result


path = r'/home/weiwu/projects/simulate/data/stats/'
df_single_period_return = gftIO.zload(path + 'df_single_period_return.pkl')
f_risk_free_rate = gftIO.zload(path + 'f_risk_free_rate.pkl')
benchmark_ret = gftIO.zload(path + 'benchmark_ret.pkl')
holding = gftIO.zload(path + 'holding.pkl')
closing_price = gftIO.zload(path + 'closing_price.pkl')
market_capital = gftIO.zload(path + 'market_capital.pkl')

if isinstance(df_single_period_return, gftIO.GftTable):
    df_single_period_return = df_single_period_return.asMatrix().copy()

if isinstance(benchmark_ret, gftIO.GftTable):
    benchmark_ret = benchmark_ret.asMatrix().copy()

if isinstance(holding, gftIO.GftTable):
    holding = holding.asMatrix().copy()

if isinstance(closing_price, gftIO.GftTable):
    closing_price = closing_price.asMatrix().copy()

if isinstance(market_capital, gftIO.GftTable):
    market_capital = market_capital.asMatrix().copy()

dt_diff = df_single_period_return.index.to_series().diff().mean()
if dt_diff < pd.Timedelta('3 days'):
    periods = gsConst.Const.DAILY
elif dt_diff > pd.Timedelta('3 days') and dt_diff < pd.Timedelta('10 days'):
    periods = gsConst.Const.WEEKLY
else:
    periods = gsConst.Const.MONTHLY


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
result[gsConst.Const.SharpeRatio] = sharpe_ratio(df_single_period_return, f_risk_free_rate, periods)
result[gsConst.Const.SortinoRatio] = sortino_ratio(df_single_period_return, required_return=f_risk_free_rate, period=periods)
result[gsConst.Const.TotalTradingDays] = int_trading_days(df_single_period_return)
result[gsConst.Const.MaxHoldingNum] = max_holding_num(holding)
result[gsConst.Const.MinHoldingNum] = min_holding_num(holding)
result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)
result[gsConst.Const.LatestHoldingNum] = latest_holding_num(holding)
result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)
if closing_price is not None:
    result[gsConst.Const.AverageHoldingNumPercentage] = average_holding_num_percentage(holding, closing_price)
    result[gsConst.Const.LatestHoldingNumPercentage] = latest_holding_num_percentage(holding, closing_price)
    result[gsConst.Const.PortfolioValueMarketCapitalRatio] = portfolio_market_ratio(holding, closing_price, market_capital)
    result[gsConst.Const.AnnualReturnDispersionAverage] = holding_dispersion_std(holding, closing_price, period=periods)
result[gsConst.Const.AggregateReturnOneDay] = aggregate_returns(df_single_period_return, 0)
result[gsConst.Const.AggregateReturnOneMonth] = aggregate_returns(df_single_period_return, 30)
result[gsConst.Const.AggregateReturnThreeMonth] = aggregate_returns(df_single_period_return, 90)
result[gsConst.Const.AggregateReturnSixMonth] = aggregate_returns(df_single_period_return, 180)
result[gsConst.Const.AggregateReturnOneYear] = aggregate_returns(df_single_period_return, 365)
result[gsConst.Const.AggregateReturnThreeYear] = aggregate_returns(df_single_period_return, 1095)

if len(benchmark_ret) > 1:
    result[gsConst.Const.BenchmarkAnnualReturn] = annual_return(benchmark_ret, period=periods)
    result[gsConst.Const.BenchmarkSharpeRatio] = sharpe_ratio(benchmark_ret, f_risk_free_rate, periods)
    result[gsConst.Const.BenchmarkAnnualVolatility] = annual_volatility(benchmark_ret, period=periods)
    result[gsConst.Const.BenchmarStdReturn] = return_std(benchmark_ret)
    result[gsConst.Const.BenchmarkMaxDrawdownRate] = cal_max_dd(benchmark_ret)
    result[gsConst.Const.BenchmarkCumulativeReturn] = cum_returns(benchmark_ret)
    result[gsConst.Const.ExcessSinglePeriodReturns] = excess_single_period_return(df_single_period_return, benchmark_ret)
    result[gsConst.Const.ExcessCumulativeReturns] = excess_cumulative_return(df_single_period_return, benchmark_ret)
    result[gsConst.Const.ExcessAnnualReturn] = excess_annual_return(df_single_period_return, benchmark_ret, period=periods)

print(result)

# if __name__ == '__main__':
#     path = '../data/'
#     df_single_period_return = pd.read_csv(path + 'single_return.csv', index_col=0)
#     df_single_period_return.index = pd.to_datetime(df_single_period_return.index)
#     df_single_period_return = df_single_period_return.ix['2015-06-02':]
#     f_risk_free_rate = 0.015
#     result = U_PNL_FITNESS(df_single_period_return, f_risk_free_rate)

#     print(result)


def PnLFitness(df_single_period_return,f_risk_free_rate,df_benchmark_ret):
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
    if isinstance(df_single_period_return, gftIO.GftTable):
        df_single_period_return = df_single_period_return.asMatrix()
    if isinstance(df_benchmark_ret, gftIO.GftTable):
        df_benchmark_ret = df_benchmark_ret.asMatrix()

    # to check the frequency of the strategy, DAILY or MONTHLY
    dt_diff = df_single_period_return.index.to_series().diff().mean()
    if dt_diff < pd.Timedelta('3 days'):
        periods = gsConst.Const.DAILY
    elif dt_diff > pd.Timedelta('3 days') and dt_diff < pd.Timedelta('10 days'):
        periods = gsConst.Const.WEEKLY
    else:
        periods = gsConst.Const.MONTHLY

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
    result[gsConst.Const.SharpeRatio] = sharpe_ratio(df_single_period_return, f_risk_free_rate, periods)
    result[gsConst.Const.SortinoRatio] = sortino_ratio(df_single_period_return, required_return=f_risk_free_rate, period=periods)
    result[gsConst.Const.AggregateReturnOneDay] = aggregate_returns(df_single_period_return, 0)
    result[gsConst.Const.AggregateReturnOneMonth] = aggregate_returns(df_single_period_return, 30)
    result[gsConst.Const.AggregateReturnThreeMonth] = aggregate_returns(df_single_period_return, 90)
    result[gsConst.Const.AggregateReturnSixMonth] = aggregate_returns(df_single_period_return, 180)
    result[gsConst.Const.AggregateReturnOneYear] = aggregate_returns(df_single_period_return, 365)
    result[gsConst.Const.AggregateReturnThreeYear] = aggregate_returns(df_single_period_return, 1095)

    if len(df_benchmark_ret) > 1:
        result[gsConst.Const.BenchmarkAnnualReturn] = annual_return(benchmark_ret, period=periods)
        result[gsConst.Const.BenchmarkSharpeRatio] = sharpe_ratio(benchmark_ret, f_risk_free_rate, periods)
        result[gsConst.Const.BenchmarkAnnualVolatility] = annual_volatility(benchmark_ret, period=periods)
        result[gsConst.Const.BenchmarStdReturn] = return_std(benchmark_ret)
        result[gsConst.Const.BenchmarkMaxDrawdownRate] = cal_max_dd(benchmark_ret)
        result[gsConst.Const.BenchmarkCumulativeReturn] = cum_returns(benchmark_ret)
        result[gsConst.Const.ExcessAnnualReturn] = excess_annual_return(df_single_period_return, benchmark_ret, period=periods)

    return result


def HoldingFitness(holding,closing_price,market_capital):
    """
    calculate holding fitness for a strategy.

    Parameters
    ----------
    holding : pd.DataFrame or np.ndarray
        holding position of a strategy.
    closing_price: pd.DataFrame
        closing price of stocks in the portfolio.
    market_capital: pd.DataFrame
        market capital of stocks in the portfolio.

    Returns
    -------
    result, dictionary
        fitness of returns.
    """
    if isinstance(holding, gftIO.GftTable):
        holding = holding.asMatrix()
    if isinstance(closing_price, gftIO.GftTable):
        closing_price = closing_price.asMatrix()
    if isinstance(market_capital, gftIO.GftTable):
        market_capital = market_capital.asMatrix()

    # to check the frequency of the strategy, DAILY or MONTHLY
    dt_diff = holding.index.to_series().diff().mean()
    if dt_diff < pd.Timedelta('3 days'):
        periods = gsConst.Const.DAILY
    elif dt_diff > pd.Timedelta('3 days') and dt_diff < pd.Timedelta('10 days'):
        periods = gsConst.Const.WEEKLY
    else:
        periods = gsConst.Const.MONTHLY

    result = {}
    result[gsConst.Const.MaxHoldingNum] = max_holding_num(holding)
    result[gsConst.Const.MinHoldingNum] = min_holding_num(holding)
    result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)
    result[gsConst.Const.LatestHoldingNum] = latest_holding_num(holding)
    result[gsConst.Const.AverageHoldingNum] = average_holding_num(holding)

    if closing_price is not None:
        result[gsConst.Const.AverageHoldingNumPercentage] = average_holding_num_percentage(holding, closing_price)
        result[gsConst.Const.LatestHoldingNumPercentage] = latest_holding_num_percentage(holding, closing_price)
        if market_capital is not None:
            result[gsConst.Const.PortfolioValueMarketCapitalRatio] = portfolio_market_ratio(holding, closing_price, market_capital)
        result[gsConst.Const.AnnualReturnDispersionAverage] = holding_dispersion_std(holding, closing_price, period=periods)

    return result
