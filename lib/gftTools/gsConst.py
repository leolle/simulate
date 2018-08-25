class MetaConst(type):
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise TypeError("const {} cannot be changed".format(key))


HTTP_HEADER_PARA_NAME_USER = "user"
HTTP_HEADER_PARA_NAME_PASSWORD = "pwd"
HTTP_HEADER_PARA_NAME_SKILL_INST_GID = "skillInstGid"
HTTP_HEADER_PARA_MD5 = "md5"
HTTP_HEADER_PARA_MR = 'mr'
HTTP_HEADER_PARA_SR = 'sr'
HTTP_HEADER_PARA_REQ_NO = 'rn'
HTTP_HEADER_PARA_IS_BYTES = 'isBytes'

class BaseConst(MetaConst):
    pass


class Const(BaseConst):
    Holding = "HOLDING"
    PortfolioValue = "PORTFOLIO_VALUE"
    SinglePeriodReturn = "SINGLE_PERIOD_RETURN"
    Weights = "WEIGHTS"
    CumulativeReturn = "CUMULATIVE_RETURN"
    Turnover = "TURNOVER"

    # PnL fitness
    AnnualDownVolatility = "ANNUAL_DOWN_VOLATILITY"
    AnnualReturn = "ANNUAL_RETURN"
    AnnualVolatility = "ANNUAL_VOLATILITY"
    DownStdReturn = "DOWN_STD_RETURN"
    EndDate = "END_DATE"
    StartDate = "START_DATE"
    MaxDrawdownRate = "MAX_DRAWDOWN_RATE"
    StdReturn = "STD_RETURN"
    TotalTradingDays = "TOTAL_TRADING_DAYS"
    SharpeRatio = "SHARPE_RATIO"
    SortinoRatio = "SORTINO_RATIO"
    BenchmarkSharpeRatio = "BENCHMARK_SHARPE_RATIO"
    BenchmarkAnnualReturn = "BENCHMARK_ANNUAL_RETURN"
    BenchmarkAnnualVolatility = "BENCHMARK_ANNUAL_VOLATILITY"
    BenchmarkMaxDrawdownRate = "BENCHMARK_MAX_DRAWDOWN_RATE"
    BenchmarkCumulativeReturn = "BENCHMARK_CUMULATIVE_RETURN"
    BenchmarStdReturn = "BENCHMARK_STD_RETURN"
    PortfolioValueMarketCapitalRatio = "PORTFOLIO_VALUE_MARKET_CAPITAL_RATIO"
    MaxHoldingNum = "MAX_HOLDING_NUMBER"
    MinHoldingNum = "MIN_HOLDING_NUMBER"
    AverageHoldingNum = "AVERAGE_HOLDING_NUMBER"
    LatestHoldingNum = "LATEST_HOLDING_NUMBER"
    AverageHoldingNumPercentage = "AVERAGE_HOLDING_NUM_PERCENTAGE"
    LatestHoldingNumPercentage = "LATEST_HOLDING_NUM_PERCENTAGE"
    AnnualReturnDispersionAverage = "ANNUAL_RETURN_DISPERSION_AVERAGE"
    ExcessAnnualReturn = "EXCESS_ANNUAL_RETURN"
    ExcessSinglePeriodReturns = "EXCESS_SINGLE_PERIOD_RETURNS"
    ExcessCumulativeReturns = "EXCESS_CUMULATIVE_RETURNS"
    AggregateReturnOneYear = "AGGREGATE_RETURN_ONE_YEAR"
    AggregateReturnOneDay = "AGGREGATE_RETURN_ONE_DAY"
    AggregateReturnOneMonth = "AGGREGATE_RETURN_ONE_MONTH"
    AggregateReturnThreeMonth = "AGGREGATE_RETURN_THREE_MONTH"
    AggregateReturnSixMonth = "AGGREGATE_RETURN_SIX_MONTH"
    AggregateReturnThreeYear = "AGGREGATE_RETURN_THREE_YEAR"
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    YEARLY = 1

    ##below is for risk attribution
    PortRisk = "PortRisk"
    BenchmarkRisk = "BenchmarkRisk"
    PortActiveRisk = "PortActiveRisk"
    InteractionRisk = "InteractionRisk"
    FactorRisk = "FactorRisk"
    SpecificRisk = "SpecificRisk"
    IndustryFMCAR = "IndustryFMCAR"
    StyleFMCAR = "StyleFMCAR"
    IndStyleFMCAR = "IndStyleFMCAR"
    PortExpo = "PortExpo"
    BenchmarkExpo = "BenchmarkExpo"

    PortExpoInd = "PortExpoInd"
    PortExpoSty = "PortExpoSty"
    BenchmarkExpoInd = "BenchmarkExpoInd"
    BenchmarkExpoSty = "BenchmarkExpoSty"
    TotIndustryFMCAR = "TotIndustryFMCAR"
    TotStyleFMCAR = "TotStyleFMCAR"

    # BELOW IS FOR MACHINE LEARNING
    WINDOW_SIZE = 5
    TOTAL_BINS = 10
    df_data = "result"
    num_mean_accuracy = "mean_accuracy"
    dict_score = "scores"
    rsquare = 'rsquare'
    match_pair_pct = 'match_pair_pct'
    max_depth = 'max_depth'

    # Below is for Brinson Attribution
    portweight = 'portweight'
    bmweight = 'bmweight'
    portfolio_length = 'portfolio_length'
    benchmark_length = 'benchmark_length'

    # portfolio optimization mode
    MinimumRisk = 0
    MinimumRiskUnderReturn = 1
    MaximumReturnUnderRisk = 2
    Feasible = 1
    Infeasible = 0

    country_gid_str = 'E8A54A95C9264162BEEC88B9CF65C78B'
    weight_gid_str = '07C1FF96E4EA49BE8DFA12B6CBAB289F'
    ret_gid_str = '925A3DAF6E954E4AAD405B3EBA03F675'
