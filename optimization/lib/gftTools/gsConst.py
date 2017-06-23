class MetaConst(type):
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise TypeError("const {} cannot be changed".format(key))


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
    PortRisk="PortRisk"
    BenchmarkRisk="BenchmarkRisk"
    PortActiveRisk="PortActiveRisk"
    InteractionRisk="InteractionRisk"
    FactorRisk="FactorRisk"
    SpecificRisk="SpecificRisk"
    IndustryFMCAR="IndustryFMCAR"
    StyleFMCAR="StyleFMCAR"
    IndStyleFMCAR="IndStyleFMCAR"
    PortExpo="PortExpo"
    BenchmarkExpo="BenchmarkExpo"
    #BELOW IS FOR MACHINE LEARNING
    WINDOW_SIZE = 5
    TOTAL_BINS = 10
 