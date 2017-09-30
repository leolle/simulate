# -*- coding: utf-8 -*-


def future_simulation(initial_holding_position, df_portfolio_weight, df_price,
                      df_multiplier, dfexecute_price, df_execute_price_return,
                      df_trading_volume, df_commission, dict_trading_param):
    """
    Keyword Arguments:
    initial_holding_position --
    df_portfolio_weight      --
    df_price                 --
    df_multiplier            --
    dfexecute_price          --
    df_execute_price_return  --
    df_trading_volume        --
    df_commission            --
    dict_trading_param       --

    Return:
    future_position          --
    portfolio_value          --
    cash                     --
    single_period_return     --
    weight                   --
    cumulative_return        --
    future_trades            --
    """
