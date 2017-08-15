df_opts_weight = pd.DataFrame(data=np.nan, columns=specific_risk.columns,
                              index=exposure_constraint.index)
for target_date in exposure_constraint.index:
    # find the nearest date next to target date from specific risk
    dt_next_to_target = specific_risk.index.searchsorted(target_date)
    try:
        dt_next_to_target = specific_risk.index[dt_next_to_target]
    except :
        continue
    target_specific_risk = specific_risk.loc[dt_next_to_target, :]
    logger.debug('target date: %s', target_date)
    logger.debug('next to target date: %s', dt_next_to_target)
    # drop duplicated rows at date
    df_industries_asset_weight = asset_weights.drop_duplicates(
        subset=['date', 'symbol'])
#    try:
#        df_industries_asset_init_weight = df_industries_asset_weight.dropna()
#    except KeyError:
#        raise KeyError('invalid input date: %s' % target_date)
    try:
        df_industries_asset_init_weight = df_industries_asset_weight[
                df_industries_asset_weight['date'] == target_date].dropna()
    except KeyError:
        raise KeyError('invalid input date: %s' % target_date)

    df_industries_asset_init_weight = df_industries_asset_init_weight.dropna(
        axis=0, subset=['industry', 'symbol'], how='any')
    unique_symbol = df_industries_asset_init_weight['symbol'].unique()
    target_symbols = target_specific_risk.index.intersection(unique_symbol)
    if position_limit > len(target_symbols):
        logger.debug("position limit is bigger than total symbols.")
        position_limit = len(target_symbols)
    
    # get random symbols at the target position limit
    arr = list(range(len(target_symbols)))
    np.random.shuffle(arr)
    target_symbols = target_symbols[arr[:position_limit]]
    #logger.debug('target_symbols: %s', len(target_symbols))

    df_industries_asset_target_init_weight = df_industries_asset_init_weight.\
                                             loc[df_industries_asset_init_weight['symbol'].isin(target_symbols)]
    df_pivot_industries_asset_weights = pd.pivot_table(
        df_industries_asset_target_init_weight, values='value', index=['date'],
        columns=['industry', 'symbol'])
    df_pivot_industries_asset_weights = df_pivot_industries_asset_weights.fillna(0)
    #logger.debug("set OOTV to hierachical index dataframe.")
    noa = len(target_symbols)
    if noa < 1:
        raise ValueError("no intersected symbols from specific risk and initial holding.")
    #logger.debug("number of asset: %s", noa)
    # get the ordered column list
    idx_level_0_value = df_pivot_industries_asset_weights.columns.get_level_values(0)
    idx_level_0_value = idx_level_0_value.drop_duplicates()
    idx_level_1_value = df_pivot_industries_asset_weights.columns.get_level_values(1)
    asset_expected_return = asset_return.loc[:target_date, idx_level_1_value].fillna(0)

    diag = specific_risk.loc[dt_next_to_target, idx_level_1_value]
    delta = pd.DataFrame(np.diag(diag), index=diag.index,
                         columns=diag.index).fillna(0)

    big_X = get_factor_exposure(risk_model, ls_factor, target_date,
                                idx_level_1_value)
    big_X = big_X.fillna(0)
    all_factors = big_X.columns

    covariance_matrix = risk_model['ret_cov'].set_index('date')

    cov_matrix = covariance_matrix.loc[dt_next_to_target]
    cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2',
                                  values='value')
    cov_matrix = cov_matrix.reindex(all_factors, all_factors, fill_value=np.nan)

    rets_mean = logrels(asset_expected_return).mean()

    covariance_matrix = risk_model['ret_cov'].set_index('date')

    cov_matrix = covariance_matrix.loc[dt_next_to_target]
    cov_matrix = cov_matrix.pivot(index='factorid1', columns='factorid2',
                                  values='value')
    cov_matrix = cov_matrix.reindex(all_factors, all_factors, fill_value=np.nan)

    cov_matrix_V = big_X.dot(cov_matrix).dot(big_X.T) + delta    

    df_asset_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                                   index=idx_level_1_value)
    df_group_weight = pd.DataFrame({'lower': [0.0], 'upper': [1.0]},
                                   index=idx_level_0_value)

    df_factor_exposure_bound = pd.DataFrame(index=exposure_constraint.columns, columns=[['lower', 'upper']])
    df_factor_exposure_bound.lower = exposure_constraint.ix[target_date].apply(lambda x: set_lower_limit(x))
    df_factor_exposure_bound.upper = exposure_constraint.ix[target_date].apply(lambda x: set_upper_limit(x))

    df_factor_exposure_lower_bnd = pd.Series(data=[big_X.values.min()]*len(all_factors), index=big_X.columns)
    df_factor_exposure_upper_bnd = pd.Series(data=[big_X.values.max()]*len(all_factors), index=big_X.columns)

    df_factor_exposure_lower_bnd.loc[df_factor_exposure_bound.index] = df_factor_exposure_bound.lower
    df_factor_exposure_upper_bnd.loc[df_factor_exposure_bound.index] = df_factor_exposure_bound.upper
    
    # for group weight constraint
    groups = df_pivot_industries_asset_weights.groupby(
        axis=1, level=0, sort=False, group_keys=False).count().\
        iloc[-1, :].values
    num_group = len(groups)
    num_asset = np.sum(groups)

    #logger.debug('number of assets in groups: %s', groups)
    #logger.debug('number of groups: %s', num_group)
    
    G_sparse_list = []
    for i in range(num_group):
        for j in range(groups[i]):
            G_sparse_list.append(i)
    Group_sub = spmatrix(1.0, G_sparse_list, range(num_asset))

    # Factor model portfolio optimization.
    w = cvx.Variable(noa)
    G_sum = np.array(matrix(Group_sub))*w
    f = big_X.T.values*w
    gamma = cvx.Parameter(sign='positive')
    Lmax = cvx.Parameter()
    ret = w.T * rets_mean.values
    risk = cvx.quad_form(f, cov_matrix.values) + cvx.quad_form(w, delta.values)
    eq_constraint = [cvx.sum_entries(w) == 1,
                     cvx.norm(w, 1) <= Lmax]
    l_eq_constraint = [w >= df_asset_weight.lower.values,
                       w <= df_asset_weight.upper.values,
                       G_sum >= df_group_weight.lower.values,
                       G_sum <= df_group_weight.upper.values]
    if exposure_constraint is not None:
        l_eq_constraint.append(f >= df_factor_exposure_lower_bnd.values)
        l_eq_constraint.append(f <= df_factor_exposure_upper_bnd.values)
    #Portfolio optimization with a leverage limit and a bound on risk
    Lmax.value = 1
    gamma.value = 1

    if target_mode == 0:
        # Solve the factor model problem.
        prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                                  eq_constraint+l_eq_constraint)
    if target_mode == 1:
        # minimum risk subject to target return, Markowitz Mean_Variance Portfolio
        prob_factor = cvx.Problem(cvx.Maximize(-gamma*risk),
                                  [ret >= target_return]+l_eq_constraint+eq_constraint)
    if target_mode == 2:
        # Computes a tangency portfolio, i.e. a maximum Sharpe ratio portfolio
        prob_factor = cvx.Problem(cvx.Maximize(ret),
                                  [risk <= target_risk]+l_eq_constraint+eq_constraint)
    prob_factor.solve(verbose=False)
    logger.debug(prob_factor.status)
    if prob_factor.status == 'infeasible':
        df_opts_weight.loc[target_date, idx_level_1_value] = np.nan
    else:
        df_opts_weight.loc[target_date, idx_level_1_value] = np.array(w.value.astype(np.float64)).T
