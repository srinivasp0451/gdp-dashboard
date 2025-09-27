# === Replace or add these updated functions into your Streamlit app ===

import math

def backtest_with_money_mgmt(df,
                            risk_per_trade_pct=0.5,     # percent of capital risked per trade
                            capital=100000,
                            target_atr=2.0,
                            sl_atr=1.0,
                            max_hold=14,
                            slippage_pct=0.0003,        # 0.03% slippage per trade side
                            commission_per_trade=1.0,   # flat commission per executed side
                            trailing_atr_mult=0.8,      # start trailing when move >= this * ATR
                            breakeven_after_atr=1.0):   # move in ATRs after which move SL to BE
    """
    Money-managed backtester, last-close entry, returns trade-level performance including position sizing,
    slippage, commission, trailing stop behavior.
    """
    df = df.copy().reset_index(drop=True)
    trades = []
    pos = None

    equity = capital

    for i in range(len(df)):
        r = df.loc[i]
        if pos is None and r['signal'] != 0:
            # determine SL and entry
            entry_price = r['Close']
            atr = r['ATR'] if (not np.isnan(r['ATR']) and r['ATR']>0) else 1e-6
            side = 'LONG' if r['signal']==1 else 'SHORT'
            if side == 'LONG':
                stop_price = entry_price - sl_atr * atr
                nominal_risk = entry_price - stop_price
            else:
                stop_price = entry_price + sl_atr * atr
                nominal_risk = stop_price - entry_price

            # position sizing: risk_per_trade_pct of equity -> position_size = (equity * risk_pct) / nominal_risk
            risk_amount = equity * (risk_per_trade_pct / 100.0)
            if nominal_risk <= 0:
                qty = 0
            else:
                qty = math.floor(risk_amount / nominal_risk)
            if qty <= 0:
                # cannot size this trade with given risk rules — skip
                continue

            # apply slippage on entry (worse fill)
            effective_entry = entry_price * (1 + slippage_pct) if side=='LONG' else entry_price * (1 - slippage_pct)
            # record position
            pos = {
                'entry_idx': i,
                'entry_date': r['Date'],
                'entry_price': entry_price,
                'effective_entry': effective_entry,
                'side': side,
                'qty': qty,
                'stop_price': stop_price,
                'atr': atr,
                'initial_nominal_risk': nominal_risk,
                'risk_amount': risk_amount,
                'entry_reason': r.get('reason',''),
                'highest_price': entry_price if side=='LONG' else None,
                'lowest_price': entry_price if side=='SHORT' else None,
                'commissions_paid': 0.0
            }
            # pay commission for entry side
            pos['commissions_paid'] += commission_per_trade
            continue

        # manage a live position
        if pos is not None:
            price = r['Close']
            exited = False
            exit_price = price
            exit_reason = None
            holding = i - pos['entry_idx']

            # update running high/low for trailing logic
            if pos['side']=='LONG':
                pos['highest_price'] = max(pos['highest_price'], price)
            else:
                pos['lowest_price'] = min(pos.get('lowest_price', price), price)

            # compute current trailing stop: after achieved X ATR move, trail at highest - trailing_atr_mult*ATR
            if pos['side']=='LONG':
                runup = (pos['highest_price'] - pos['entry_price'])
                if runup >= breakeven_after_atr * pos['atr']:
                    trailing = pos['highest_price'] - trailing_atr_mult * pos['atr']
                    # don't move stop below original stop
                    effective_stop = max(pos['stop_price'], trailing)
                else:
                    effective_stop = pos['stop_price']
                # check target, sl, and trailing
                target_price = pos['entry_price'] + target_atr * pos['atr']
                if price >= target_price:
                    exit_price = target_price
                    exit_reason = 'Target Hit'
                    exited = True
                elif price <= effective_stop:
                    exit_price = effective_stop
                    exit_reason = 'SL/Trailing Hit'
                    exited = True
                elif price < r['EMA_regime']:
                    exit_price = price
                    exit_reason = 'EMA Reversal'
                    exited = True
            else:  # SHORT
                runup = (pos['entry_price'] - pos['lowest_price'])
                if runup >= breakeven_after_atr * pos['atr']:
                    trailing = pos['lowest_price'] + trailing_atr_mult * pos['atr']
                    effective_stop = min(pos['stop_price'], trailing)
                else:
                    effective_stop = pos['stop_price']
                target_price = pos['entry_price'] - target_atr * pos['atr']
                if price <= target_price:
                    exit_price = target_price
                    exit_reason = 'Target Hit'
                    exited = True
                elif price >= effective_stop:
                    exit_price = effective_stop
                    exit_reason = 'SL/Trailing Hit'
                    exited = True
                elif price > r['EMA_regime']:
                    exit_price = price
                    exit_reason = 'EMA Reversal'
                    exited = True

            # forced exit by time
            if not exited and holding >= max_hold:
                exit_price = price
                exit_reason = 'Max Hold'
                exited = True

            # exit on opposite signal
            if not exited and r['signal'] != 0:
                if (pos['side']=='LONG' and r['signal']==-1) or (pos['side']=='SHORT' and r['signal']==1):
                    exit_price = price
                    exit_reason = 'Opposite Signal'
                    exited = True

            if exited:
                # apply slippage to exit (worse fill)
                if pos['side']=='LONG':
                    effective_exit = exit_price * (1 - slippage_pct)
                else:
                    effective_exit = exit_price * (1 + slippage_pct)

                # pay commission for exit
                commissions = commission_per_trade
                pos['commissions_paid'] += commissions

                # PnL per share
                pnl_per_unit = (effective_exit - pos['effective_entry']) if pos['side']=='LONG' else (pos['effective_entry'] - effective_exit)
                gross_pnl = pnl_per_unit * pos['qty']
                net_pnl = gross_pnl - pos['commissions_paid']  # subtract commissions
                # update equity
                equity += net_pnl

                # record trade
                trades.append({
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'effective_entry': pos['effective_entry'],
                    'exit_date': r['Date'],
                    'exit_price': exit_price,
                    'effective_exit': effective_exit,
                    'side': pos['side'],
                    'qty': int(pos['qty']),
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'net_pnl_pct_of_capital': (net_pnl / capital) * 100.0,
                    'holding_bars': holding,
                    'entry_reason': pos['entry_reason'],
                    'exit_reason': exit_reason,
                    'commissions': pos['commissions_paid']
                })
                pos = None

    trades_df = pd.DataFrame(trades)
    # compute summary stats
    if trades_df.empty:
        summary = {
            'total_trades': 0, 'positive_trades': 0, 'negative_trades':0,
            'accuracy': 0.0, 'net_pnl': 0.0, 'net_pnl_pct': 0.0, 'final_equity': equity
        }
    else:
        pos_ct = (trades_df['net_pnl']>0).sum()
        neg_ct = (trades_df['net_pnl']<=0).sum()
        net_pnl = trades_df['net_pnl'].sum()
        summary = {
            'total_trades': int(len(trades_df)),
            'positive_trades': int(pos_ct),
            'negative_trades': int(neg_ct),
            'accuracy': float(pos_ct / len(trades_df)),
            'net_pnl': float(net_pnl),
            'net_pnl_pct': float((net_pnl / capital) * 100.0),
            'final_equity': float(equity)
        }
    return trades_df, summary

# === Walk-forward updated to optimize for net_pnl after costs per fold ===

def walk_forward_optimize(df, n_splits=4, param_grid=None, risk_per_trade_pct=0.5, capital=100000,
                          slippage_pct=0.0003, commission_per_trade=1.0):
    """
    For each fold:
      - Tune params on train to maximize train net_pnl using backtest_with_money_mgmt
      - Test chosen params on OOS chunk, collect OOS net_pnl and accuracy
    """
    df = df.copy().reset_index(drop=True)
    if param_grid is None:
        param_grid = {
            'atr_mul': [0.6, 0.8, 1.0],
            'target_atr': [1.5, 2.0, 2.5],
            'sl_atr': [0.8, 1.0, 1.2],
            'max_hold': [8, 12, 18]
        }
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(param_grid[k] for k in keys))]
    n = len(df)
    fold_size = n // (n_splits + 1)
    wf_results = []
    for k in range(n_splits):
        train_end = (k+1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if len(train_df) < 50 or len(test_df) < 20:
            continue
        best = None
        best_metric = -np.inf
        for c in combos:
            # compute signals on train (use c['atr_mul'] for trigger)
            train_ind = compute_hpm_indicators(train_df)
            train_sig = generate_hpm_signals(train_ind, atr_mul=c['atr_mul'], min_mom=0.0015)
            tr_trades, tr_summary = backtest_with_money_mgmt(train_sig,
                                                            risk_per_trade_pct=risk_per_trade_pct,
                                                            capital=capital,
                                                            target_atr=c['target_atr'],
                                                            sl_atr=c['sl_atr'],
                                                            max_hold=c['max_hold'],
                                                            slippage_pct=slippage_pct,
                                                            commission_per_trade=commission_per_trade)
            metric = tr_summary['net_pnl']  # optimize net pnl after costs
            if metric > best_metric:
                best_metric = metric
                best = c
        # test best on OOS
        test_ind = compute_hpm_indicators(test_df)
        test_sig = generate_hpm_signals(test_ind, atr_mul=best['atr_mul'], min_mom=0.0015)
        test_trades, test_summary = backtest_with_money_mgmt(test_sig,
                                                            risk_per_trade_pct=risk_per_trade_pct,
                                                            capital=capital,
                                                            target_atr=best['target_atr'],
                                                            sl_atr=best['sl_atr'],
                                                            max_hold=best['max_hold'],
                                                            slippage_pct=slippage_pct,
                                                            commission_per_trade=commission_per_trade)
        wf_results.append({
            'fold': k+1,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best,
            'train_net_pnl': best_metric,
            'test_net_pnl': test_summary['net_pnl'],
            'test_accuracy': test_summary['accuracy'],
            'test_trades': test_summary['total_trades']
        })
    wf_df = pd.DataFrame(wf_results)
    agg = {}
    if not wf_df.empty:
        agg = {
            'folds': len(wf_df),
            'avg_test_net_pnl': float(wf_df['test_net_pnl'].mean()),
            'avg_test_accuracy': float(wf_df['test_accuracy'].mean())
        }
    return wf_df, agg

# === End of code block ===
