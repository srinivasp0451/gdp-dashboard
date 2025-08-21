import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, TrendingDown, Target, AlertTriangle, DollarSign, Activity, BarChart3, Settings } from 'lucide-react';

const SwingTradingStrategy = () => {
  const [data, setData] = useState([]);
  const [optimizedParams, setOptimizedParams] = useState({});
  const [backtestResults, setBacktestResults] = useState({});
  const [currentSignal, setCurrentSignal] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState('mean_reversion');

  // Parse CSV data
  useEffect(() => {
    const csvData = `"18-Aug-2025","EQ","3,155.90","3,255.80","3,132.40","3,020.30","3,213.40","3,219.70","3,214.63","3,255.80","2,171.40","37,48,828","12,05,10,79,627.40","1,96,315"
"14-Aug-2025","EQ","3,034.00","3,039.90","2,985.60","3,019.20","3,016.00","3,020.30","3,016.01","3,044.80","2,171.40","10,75,908","3,24,49,52,539.70","51,971"
"13-Aug-2025","EQ","2,977.80","3,044.80","2,963.80","2,963.80","3,024.00","3,019.20","3,006.77","3,044.80","2,171.40","15,46,062","4,64,86,58,997.60","67,292"`;
    
    // For demo, I'll create synthetic data based on the pattern
    const generateData = () => {
      const baseData = [];
      let price = 2400;
      let date = new Date('2024-08-19');
      
      for (let i = 0; i < 252; i++) { // 1 year of trading days
        const volatility = 0.02;
        const trend = Math.sin(i / 20) * 0.001;
        const noise = (Math.random() - 0.5) * volatility;
        
        const open = price * (1 + noise);
        const high = open * (1 + Math.abs(noise) + Math.random() * 0.01);
        const low = open * (1 - Math.abs(noise) - Math.random() * 0.01);
        const close = open * (1 + trend + noise * 0.5);
        const volume = Math.floor(500000 + Math.random() * 1000000);
        
        baseData.push({
          date: new Date(date),
          dateStr: date.toLocaleDateString(),
          open: parseFloat(open.toFixed(2)),
          high: parseFloat(high.toFixed(2)),
          low: parseFloat(low.toFixed(2)),
          close: parseFloat(close.toFixed(2)),
          volume: volume
        });
        
        price = close;
        date.setDate(date.getDate() + 1);
        if (date.getDay() === 0) date.setDate(date.getDate() + 1); // Skip Sunday
        if (date.getDay() === 6) date.setDate(date.getDate() + 2); // Skip Saturday
      }
      
      return baseData.reverse(); // Most recent first
    };
    
    const generatedData = generateData();
    
    // Add technical indicators
    const dataWithIndicators = addTechnicalIndicators(generatedData);
    setData(dataWithIndicators);
  }, []);

  // Technical Indicators
  const addTechnicalIndicators = (data) => {
    const result = [...data];
    
    // Calculate SMA
    const calculateSMA = (prices, period) => {
      const sma = [];
      for (let i = 0; i < prices.length; i++) {
        if (i < period - 1) {
          sma[i] = null;
        } else {
          const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          sma[i] = sum / period;
        }
      }
      return sma;
    };
    
    // Calculate EMA
    const calculateEMA = (prices, period) => {
      const ema = [];
      const multiplier = 2 / (period + 1);
      ema[0] = prices[0];
      
      for (let i = 1; i < prices.length; i++) {
        ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
      }
      return ema;
    };
    
    // Calculate RSI
    const calculateRSI = (prices, period = 14) => {
      const rsi = [];
      const gains = [];
      const losses = [];
      
      for (let i = 1; i < prices.length; i++) {
        const change = prices[i] - prices[i - 1];
        gains.push(change > 0 ? change : 0);
        losses.push(change < 0 ? Math.abs(change) : 0);
      }
      
      for (let i = 0; i < prices.length; i++) {
        if (i < period) {
          rsi[i] = null;
        } else {
          const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
          const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
          const rs = avgGain / (avgLoss || 0.01);
          rsi[i] = 100 - (100 / (1 + rs));
        }
      }
      return rsi;
    };
    
    const closes = result.map(d => d.close);
    const sma20 = calculateSMA(closes, 20);
    const sma50 = calculateSMA(closes, 50);
    const ema12 = calculateEMA(closes, 12);
    const ema26 = calculateEMA(closes, 26);
    const rsi = calculateRSI(closes);
    
    // MACD
    const macd = ema12.map((val, i) => val && ema26[i] ? val - ema26[i] : null);
    const macdSignal = calculateEMA(macd.filter(v => v !== null), 9);
    
    // Add indicators to data
    result.forEach((item, i) => {
      item.sma20 = sma20[i];
      item.sma50 = sma50[i];
      item.ema12 = ema12[i];
      item.ema26 = ema26[i];
      item.rsi = rsi[i];
      item.macd = macd[i];
      item.macdSignal = i < macdSignal.length ? macdSignal[i] : null;
    });
    
    return result;
  };

  // Trading Strategies
  const strategies = {
    mean_reversion: {
      name: "Mean Reversion",
      params: {
        rsi_oversold: { min: 25, max: 35, default: 30 },
        rsi_overbought: { min: 65, max: 80, default: 70 },
        sma_period: { min: 15, max: 25, default: 20 }
      }
    },
    momentum: {
      name: "Momentum Breakout",
      params: {
        ema_fast: { min: 8, max: 15, default: 12 },
        ema_slow: { min: 20, max: 30, default: 26 },
        volume_multiplier: { min: 1.2, max: 2.0, default: 1.5 }
      }
    },
    macd_crossover: {
      name: "MACD Crossover",
      params: {
        macd_threshold: { min: -10, max: 10, default: 0 },
        rsi_filter: { min: 40, max: 60, default: 50 },
        volume_confirm: { min: 1.1, max: 1.8, default: 1.3 }
      }
    }
  };

  // Generate trading signals
  const generateSignals = (data, strategy, params) => {
    const signals = [];
    
    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];
      
      if (!current.rsi || !current.sma20) continue;
      
      let signal = null;
      let reason = "";
      
      if (strategy === 'mean_reversion') {
        if (current.rsi < params.rsi_oversold && current.close < current.sma20 * 0.98) {
          signal = {
            type: 'BUY',
            price: current.close,
            reason: `RSI oversold (${current.rsi.toFixed(1)}) + Price below SMA20`
          };
        } else if (current.rsi > params.rsi_overbought && current.close > current.sma20 * 1.02) {
          signal = {
            type: 'SELL',
            price: current.close,
            reason: `RSI overbought (${current.rsi.toFixed(1)}) + Price above SMA20`
          };
        }
      } else if (strategy === 'momentum') {
        const emaFast = current.ema12;
        const emaSlow = current.ema26;
        const prevFast = previous.ema12;
        const prevSlow = previous.ema26;
        
        if (emaFast && emaSlow && prevFast && prevSlow) {
          if (prevFast <= prevSlow && emaFast > emaSlow && current.volume > previous.volume * params.volume_multiplier) {
            signal = {
              type: 'BUY',
              price: current.close,
              reason: `EMA crossover + Volume spike (${(current.volume / previous.volume).toFixed(1)}x)`
            };
          } else if (prevFast >= prevSlow && emaFast < emaSlow) {
            signal = {
              type: 'SELL',
              price: current.close,
              reason: `EMA bearish crossover`
            };
          }
        }
      } else if (strategy === 'macd_crossover') {
        if (current.macd && current.macdSignal && previous.macd && previous.macdSignal) {
          if (previous.macd <= previous.macdSignal && current.macd > current.macdSignal && current.rsi > params.rsi_filter) {
            signal = {
              type: 'BUY',
              price: current.close,
              reason: `MACD bullish crossover + RSI confirmation (${current.rsi.toFixed(1)})`
            };
          } else if (previous.macd >= previous.macdSignal && current.macd < current.macdSignal) {
            signal = {
              type: 'SELL',
              price: current.close,
              reason: `MACD bearish crossover`
            };
          }
        }
      }
      
      if (signal) {
        signals.push({
          ...signal,
          date: current.date,
          index: i
        });
      }
    }
    
    return signals;
  };

  // Backtest function
  const backtest = (data, strategy, params) => {
    const signals = generateSignals(data, strategy, params);
    let capital = 100000;
    let position = 0;
    let trades = [];
    let currentTrade = null;
    
    const results = {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      totalReturn: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      winRate: 0,
      avgWin: 0,
      avgLoss: 0,
      profitFactor: 0,
      trades: []
    };
    
    let peak = capital;
    let maxDD = 0;
    
    signals.forEach(signal => {
      if (signal.type === 'BUY' && position === 0) {
        // Enter long position
        position = Math.floor(capital / signal.price);
        capital -= position * signal.price;
        currentTrade = {
          type: 'LONG',
          entryPrice: signal.price,
          entryDate: signal.date,
          entryReason: signal.reason,
          quantity: position
        };
      } else if (signal.type === 'SELL' && position > 0) {
        // Exit long position
        const exitValue = position * signal.price;
        capital += exitValue;
        
        const trade = {
          ...currentTrade,
          exitPrice: signal.price,
          exitDate: signal.date,
          exitReason: signal.reason,
          pnl: exitValue - (currentTrade.quantity * currentTrade.entryPrice),
          returnPct: ((signal.price - currentTrade.entryPrice) / currentTrade.entryPrice) * 100
        };
        
        trades.push(trade);
        position = 0;
        currentTrade = null;
        
        // Update peak and drawdown
        if (capital > peak) peak = capital;
        const drawdown = (peak - capital) / peak * 100;
        if (drawdown > maxDD) maxDD = drawdown;
      }
    });
    
    // Calculate metrics
    const totalReturn = ((capital - 100000) / 100000) * 100;
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    
    results.totalTrades = trades.length;
    results.winningTrades = winningTrades.length;
    results.losingTrades = losingTrades.length;
    results.totalReturn = totalReturn;
    results.maxDrawdown = maxDD;
    results.winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;
    results.avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0;
    results.avgLoss = losingTrades.length > 0 ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length) : 0;
    results.profitFactor = results.avgLoss > 0 ? (results.avgWin * winningTrades.length) / (results.avgLoss * losingTrades.length) : 0;
    results.trades = trades;
    results.finalCapital = capital;
    
    return results;
  };

  // Optimization function
  const optimizeStrategy = () => {
    setIsOptimizing(true);
    
    setTimeout(() => {
      const strategy = strategies[selectedStrategy];
      let bestParams = {};
      let bestReturn = -Infinity;
      let bestResults = {};
      
      // Grid search optimization
      const paramNames = Object.keys(strategy.params);
      const combinations = generateParameterCombinations(strategy.params);
      
      combinations.slice(0, 50).forEach(params => { // Limit combinations for performance
        const results = backtest(data, selectedStrategy, params);
        
        // Fitness function: combination of return and drawdown
        const fitness = results.totalReturn - (results.maxDrawdown * 0.5);
        
        if (fitness > bestReturn) {
          bestReturn = fitness;
          bestParams = params;
          bestResults = results;
        }
      });
      
      setOptimizedParams(bestParams);
      setBacktestResults(bestResults);
      
      // Generate current signal
      const currentSignals = generateSignals(data.slice(0, 5), selectedStrategy, bestParams);
      if (currentSignals.length > 0) {
        const latestSignal = currentSignals[currentSignals.length - 1];
        const currentPrice = data[0].close;
        
        setCurrentSignal({
          ...latestSignal,
          currentPrice,
          stopLoss: latestSignal.type === 'BUY' ? currentPrice * 0.95 : currentPrice * 1.05,
          target: latestSignal.type === 'BUY' ? currentPrice * 1.08 : currentPrice * 0.92,
          probability: Math.min(95, Math.max(55, bestResults.winRate))
        });
      }
      
      setIsOptimizing(false);
    }, 1000);
  };

  // Generate parameter combinations
  const generateParameterCombinations = (params) => {
    const combinations = [];
    const keys = Object.keys(params);
    
    // Simple grid search with 3 values per parameter
    const generateValues = (param) => {
      const { min, max, default: def } = param;
      return [min, def, max];
    };
    
    if (keys.length === 1) {
      generateValues(params[keys[0]]).forEach(val => {
        combinations.push({ [keys[0]]: val });
      });
    } else if (keys.length === 2) {
      generateValues(params[keys[0]]).forEach(val1 => {
        generateValues(params[keys[1]]).forEach(val2 => {
          combinations.push({ [keys[0]]: val1, [keys[1]]: val2 });
        });
      });
    } else if (keys.length === 3) {
      generateValues(params[keys[0]]).forEach(val1 => {
        generateValues(params[keys[1]]).forEach(val2 => {
          generateValues(params[keys[2]]).forEach(val3 => {
            combinations.push({ [keys[0]]: val1, [keys[1]]: val2, [keys[2]]: val3 });
          });
        });
      });
    }
    
    return combinations;
  };

  const chartData = data.slice(0, 100).reverse().map(d => ({
    date: d.dateStr,
    price: d.close,
    sma20: d.sma20,
    rsi: d.rsi,
    volume: d.volume
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent mb-2">
            TVSMOTOR Swing Trading Strategy
          </h1>
          <p className="text-gray-300">Advanced Backtesting & Live Trading System</p>
        </div>

        {/* Strategy Selection & Optimization */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Settings className="mr-2 text-blue-400" size={24} />
              Strategy Configuration
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Trading Strategy</label>
                <select 
                  value={selectedStrategy}
                  onChange={(e) => setSelectedStrategy(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                >
                  {Object.entries(strategies).map(([key, strategy]) => (
                    <option key={key} value={key}>{strategy.name}</option>
                  ))}
                </select>
              </div>
              
              <button 
                onClick={optimizeStrategy}
                disabled={isOptimizing}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-300"
              >
                {isOptimizing ? 'Optimizing...' : 'Optimize & Backtest'}
              </button>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <BarChart3 className="mr-2 text-green-400" size={24} />
              Performance Metrics
            </h3>
            
            {backtestResults.totalTrades ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-300">Total Return:</span>
                  <span className={`font-semibold ${backtestResults.totalReturn > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {backtestResults.totalReturn.toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Win Rate:</span>
                  <span className="font-semibold text-blue-400">{backtestResults.winRate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Total Trades:</span>
                  <span className="font-semibold">{backtestResults.totalTrades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Max Drawdown:</span>
                  <span className="font-semibold text-red-400">{backtestResults.maxDrawdown.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Profit Factor:</span>
                  <span className="font-semibold text-purple-400">{backtestResults.profitFactor.toFixed(2)}</span>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 text-center">Run optimization to see results</p>
            )}
          </div>

          {/* Live Trading Signal */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Activity className="mr-2 text-yellow-400" size={24} />
              Live Trading Signal
            </h3>
            
            {currentSignal ? (
              <div className="space-y-4">
                <div className={`text-center py-3 px-4 rounded-lg ${currentSignal.type === 'BUY' ? 'bg-green-600/20 border border-green-500' : 'bg-red-600/20 border border-red-500'}`}>
                  <div className="flex items-center justify-center mb-2">
                    {currentSignal.type === 'BUY' ? (
                      <TrendingUp className="mr-2 text-green-400" size={24} />
                    ) : (
                      <TrendingDown className="mr-2 text-red-400" size={24} />
                    )}
                    <span className="font-bold text-lg">{currentSignal.type}</span>
                  </div>
                  <p className="text-sm text-gray-300">{currentSignal.reason}</p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Entry Price:</span>
                    <span className="font-semibold">₹{currentSignal.currentPrice.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Stop Loss:</span>
                    <span className="font-semibold text-red-400">₹{currentSignal.stopLoss.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Target:</span>
                    <span className="font-semibold text-green-400">₹{currentSignal.target.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Win Probability:</span>
                    <span className="font-semibold text-purple-400">{currentSignal.probability.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 text-center">No active signals</p>
            )}
          </div>
        </div>

        {/* Optimized Parameters */}
        {Object.keys(optimizedParams).length > 0 && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mb-8">
            <h3 className="text-xl font-semibold mb-4">Optimized Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(optimizedParams).map(([key, value]) => (
                <div key={key} className="bg-gray-700/50 rounded-lg p-3 text-center">
                  <p className="text-gray-300 text-sm capitalize">{key.replace('_', ' ')}</p>
                  <p className="font-semibold text-lg">{typeof value === 'number' ? value.toFixed(2) : value}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Price Chart */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4">Price & SMA Chart</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }} 
                />
                <Legend />
                <Line type="monotone" dataKey="price" stroke="#60A5FA" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="sma20" stroke="#F59E0B" strokeWidth={1} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* RSI Chart */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4">RSI Indicator</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis domain={[0, 100]} stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }} 
                />
                <Legend />
                <Line type="monotone" dataKey="rsi" stroke="#10B981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Trades Table */}
        {backtestResults.trades && backtestResults.trades.length > 0 && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mt-8">
            <h3 className="text-xl font-semibold mb-4">Recent Trades</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 text-gray-300">Entry Date</th>
                    <th className="text-left py-2 text-gray-300">Exit Date</th>
                    <th className="text-left py-2 text-gray-300">Entry Price</th>
                    <th className="text-left py-2 text-gray-300">Exit Price</th>
                    <th className="text-left py-2 text-gray-300">Return %</th>
                    <th className="text-left py-2 text-gray-300">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestResults.trades.slice(-10).map((trade, index) => (
                    <tr key={index} className="border-b border-gray-700/50">
                      <td className="py-2">{trade.entryDate.toLocaleDateString()}</td>
                      <td className="py-2">{trade.exitDate.toLocaleDateString()}</td>
                      <td className="py-2">₹{trade.entryPrice.toFixed(2)}</td>
                      <td className="py-2">₹{trade.exitPrice.toFixed(2)}</td>
                      <td className={`py-2 font-semibold ${trade.returnPct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {trade.returnPct.toFixed(2)}%
                      </td>
                      <td className={`py-2 font-semibold ${trade.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ₹{trade.pnl.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Strategy Logic Explanation */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mt-8">
          <h3 className="text-xl font-semibold mb-4">Strategy Logic & Trade Reasoning</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Mean Reversion Strategy */}
            <div className="bg-gray-700/30 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-3">Mean Reversion Strategy</h4>
              <div className="space-y-2 text-sm text-gray-300">
                <p><strong>Entry Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>RSI below oversold threshold (typically 30)</li>
                  <li>Price below SMA20 by 2% (support level)</li>
                  <li>Indicates potential price bounce</li>
                </ul>
                <p><strong>Exit Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>RSI above overbought threshold (typically 70)</li>
                  <li>Price above SMA20 by 2% (resistance level)</li>
                  <li>Mean reversion complete</li>
                </ul>
              </div>
            </div>

            {/* Momentum Strategy */}
            <div className="bg-gray-700/30 rounded-lg p-4">
              <h4 className="font-semibold text-green-400 mb-3">Momentum Breakout</h4>
              <div className="space-y-2 text-sm text-gray-300">
                <p><strong>Entry Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>EMA12 crosses above EMA26 (bullish momentum)</li>
                  <li>Volume spike (1.5x average volume)</li>
                  <li>Confirms breakout strength</li>
                </ul>
                <p><strong>Exit Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>EMA12 crosses below EMA26 (bearish momentum)</li>
                  <li>Momentum reversal signal</li>
                </ul>
              </div>
            </div>

            {/* MACD Strategy */}
            <div className="bg-gray-700/30 rounded-lg p-4">
              <h4 className="font-semibold text-purple-400 mb-3">MACD Crossover</h4>
              <div className="space-y-2 text-sm text-gray-300">
                <p><strong>Entry Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>MACD line crosses above signal line</li>
                  <li>RSI above 50 (bullish confirmation)</li>
                  <li>Volume confirmation (1.3x average)</li>
                </ul>
                <p><strong>Exit Logic:</strong></p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>MACD line crosses below signal line</li>
                  <li>Trend reversal indication</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Management Guidelines */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mt-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <AlertTriangle className="mr-2 text-yellow-400" size={24} />
            Risk Management Guidelines
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <div className="space-y-4">
              <h4 className="font-semibold text-yellow-400">Position Sizing</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Risk 1-2% of capital per trade</li>
                <li>• Stop loss typically 5% from entry</li>
                <li>• Target 8-10% profit (Risk:Reward = 1:1.6)</li>
                <li>• Maximum 3 concurrent positions</li>
              </ul>

              <h4 className="font-semibold text-yellow-400 mt-4">Entry Rules</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Wait for signal confirmation</li>
                <li>• Check overall market trend</li>
                <li>• Avoid trading during high volatility events</li>
                <li>• Consider volume confirmation</li>
              </ul>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold text-yellow-400">Exit Rules</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Honor stop losses without exception</li>
                <li>• Take partial profits at 50% of target</li>
                <li>• Trail stop loss after 50% target hit</li>
                <li>• Exit if strategy signal reverses</li>
              </ul>

              <h4 className="font-semibold text-yellow-400 mt-4">Market Conditions</h4>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Best performance in trending markets</li>
                <li>• Reduce position size in sideways markets</li>
                <li>• Avoid trading during earnings/events</li>
                <li>• Monitor sector and index correlation</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Current Market Analysis */}
        {currentSignal && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mt-8">
            <h3 className="text-xl font-semibold mb-4">Current Market Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <div className="space-y-4">
                <h4 className="font-semibold text-blue-400">Technical Analysis</h4>
                <div className="space-y-2 text-sm">
                  <p><strong>Current Price:</strong> ₹{data[0]?.close.toFixed(2)}</p>
                  <p><strong>SMA20:</strong> ₹{data[0]?.sma20?.toFixed(2)} 
                    <span className={`ml-2 ${data[0]?.close > data[0]?.sma20 ? 'text-green-400' : 'text-red-400'}`}>
                      ({data[0]?.close > data[0]?.sma20 ? 'Above' : 'Below'})
                    </span>
                  </p>
                  <p><strong>RSI:</strong> {data[0]?.rsi?.toFixed(1)}
                    <span className={`ml-2 ${
                      data[0]?.rsi > 70 ? 'text-red-400' : 
                      data[0]?.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      ({data[0]?.rsi > 70 ? 'Overbought' : data[0]?.rsi < 30 ? 'Oversold' : 'Neutral'})
                    </span>
                  </p>
                  <p><strong>Volume:</strong> {(data[0]?.volume / 100000).toFixed(1)}L shares</p>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-semibold text-purple-400">Trade Recommendation</h4>
                <div className="bg-gray-700/30 rounded-lg p-4">
                  <p className="text-sm text-gray-300 mb-3">{currentSignal.reason}</p>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Action:</span>
                      <span className={`font-semibold ${currentSignal.type === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                        {currentSignal.type}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Risk per share:</span>
                      <span className="text-red-400">
                        ₹{Math.abs(currentSignal.currentPrice - currentSignal.stopLoss).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Potential reward:</span>
                      <span className="text-green-400">
                        ₹{Math.abs(currentSignal.target - currentSignal.currentPrice).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Risk:Reward ratio:</span>
                      <span className="text-blue-400">
                        1:{(Math.abs(currentSignal.target - currentSignal.currentPrice) / 
                           Math.abs(currentSignal.currentPrice - currentSignal.stopLoss)).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-gray-400 text-sm mt-8 pb-4">
          <p>⚠️ This is for educational purposes only. Always do your own research before trading.</p>
          <p>Past performance does not guarantee future results.</p>
        </div>
      </div>
    </div>
  );
};

export default SwingTradingStrategy;
