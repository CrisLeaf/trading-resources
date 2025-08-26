# Technical Indicators Documentation

This repository contains implementations of various technical indicators and candlestick patterns for trading analysis. Each indicator is organized by category and includes a Python implementation.

---

## 1. Advanced Indicators

### Chandelier Exit (`ch_exit.py`)
The Chandelier Exit is a volatility-based trailing stop indicator. It sets a stop-loss level below the highest high (for long positions) or above the lowest low (for short positions), adjusted by the Average True Range (ATR). It helps traders lock in profits while giving trades room to develop.

### Chop Zone (`cz.py`)
The Chop Zone indicator identifies whether the market is trending or choppy. It is derived from the Choppiness Index and uses a color-coded system to highlight periods of consolidation versus directional movement.

### Nadaraya-Watson Envelope (`nw_env.py`)
This indicator applies the Nadaraya-Watson kernel regression to price data, creating an envelope around the regression line. It can be used to detect dynamic support and resistance levels and filter out market noise.

### Nadaraya-Watson Estimator (`nw_est.py`)
The Nadaraya-Watson Estimator is a non-parametric regression method applied to time series data. It smooths the price curve using kernel-based weighting, providing traders with a clearer view of the underlying trend.

### Squeeze Momentum (`sqz_m.py`)
The Squeeze Momentum indicator detects periods of low volatility that are likely to be followed by strong breakouts. It combines Bollinger Bands and Keltner Channels to identify “squeeze” conditions and uses momentum to signal breakout direction.

### Super Trend (`st.py`)
Super Trend is a trend-following indicator that overlays on the price chart. It is calculated using the ATR and helps traders identify bullish and bearish trends by plotting stop-and-reverse levels above or below price action.

### Turtle Trading Channel (`ttc.py`)
The Turtle Trading Channel is based on the breakout strategy popularized by the Turtle Traders. It plots the highest high and lowest low over a defined period, signaling potential entries when price breaks out of these channels.

---

## 2. Candlestick Patterns

### Doji (`doji.py`)
A Doji occurs when the opening and closing prices are nearly the same, forming a cross-like candlestick. It indicates market indecision and can signal potential reversals or continuation depending on context.

### Evening Star (`es.py`)
The Evening Star is a three-candle bearish reversal pattern. It forms after an uptrend, signaling exhaustion of buying pressure and the potential start of a downtrend.

### Hammer (`hammer.py`)
The Hammer is a bullish reversal candlestick with a small body and a long lower wick. It usually appears after a decline and signals potential bottoming as buyers step in.

### Hanging Man (`hanging_man.py`)
The Hanging Man is visually similar to the Hammer but appears after an uptrend. It is a bearish reversal signal, indicating potential selling pressure despite the market closing near its highs.

### Marubozu (`marubozu.py`)
A Marubozu candlestick has no shadows, with the open equal to the high (bearish) or the low (bullish). It reflects strong conviction from buyers or sellers and signals momentum continuation.

### Morning Star (`ms.py`)
The Morning Star is a bullish reversal pattern composed of three candles: a bearish candle, a small-bodied indecision candle, and a strong bullish candle. It indicates a potential shift from downtrend to uptrend.

### Three Black Crows (`tbc.py`)
This bearish pattern consists of three consecutive long-bodied bearish candles, each closing lower than the previous. It signals strong selling pressure and the potential start of a downtrend.

### Three Outside Down (`tod.py`)
A bearish reversal pattern formed when a bullish candle is followed by a larger bearish candle that engulfs it, then confirmed by another bearish candle. It indicates a shift from buying to selling dominance.

### Three Outside Up (`tou.py`)
The bullish counterpart of Three Outside Down. It begins with a bearish candle, followed by a larger bullish candle that engulfs it, and confirmation with another bullish candle. It signals a shift to upward momentum.

### Three White Soldiers (`tws.py`)
This bullish reversal pattern features three consecutive long-bodied bullish candles, each closing higher than the last. It indicates strong and sustained buying pressure.

---

## 3. Momentum Indicators

### Relative Strength Index (`rsi.py`)
The RSI measures the speed and magnitude of price movements, oscillating between 0 and 100. Values above 70 suggest overbought conditions, while values below 30 indicate oversold conditions.

### Stochastic RSI (`stochrsi.py`)
Stochastic RSI applies the stochastic oscillator formula to RSI values instead of price. It enhances sensitivity, making it effective for identifying short-term overbought and oversold conditions.

### True Strength Index (`tsi.py`)
The TSI is a momentum oscillator that uses double-smoothed price changes to capture trend direction and strength. It helps identify overbought/oversold levels and trend reversals.

### Williams %R (`williams_r.py`)
Williams %R is a momentum oscillator that compares the closing price to the high-low range over a set period. It helps identify potential reversal zones by signaling overbought or oversold conditions.

---

## 4. Trend Indicators

### Commodity Channel Index (`cci.py`)
The CCI measures how far the price is from its statistical mean. High positive values indicate overbought conditions, while low negative values indicate oversold conditions, often signaling reversals.

### Directional Movement Index (`dmi.py`)
The DMI, along with the Average Directional Index (ADX), evaluates trend strength. It compares positive and negative directional movements to determine if bulls or bears dominate the market.

### Exponential Moving Average (`ema.py`)
EMA gives more weight to recent prices compared to the simple moving average, making it more responsive to short-term trends. It is widely used for smoothing and crossover strategies.

### Ichimoku Cloud (`ichimoku.py`)
Ichimoku Cloud is a comprehensive indicator that provides support/resistance levels, trend direction, and momentum. Its cloud component identifies areas of potential support and resistance.

### Moving Average Crossovers (`ma_crossover.py`)
This strategy uses crossovers of short-term and long-term moving averages to generate buy or sell signals. A bullish signal occurs when the short-term average crosses above the long-term average.

### Moving Average Convergence Divergence (`macd.py`)
The MACD tracks the relationship between two EMAs and includes a signal line for trade triggers. It is useful for identifying momentum, trend strength, and potential reversals.

### Parabolic SAR (`parabolic_sar.py`)
The Parabolic Stop and Reverse indicator plots trailing stop levels that adjust as trends develop. It helps traders set exit points and identify trend reversals.

### Simple Moving Average (`sma.py`)
The SMA calculates the average price over a defined period. It smooths price data and is commonly used to confirm trends or identify support and resistance.

### Weighted Moving Average (`wma.py`)
WMA assigns greater weight to more recent prices compared to SMA. This makes it more responsive to price changes while still reducing noise.

---

## 5. Volatility Indicators

### Average True Range (`atr.py`)
ATR measures market volatility by calculating the average of true ranges over a period. It does not indicate direction but helps assess risk and set stop-loss levels.

### Bollinger Bands (`bb.py`)
Bollinger Bands consist of a moving average with upper and lower bands set by standard deviations. They measure volatility and identify overbought or oversold conditions relative to the bands.

### Historic Volatility (`hist_vol.py`)
Historic volatility measures the standard deviation of past price returns. It provides insight into how much an asset’s price has fluctuated over time.

### Relative Volatility Index (`rvi.py`)
The RVI is similar to RSI but measures standard deviation of price changes instead of price levels. It indicates whether volatility is increasing or decreasing relative to recent history.

### Standard Deviation (`std_dev.py`)
This simple volatility measure calculates the dispersion of price around its mean. Higher values indicate more volatility, while lower values suggest stability.

### Ulcer Index (`ui.py`)
The Ulcer Index measures downside risk by quantifying the depth and duration of drawdowns. It is particularly useful for evaluating risk-adjusted performance.

---

## 6. Volume Indicators

### Accumulation Distribution Index (`adi.py`)
ADI combines price and volume to determine whether money is flowing into or out of an asset. It helps identify divergences between price movement and volume.

### Chaikin Money Flow (`cmf.py`)
CMF measures buying and selling pressure over a period by combining price and volume. Positive values suggest accumulation, while negative values indicate distribution.

### Money Flow Index (`mfi.py`)
The MFI is a momentum indicator that incorporates both price and volume. It oscillates between 0 and 100, signaling overbought and oversold conditions.

### On Balance Volume (`obv.py`)
OBV adds or subtracts volume based on price direction. It tracks the cumulative flow of volume and helps confirm trends or detect divergences.

### Volume Oscillator (`vo.py`)
The Volume Oscillator measures the difference between short-term and long-term volume moving averages. It helps identify changes in volume trends and potential strength behind price moves.

---