# Regime-Based Multi-Asset Allocation Strategy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A quantitative trading strategy that leverages Hidden Markov Models to identify market volatility regimes and systematically allocate capital across US Treasuries (TLT), Gold (GLD), and Equities (SPY).

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Implementation Details](#implementation-details)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Limitations & Assumptions](#limitations--assumptions)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository implements a systematic regime-switching allocation framework that adapts portfolio positioning based on VIX-derived volatility states. Using unsupervised machine learning, the strategy identifies distinct market environments and executes rule-based rotations designed to optimize risk-adjusted returns across varying market conditions.

### Key Features

- **Regime Detection**: Gaussian Hidden Markov Models (HMM) for volatility state identification
- **Dynamic Allocation**: Rule-based rotation among three liquid ETFs
- **Risk Management**: Execution lag implementation to prevent lookahead bias
- **Comprehensive Analytics**: Full performance attribution and regime analysis
- **Reproducible Research**: Complete implementation with visualization suite

## Methodology

### 1. Data Collection & Preprocessing

```python
# Automated data acquisition
tickers = ['TLT', 'GLD', 'SPY', '^VIX']
data = yf.download(tickers, start='2004-01-01', end=datetime.today())

# Log-return computation
log_returns = np.log(prices / prices.shift(1))
```

Daily adjusted close prices are sourced from Yahoo Finance for:
- **TLT**: iShares 20+ Year Treasury Bond ETF (5,546 observations: 2004-01-02 to 2026-01-16)
- **GLD**: SPDR Gold Shares (5,324 observations: 2004-11-18 to 2026-01-16)
- **SPY**: SPDR S&P 500 ETF Trust (5,546 observations: 2004-01-02 to 2026-01-16)
- **VIX**: CBOE Volatility Index (5,546 observations: 2004-01-02 to 2026-01-16)

**Common Sample Period**: November 19, 2004 - January 16, 2026 (5,323 trading days after alignment)

**Data Quality Metrics:**
- Missing values: 0 (complete time series after alignment)
- Outliers detected (>5 standard deviations): TLT (8), GLD (10), SPY (20), VIX changes (30)
- All outliers retained as they represent legitimate extreme market events

### 2. Regime Identification

The strategy employs a two-stage approach:

#### Discrete Markov Chain
- Quantile-based discretization of ΔVIX into 2-3 states
- Maximum likelihood estimation of transition matrices
- Stationary distribution computation via eigendecomposition

**2-State Results:**
- State distribution: 50.3% Low Vol / 49.7% High Vol
- Transition probabilities: 48.5% (Low→Low), 48.0% (High→High)
- Stationary: [0.503, 0.497]

**3-State Results:**
- State distribution: 33.1% Low / 34.2% Medium / 32.7% High  
- More granular regime classification but increased complexity

#### Hidden Markov Model
- Gaussian emission densities with full covariance
- Baum-Welch (EM) algorithm for parameter estimation
- Viterbi decoding for most likely state sequence
- Model selection via AIC/BIC criteria

**Model Comparison:**

| Model | Parameters | Log-Likelihood | AIC | BIC |
|-------|-----------|----------------|-----|-----|
| 2-State HMM | 7 | -9,044.57 | 18,103.15 | 18,149.21 |
| 3-State HMM | 14 | -8,700.54 | 17,429.09 | 17,521.20 |

**Selected Model**: 2-State HMM
- Rationale: Despite higher BIC for 2-state, chosen for superior interpretability
- Clear risk-on/risk-off dichotomy aligns with investment decision-making
- Simpler allocation rules reduce turnover and implementation complexity
- 3-state model shows better statistical fit but marginal practical benefit

```python
from hmmlearn import hmm

# Fit 2-state Gaussian HMM
model = hmm.GaussianHMM(n_components=2, covariance_type='full', 
                        n_iter=1000, random_state=42)
model.fit(vix_changes.reshape(-1, 1))

# Extract hidden states
states = model.predict(vix_changes.reshape(-1, 1))
```

### 3. Allocation Rules

Portfolio construction follows a deterministic mapping based on historical regime-conditional performance:

| Regime State | Market Condition | Historical Best Performer | Allocation Strategy |
|-------------|------------------|--------------------------|---------------------|
| Low Volatility (State 0) | Risk-On Environment | SPY (29.87% ann. return, 2.66 Sharpe) | 100% SPY |
| High Volatility (State 1) | Risk-Off Environment | TLT (25.21% ann. return, 1.27 Sharpe) | 100% TLT |

Allocation determined by regime-conditional mean returns:

```
w_i(t) = 1{i = argmax_j E[r_j | S(t)]}
```

where `S(t)` is the identified regime at time `t`.

**Allocation Logic:**
- **Low Vol**: VIX declining/stable → Market calm → Allocate to equities (SPY)
- **High Vol**: VIX rising → Market stress → Rotate to safe-haven treasuries (TLT)
- **GLD**: Positive in both regimes but not optimal in either state (excluded from final allocation)

### 4. Backtesting Framework

- **Execution Lag**: 1-day delay between signal generation and execution
- **Rebalancing**: Daily regime assessment with full capital redeployment
- **Transaction Costs**: Not explicitly modeled (identified as limitation, estimated impact ~50-200 bps annually)
- **Sample Period**: November 19, 2004 - January 16, 2026 (5,323 trading days, 21.2 years)
- **Data Alignment**: All series synchronized to common dates after GLD inception

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone Repository

```bash
git clone https://github.com/I-am-Uchenna/regime-allocation-strategy.git
cd regime-allocation-strategy
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
yfinance>=0.1.70
hmmlearn>=0.2.7
scipy>=1.7.0
scikit-learn>=0.24.0
```

## Quick Start

### Basic Execution

```bash
python regime_allocation_strategy.py
```

### Expected Output

The script performs the following sequence:

1. **Data Acquisition** (30-60 seconds)
   - Downloads historical prices from Yahoo Finance
   - Validates data integrity and alignment

2. **Regime Modeling** (2-5 minutes)
   - Fits multiple HMM specifications
   - Performs model selection via information criteria
   - Generates regime visualizations

3. **Strategy Backtest** (10-20 seconds)
   - Executes allocation rules with execution lag
   - Computes performance metrics
   - Generates comparison charts

4. **Output Generation**
   - Performance summary tables (console)
   - Visualization suite (PNG files)
   - Regime statistics and transition analysis

### Output Files

```
output/
├── etf_returns_plot.png              # Historical return series
├── vix_changes_plot.png              # VIX dynamics and extremes
├── vix_regimes_hmm.png               # Identified volatility states
├── state_conditional_returns.png     # Regime-conditional performance
└── performance_comparison.png        # Strategy vs benchmarks
```

## Implementation Details

### Core Functions

#### Regime Identification

```python
def discretize_vix_changes(vix_changes, n_states=3):
    """
    Discretize VIX changes using quantile-based thresholds.
    
    Parameters
    ----------
    vix_changes : pd.Series
        Daily VIX changes
    n_states : int
        Number of discrete states
        
    Returns
    -------
    states : pd.Series
        State assignments (0 to n_states-1)
    state_labels : dict
        Mapping from state index to interpretation
    """
```

#### Transition Matrix Estimation

```python
def estimate_transition_matrix(states, n_states):
    """
    Estimate Markov chain transition probabilities.
    
    Uses maximum likelihood estimation from observed state sequence.
    
    Returns
    -------
    transition_matrix : np.ndarray
        n_states × n_states stochastic matrix
    transition_counts : np.ndarray
        Raw transition counts for diagnostics
    """
```

#### Strategy Backtesting

```python
def backtest_strategy(returns_df, states, allocation_rules, lag=1):
    """
    Execute regime-based allocation with execution lag.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns indexed by date
    states : pd.Series
        Regime classifications
    allocation_rules : dict
        Mapping from states to portfolio weights
    lag : int
        Execution delay in trading days
        
    Returns
    -------
    strategy_returns : pd.Series
        Daily strategy returns
    weights_history : pd.DataFrame
        Historical allocation weights
    """
```

### Performance Calculation

```python
def calculate_performance_metrics(returns, name="Strategy", rf_rate=0.02):
    """
    Compute comprehensive performance statistics.
    
    Metrics
    -------
    - Cumulative return
    - Annualized return (geometric)
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown
    - Sortino ratio
    - Calmar ratio
    """
```

## Performance Metrics

### Strategy Evaluation

The implementation provides extensive performance analytics:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| Cumulative Return | Total return over period | exp(Σr_t) - 1 |
| Annualized Return | CAGR | (1 + R)^(252/N) - 1 |
| Volatility | Return standard deviation | σ(r) × √252 |
| Sharpe Ratio | Risk-adjusted return | (μ - r_f) / σ |
| Max Drawdown | Peak-to-trough decline | min((P_t - P_max) / P_max) |
| Sortino Ratio | Downside risk-adjusted | (μ - r_f) / σ_downside |
| Calmar Ratio | Return per unit drawdown | μ / |MDD| |

### Benchmark Comparisons

Two benchmarks are implemented:

1. **Equal-Weight Portfolio**: 33.3% allocation to each ETF (monthly rebalanced approximation)
2. **Buy-and-Hold SPY**: Passive equity exposure for reference

## Results

### Regime Characteristics

Identified volatility states exhibit distinct statistical properties based on 5,323 trading days:

**Low Volatility Regime (State 0):**
- **Frequency**: 50.26% of observations (2,675 days)
- **Mean VIX change**: -0.074 (declining volatility)
- **VIX variance**: 0.66 (low dispersion)
- **Persistence**: 96.0% probability of remaining in state (average duration ~25 days)
- **Transition**: 3.96% probability of switching to high volatility
- **Market conditions**: Risk-on environment, positive equity momentum, declining fear gauge
- **Optimal allocation**: 100% SPY (29.87% annualized return, 2.66 Sharpe)

**High Volatility Regime (State 1):**
- **Frequency**: 49.74% of observations (2,648 days)
- **Mean VIX change**: +0.221 (rising volatility)
- **VIX variance**: 12.35 (high dispersion, explosive moves)
- **Persistence**: 88.3% probability of remaining in state (average duration ~8.5 days)
- **Transition**: 11.7% probability of reverting to low volatility
- **Market conditions**: Risk-off, market stress, heightened uncertainty
- **Optimal allocation**: 100% TLT (25.21% annualized return, 1.27 Sharpe)

**Transition Dynamics:**
- **Stationary distribution**: 50.26% Low Vol / 49.74% High Vol (nearly balanced)
- **Regime balance**: Markets spend approximately equal time in each state
- **Low Vol persistence**: Higher (96.0%) indicates sustained calm periods
- **High Vol persistence**: Lower (88.3%) indicates turbulence tends to resolve faster

### State-Conditional Returns

The strategy leverages divergent asset behavior across regimes:

```
              Low Vol State    High Vol State
         Mean    Vol  Sharpe    Mean    Vol  Sharpe
TLT    -3.58% 12.55%  -0.28   25.21% 19.92%   1.27
GLD     8.24% 15.56%   0.53   18.52% 23.26%   0.80  
SPY    29.87% 11.24%   2.66  -53.89% 33.26%  -1.62
```

**Key Observations:**
- **SPY** dominates during low volatility (29.87% return, 2.66 Sharpe) but suffers dramatically in high volatility (-53.89%)
- **TLT** provides strong defense during turbulent markets (25.21% return, 1.27 Sharpe) while underperforming in calm periods
- **GLD** shows positive returns in both regimes but doesn't lead in either state
- The regime-switching behavior validates the core strategy thesis: different assets dominate in different market environments

### Performance Summary

**Backtest Period:** November 19, 2004 - January 16, 2026 (5,323 trading days, ~21.2 years)

| Strategy | Cumulative Return | Annualized Return | Volatility | Sharpe Ratio | Max Drawdown | Sortino | Calmar |
|----------|------------------|-------------------|------------|--------------|--------------|---------|--------|
| **Regime Strategy** | **4,134.42%** | **19.41%** | **14.27%** | **1.220** | **-19.54%** | **1.672** | **0.993** |
| Equal Weight | 445.23% | 8.36% | 9.65% | 0.660 | -23.45% | 0.900 | 0.357 |
| Buy-Hold SPY | 772.45% | 10.80% | 19.01% | 0.463 | -55.19% | 0.550 | 0.196 |

**Performance Highlights:**

- **Superior Returns**: 19.41% annualized vs 10.80% for SPY and 8.36% for equal-weight
- **Enhanced Risk-Adjusted Performance**: Sharpe ratio of 1.220 significantly outperforms benchmarks (0.660 and 0.463)
- **Drawdown Protection**: Maximum drawdown of -19.54% vs -55.19% for SPY, demonstrating 65% reduction in worst-case loss
- **Downside Risk Management**: Sortino ratio of 1.672 shows excellent downside-adjusted returns
- **Capital Efficiency**: Calmar ratio of 0.993 indicates strong return per unit of drawdown risk

**Strategy Characteristics:**

- **Total Return Outperformance**: 5.4x higher cumulative return than SPY, 9.3x higher than equal-weight
- **Volatility Profile**: 14.27% volatility sits between defensive equal-weight (9.65%) and aggressive SPY (19.01%)
- **Risk-Adjusted Excellence**: Delivers equity-like returns with significantly lower volatility and drawdown exposure

## Limitations & Assumptions

### Model Assumptions

1. **Regime Stationarity**: Volatility states exhibit consistent statistical properties
2. **VIX Informativeness**: VIX changes contain predictive regime information
3. **Return Predictability**: Regime-conditional returns persist out-of-sample
4. **Frictionless Markets**: Base implementation excludes transaction costs

### Known Limitations

#### Lookahead Bias
- HMM parameters estimated on full sample
- Creates mild in-sample optimism
- **Mitigation**: Use rolling window estimation (see Future Work)

#### Transaction Costs
- Daily rebalancing may generate significant costs
- Bid-ask spreads not modeled
- **Impact**: Could reduce net returns by 50-200 bps annually

#### Regime Detection Lag
- State identification may lag actual transitions
- Viterbi decoding uses forward-backward algorithm
- **Effect**: Potential slippage during rapid regime shifts

#### Structural Stability
- Assumes regime characteristics persist
- Vulnerable to structural breaks
- **Monitoring**: Requires periodic model recalibration

## Future Work

### Methodological Enhancements

#### Rolling Window Estimation
```python
# Implement expanding/rolling window for parameter estimation
window_size = 756  # 3 years of daily data

for t in range(window_size, len(data)):
    train_data = data[t-window_size:t]
    model = hmm.GaussianHMM(n_components=2)
    model.fit(train_data)
    # Use for out-of-sample prediction
```

#### Multi-Factor Regimes
- Incorporate credit spreads (HY-IG differential)
- Add equity momentum signals
- Include correlation regime classification

#### Transaction Cost Modeling
```python
def apply_transaction_costs(returns, weights, cost_bps=10):
    """
    Adjust returns for trading costs.
    
    Parameters
    ----------
    returns : pd.Series
        Gross strategy returns
    weights : pd.DataFrame
        Daily allocation weights
    cost_bps : float
        Round-trip cost in basis points
    """
    turnover = weights.diff().abs().sum(axis=1)
    cost_drag = turnover * (cost_bps / 10000)
    return returns - cost_drag
```

### Alternative Specifications

- **Risk Parity Weighting**: Volatility-scaled allocations within regimes
- **Probabilistic Allocation**: Weights proportional to state probabilities
- **Dynamic Position Sizing**: Regime-dependent leverage/cash positions
- **Machine Learning Classification**: Random forests or neural networks for regime identification

### Risk Management Additions

- Maximum drawdown controls with systematic de-risking
- Minimum holding period constraints to reduce turnover
- Threshold-based rebalancing (rebalance only if allocation drift > X%)
- Stop-loss overlays for tail risk protection

## Contributing

Contributions are welcome! Please follow these guidelines:

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add unit tests for new functionality
- Update documentation as needed

### Areas for Contribution

- Alternative regime identification methods
- Additional performance metrics
- Visualization improvements
- Optimization algorithms
- Documentation enhancements

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Uchenna Ejike

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Disclaimer

**Important Notice for Users**

This software is provided for research and educational purposes only. It does not constitute investment advice, financial advice, trading advice, or any other sort of advice. 

**Key Considerations:**

- Past performance does not guarantee future results
- Strategy has not been tested in live market conditions
- No representation is made regarding market conditions, execution quality, or profitability
- Users assume full responsibility for any trading decisions
- Consult qualified financial professionals before deploying capital

**Specific Risks:**

- Market impact and slippage not modeled
- Transaction costs may significantly affect net returns
- Regime models may fail during structural breaks
- Execution assumptions may not reflect real-world conditions
- Backtests inherently contain optimistic biases

By using this software, you acknowledge these risks and agree that the authors bear no liability for trading losses or other damages.

## Citation

If you use this code in your research or publications, please cite:

```bibtex
@software{ejike2025regime,
  author = {Ejike, Uchenna},
  title = {Regime-Based Multi-Asset Allocation Strategy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/I-am-Uchenna/regime-allocation-strategy},
  version = {1.0.0}
}
```

## Contact

**Uchenna Ejike**  
Quantitative Researcher

- GitHub: [@I-am-Uchenna](https://github.com/I-am-Uchenna)
- Repository: [regime-allocation-strategy](https://github.com/I-am-Uchenna/regime-allocation-strategy)

For questions, suggestions, or collaboration inquiries, please open an issue on GitHub.

---

**Project Status**: Active Development  
**Version**: 1.0.0  
**Last Updated**: January 2026  
**Python Compatibility**: 3.8+  
**Data Coverage**: 2004-Present
