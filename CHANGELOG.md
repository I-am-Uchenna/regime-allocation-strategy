# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-18

### Added
- Initial release of regime-based multi-asset allocation strategy
- Hidden Markov Model implementation for VIX regime identification
- 2-state and 3-state HMM fitting with EM algorithm
- Discrete Markov chain analysis with transition matrices
- Rule-based allocation framework (SPY in low vol, TLT in high vol)
- Comprehensive backtesting engine with 1-day execution lag
- Performance metrics calculation (Sharpe, Sortino, Calmar, Max Drawdown)
- Benchmark comparisons (Equal-weight portfolio, Buy-and-hold SPY)
- Visualization suite:
  - ETF returns over time
  - VIX dynamics and regime identification
  - State-conditional performance analysis
  - Strategy performance comparison charts
- Data acquisition from Yahoo Finance (TLT, GLD, SPY, VIX)
- Log-return computation and data alignment
- Statistical analysis of regime-conditional returns
- Documentation and code comments

### Features
- Automated data download for 2004-2026 period (5,323 trading days)
- Model selection via AIC/BIC criteria
- State sorting for consistent interpretation
- Transition probability matrices
- Stationary distribution computation
- Outlier detection and data quality checks
- Rolling Sharpe ratio visualization
- Drawdown analysis
- Allocation weight tracking over time

### Performance
- Annualized return: 19.41%
- Sharpe ratio: 1.220
- Maximum drawdown: -19.54%
- Outperformance vs SPY: 8.61% annually
- Drawdown reduction vs SPY: 65%

### Technical Details
- Python 3.8+ compatibility
- Dependencies: numpy, pandas, matplotlib, yfinance, hmmlearn, scipy
- Full covariance Gaussian HMM
- Viterbi algorithm for state sequence extraction
- 1,000 EM iterations for convergence
- Random seed (42) for reproducibility

## [Unreleased]

### Planned Features
- Rolling window parameter estimation
- Transaction cost modeling
- Multi-factor regime identification
- Risk parity weighting
- Probabilistic allocation based on state probabilities
- Interactive dashboard
- Real-time regime monitoring
- Alternative model specifications

### Under Consideration
- Machine learning classification approaches
- Correlation-based regime identification
- Fundamental macro regime definitions
- Stop-loss overlays
- Position sizing optimization
- Additional asset classes (commodities, international equities)

---

## Version History

### [1.0.0] - 2025-01-18
Initial public release

---

## How to Read This Changelog

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to suggest changes to this changelog.
