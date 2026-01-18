# Setup Guide

Complete installation and setup instructions for the Regime-Based Multi-Asset Allocation Strategy.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum
- **Storage**: 500 MB free space
- **Internet**: Required for data download

### Recommended
- **Python**: 3.9 or 3.10
- **RAM**: 8 GB or more
- **Storage**: 1 GB free space

## Installation Methods

### Method 1: Quick Install (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/I-am-Uchenna/regime-allocation-strategy.git
cd regime-allocation-strategy
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (choose based on your OS)
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Run Strategy
```bash
python regime_allocation_strategy.py
```

### Method 2: Conda Environment

If you use Anaconda or Miniconda:

```bash
# Create conda environment
conda create -n regime-strategy python=3.9

# Activate environment
conda activate regime-strategy

# Install dependencies
pip install -r requirements.txt

# Run strategy
python regime_allocation_strategy.py
```

### Method 3: Google Colab (No Installation)

For cloud-based execution:

1. Upload `regime_allocation_strategy.py` to Google Drive
2. Open Google Colab: https://colab.research.google.com
3. Create new notebook
4. Run:
```python
!pip install yfinance hmmlearn

# Upload your script or copy-paste the code
%run regime_allocation_strategy.py
```

## Verification

### Check Python Version
```bash
python --version
# Should show: Python 3.8.x or higher
```

### Verify Installations
```bash
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import yfinance; print(f'yfinance: {yfinance.__version__}')"
python -c "import hmmlearn; print(f'hmmlearn: {hmmlearn.__version__}')"
```

Expected output (versions may vary):
```
NumPy: 1.24.x
Pandas: 2.0.x
yfinance: 0.2.x
hmmlearn: 0.3.x
```

### Test Run

Create a test script `test_install.py`:
```python
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

print("✓ All core dependencies imported successfully!")

# Quick data test
try:
    data = yf.download('SPY', start='2024-01-01', end='2024-01-31', progress=False)
    print(f"✓ Data download working! Retrieved {len(data)} rows")
except Exception as e:
    print(f"✗ Data download failed: {e}")
```

Run: `python test_install.py`

## Troubleshooting

### Common Issues

#### Issue 1: `pip` not found
**Solution:**
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

#### Issue 2: Permission errors during installation
**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

#### Issue 3: hmmlearn installation fails
**Solution:**
```bash
# Install build tools first
# Windows:
# Download Visual Studio Build Tools

# macOS:
xcode-select --install

# Linux:
sudo apt-get install python3-dev

# Then retry:
pip install hmmlearn
```

#### Issue 4: yfinance download errors
**Solution:**
```bash
# Update yfinance
pip install --upgrade yfinance

# If still failing, try alternative:
pip install yfinance==0.2.28
```

#### Issue 5: Matplotlib display issues
**Solution:**
```bash
# Install backend
pip install PyQt5

# Or use non-interactive backend
# Add to top of script:
import matplotlib
matplotlib.use('Agg')
```

#### Issue 6: Memory errors during execution
**Solution:**
- Reduce data period in script
- Close other applications
- Use 64-bit Python
- Upgrade RAM if possible

### Platform-Specific Issues

#### Windows
**Error**: `Microsoft Visual C++ 14.0 is required`
**Solution**: Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### macOS
**Error**: `SSL: CERTIFICATE_VERIFY_FAILED`
**Solution**:
```bash
# Run Python's certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command
```

#### Linux
**Error**: `ImportError: libGL.so.1`
**Solution**:
```bash
sudo apt-get install python3-tk
sudo apt-get install libgl1-mesa-glx
```

## Environment Variables (Optional)

For advanced users who want to customize paths:

```bash
# Create .env file
echo "DATA_PATH=./data" >> .env
echo "OUTPUT_PATH=./output" >> .env
echo "CACHE_PATH=./cache" >> .env
```

## Performance Optimization

### Speed Up Execution

1. **Use faster data source** (if available):
```python
# Cache downloaded data
import pandas as pd
data.to_pickle('cached_data.pkl')
# Later: data = pd.read_pickle('cached_data.pkl')
```

2. **Reduce HMM iterations** for testing:
```python
# In script, change:
hmm_model = hmm.GaussianHMM(n_components=2, n_iter=100)  # Instead of 1000
```

3. **Multi-core processing** (future enhancement):
```python
# Install joblib
pip install joblib
```

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Uninstalling

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # Linux/macOS
# or
rmdir /s venv  # Windows

# Remove repository
cd ..
rm -rf regime-allocation-strategy
```

## Next Steps

After successful installation:

1. **Run the strategy**: `python regime_allocation_strategy.py`
2. **Check output folder**: Review generated PNG files
3. **Read the code**: Understand the implementation
4. **Modify parameters**: Experiment with different settings
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Getting Help

If you encounter issues not covered here:

1. Check [existing issues](https://github.com/I-am-Uchenna/regime-allocation-strategy/issues)
2. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps you've tried
3. Include output from verification steps

## Additional Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [pip Documentation](https://pip.pypa.io/en/stable/)
- [yfinance Guide](https://pypi.org/project/yfinance/)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)

---

**Last Updated**: January 2025  
**Tested On**: Windows 11, macOS 14, Ubuntu 22.04
