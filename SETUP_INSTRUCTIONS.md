# Python Environment Setup for Bayesian Optimization Notebook

This guide will help you set up a Python environment to run the Jupyter notebook for the Bayesian optimization competition.

## Prerequisites

- Python 3.8 or higher installed on your system
- Basic familiarity with command line/terminal

## Option 1: Using venv (Recommended for Beginners)

### Step 1: Open Terminal

Navigate to your project directory:

```bash
cd /home/robin/Personal_Development/Capstone-Project-ML-AI-Imperial-College
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
```

This creates a new virtual environment in a folder called `venv`.

### Step 3: Activate the Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 4: Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Add Kernel to Jupyter

```bash
python -m ipykernel install --user --name=bayesian-opt --display-name "Bayesian Optimization"
```

### Step 6: Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to `notebooks/bayesian_optimization.ipynb`.

### Step 7: Select the Correct Kernel

In Jupyter:
1. Click `Kernel` → `Change Kernel`
2. Select "Bayesian Optimization"

Now you're ready to run the notebook!

### Deactivating

When you're done, deactivate the environment:

```bash
deactivate
```

---

## Option 2: Using Conda (If You Have Anaconda/Miniconda)

### Step 1: Create Conda Environment

```bash
conda create -n bayesian-opt python=3.9
```

### Step 2: Activate Environment

```bash
conda activate bayesian-opt
```

### Step 3: Install Packages

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
conda install numpy scipy matplotlib scikit-learn jupyter ipykernel
```

### Step 4: Add to Jupyter

```bash
python -m ipykernel install --user --name=bayesian-opt --display-name "Bayesian Optimization"
```

### Step 5: Launch Jupyter

```bash
jupyter notebook
```

### Deactivating

```bash
conda deactivate
```

---

## Quick Start Commands (Copy-Paste)

For Linux/Mac users, here's a quick copy-paste setup:

```bash
# Navigate to project
cd /home/robin/Personal_Development/Capstone-Project-ML-AI-Imperial-College

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add kernel to Jupyter
python -m ipykernel install --user --name=bayesian-opt --display-name "Bayesian Optimization"

# Launch Jupyter
jupyter notebook
```

---

## Verifying Installation

After activating your environment, you can verify packages are installed:

```python
python -c "import numpy, scipy, matplotlib, sklearn; print('All packages installed successfully!')"
```

---

## Troubleshooting

### Issue: "jupyter: command not found"

Make sure you've activated your virtual environment and installed jupyter:

```bash
pip install jupyter
```

### Issue: Kernel doesn't appear in Jupyter

Try reinstalling the kernel:

```bash
python -m ipykernel install --user --name=bayesian-opt --display-name "Bayesian Optimization" --force
```

### Issue: Import errors in notebook

Make sure you:
1. Activated the correct environment
2. Selected the correct kernel in Jupyter (Kernel → Change Kernel)
3. Installed all requirements: `pip install -r requirements.txt`

### Issue: Can't find data files

Make sure you're running Jupyter from the project root directory, not from inside the `notebooks` folder.

---

## Next Steps

Once setup is complete:

1. Open `notebooks/bayesian_optimization.ipynb`
2. Run all cells in order (Cell → Run All)
3. Start with the example cells to familiarize yourself with the workflow
4. Use Section 5 to generate weekly queries
5. Use Section 6 to update with results when you receive them


