# Bayesian Optimization Competition Notebook

## Overview

This notebook provides a complete framework for the Bayesian optimization competition, allowing you to:

- Manage and visualize data for all 8 competition functions
- Train surrogate models (starting with Gaussian Process with RBF kernel)
- Use acquisition functions to select optimal next query points
- Track progress over time
- Update models with weekly results

## Quick Start

1. **Setup Environment** (first time only)
   - Follow instructions in `../SETUP_INSTRUCTIONS.md`

2. **Run the Notebook**
   - Open `bayesian_optimization.ipynb` in Jupyter
   - Run the cells in **Section 1** to load all function data
   - You'll see a summary table of all 8 functions

3. **Analyze Individual Functions**
   - Go to **Section 4: Function Analysis**
   - Find the cell under "Example: Analyze a Specific Function"
   - Modify the function_id and run to analyze different functions
   - For 2D functions (1 & 2), you'll see visualizations

4. **Generate Weekly Queries**
   - Go to **Section 5: Weekly Query Generator**
   - Run the cells to generate queries for all functions
   - The last cell formats output for easy copy-paste submission

5. **Update with Results**
   - When you receive results, go to **Section 6: Update Models with New Results**
   - Use the template cells there
   - Uncomment and fill in actual values
   - Run to add new observations

6. **Track Progress**
   - Go to **Section 7: Progress Tracking**
   - Run the cells to visualize optimization progress
   - See cumulative best values over time

## Notebook Structure

- **Section 1**: Setup and Data Loading
- **Section 2**: Surrogate Model Framework (extensible)
- **Section 3**: Acquisition Functions (UCB, EI, PI)
- **Section 4**: Function Analysis Dashboard
- **Section 5**: Weekly Query Generator
- **Section 6**: Update Models with Results
- **Section 7**: Progress Tracking

## Customization

### Change Acquisition Function

In **Section 5**, modify the `generate_weekly_queries` call:

```python
# Default: UCB with beta=2.0
weekly_queries, _ = generate_weekly_queries(acq_func='ucb', beta=2.0)

# Expected Improvement
weekly_queries, _ = generate_weekly_queries(acq_func='ei', xi=0.01)

# Probability of Improvement
weekly_queries, _ = generate_weekly_queries(acq_func='pi', xi=0.01)
```

### Adjust Surrogate Model

In **Section 4** or when calling `analyze_function`:

```python
# Change GP hyperparameters
surrogate = GPSurrogate(length_scale=0.3, optimize=True)
analyze_function(1, surrogate=surrogate)
```

### Add New Surrogate Models

The framework is designed to be extensible. To add a new surrogate model:

1. Create a class that inherits from `SurrogateModel`
2. Implement `fit()`, `predict()`, and `get_name()` methods
3. Use it anywhere you'd use `GPSurrogate`

Example:
```python
class MyNewSurrogate(SurrogateModel):
    def fit(self, X, y):
        # Your implementation
        pass
    
    def predict(self, X):
        # Return mean and std
        return mean, std
    
    def get_name(self):
        return "My New Model"
```

## All Notebooks

| Notebook | Purpose |
|----------|---------|
| **`weekly_workflow.ipynb`** | Main pipeline: load `data/results/week_*`, set per-function strategies, run acquisition, format output with `format_for_portal()`. This is the notebook used every round. |
| **`bayesian_optimization.ipynb`** | Standalone Bayesian optimisation walkthrough covering GP fitting, acquisition functions, and visualisation. Good starting point for understanding the method. |
| **`data_management.ipynb`** | Ingest new weekly `inputs.txt` / `outputs.txt` from the portal into per-function cumulative `.npy` checkpoints. Run this first after each round. |
| **`model_comparison.ipynb`** | Leave-one-out cross-validation comparing GP, GP-ARD, SVR ensemble, and MLP surrogates per function. Used to justify per-function surrogate assignment. |
| **`pca_analysis.ipynb`** | Linear PCA and kernel PCA on each function's best-point cluster. Exports focus regions to `data/results/week_12/pca_week12.json`. Introduced in Week 12. |
| **`llm_experiments.ipynb`** | Systematic LLM prompt / temperature / domain-context experiment grid (Week 8). Logs to `data/results/week_8/llm_experiments.json`. Requires `openai` and/or `anthropic` packages and API keys. |

## Weekly Workflow

### Week N: Generate Queries

1. Navigate to **Section 5: Weekly Query Generator**
2. Run the "Generate Queries for This Week" cell
3. Run the "Format Queries for Submission" cell
4. Copy the formatted output and submit via competition portal
5. Wait for results

### Week N+1: Update and Repeat

1. Receive results from previous week
2. Navigate to **Section 6: Update Models with New Results**
3. Update each function using the template:
   ```python
   update_function_with_result(func_id, x_query, y_result, week=N)
   ```
4. Go back to **Section 5** to generate new queries
5. Repeat the cycle

## Tips

- **Navigation**: Use section headings (e.g., "Section 1:", "Section 5:") to navigate the notebook
- **Running Cells**: Use `Shift + Enter` to run a cell and move to the next one
- **2D Functions**: Take advantage of visualizations to understand the landscape
- **High-D Functions**: Focus on acquisition function values and predictions
- **Exploration vs Exploitation**: 
  - Early weeks: Higher beta in UCB or use EI for exploration
  - Later weeks: Lower beta for exploitation of known good regions
- **Save Progress**: The notebook saves weekly data automatically when you update functions

## Data Files

- Initial data: `../data/function_X/initial_inputs.npy` and `initial_outputs.npy`
- Weekly updates: `../data/function_X/week_N_inputs.npy` and `week_N_outputs.npy`

## Need Help?

- Check `../SETUP_INSTRUCTIONS.md` for environment setup issues
- Read docstrings in code cells for function documentation
- Experiment with the example cells before generating real queries

Good luck with the competition!
