
# The Entry Deterrence Game

## Project Overview

This project develops an interactive dashboard to examine startup
market-entry decisions under incomplete information.

It uses a preliminary expected-payoff model, Bayesian belief updating,
scenario analysis, and interactive visualizations.

## Research Question

How can AI-assisted Bayesian modelling support startup market-entry
decisions when the incumbent's competitive behaviour is unknown?

## Dashboard Features

- Interactive model assumptions
- Expected-payoff sensitivity analysis
- Entry-cost analysis
- Entrant payoff comparisons
- Decision heatmap
- Bayesian belief-updating illustration
- Scenario comparison
- CSV downloads

## Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- Plotly

## How to Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Data and Assumptions

All payoffs and signal likelihoods in the initial version are
illustrative assumptions. They do not represent actual company data.

The initial model assumes that an aggressive incumbent fights and an
accommodating incumbent accommodates. The incumbent's type-specific
optimal responses have not yet been solved.

The displayed entry decision is a preliminary expected-payoff result,
not a solved Bayesian Nash Equilibrium or Perfect Bayesian Equilibrium.

## Responsible AI

AI tools may assist in coding and interpreting results. All model
assumptions, calculations, sources, and AI-assisted contributions
should be checked and documented.

## Academic Context

Working with AI (WAI) project
Games of Incomplete Information
Indian Institute of Management Ranchi
