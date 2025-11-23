Forecasting Emerging Pandemic Risks Through Random-Walk Metropolis–Hastings Monte Carlo Simulations


## 🔗 Affiliations

- Department of Microbiology, Institute of Biomedical Sciences, University of São Paulo, Brazil
- Faculty of Science and Engineering, University of Groningen, Netherlands

---

# Forecasting Emerging Pandemic Risks

A Bayesian framework for predicting pandemic potential of emerging pathogens using Random-Walk Metropolis–Hastings Monte Carlo simulations.

## Abstract

We propose that pandemic potential arises from a quantifiable interplay of intrinsic features—transmissibility, stealth, and virulence. By integrating epidemiological parameters into a unified Bayesian framework, we developed a predictive model that distinguishes pathogens with pandemic capacity from those likely to remain contained. Our approach derives a **Novel Pandemic Potential Index (NPPI)**, transforming raw data into an actionable measure of threat.

## Key Features

- **Bayesian classification model** using Random-Walk Metropolis–Hastings algorithm
- **NPPI metric**: Composite score quantifying outbreak risk
- **Multi-parameter integration**: R₀, incubation period, infectious period, lethality, immunity
- **Probabilistic forecasting** of pathogen trajectories

## Model Framework

### Core Parameters:
- Minimum basic reproduction number (R₀min)
- Infectious period (days)
- Lethality rate (%)  
- Minimum incubation period (days)
- Permanent immunity (binary)

### NPPI Formula:

NPPI = -0.07 + 0.93×R₀min - 0.21×InfectiousPeriod - 0.23×Lethality + 0.30×IncubationMin - 0.07×PermanentImmunity

## Performance

- **Accuracy**: ~80% (training and test sets)
- **ROC AUC**: 0.85 (training), 0.83 (test)
- **Key Insight**: Longer incubation periods amplify risk, while high lethality constrains global spread

## Applications

- Early risk assessment for emerging pathogens
- Strategic allocation of surveillance resources
- Data-driven prioritization of vaccine/therapy development
- Transition from reactive to anticipatory global health security
