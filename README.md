# AI-Driven Creative Destruction and Labor Market Modeling

Computational modeling project analyzing how artificial intelligence–driven innovation affects employment, wages, and productivity through a creative destruction framework.

The project implements and evaluates dynamic economic models before and after AI adoption using differential equations, parameter estimation, identifiability analysis, and sensitivity analysis.

## Overview

We model labor market dynamics under technological change by comparing:
- A **Pre-AI economy**, capturing baseline innovation and labor dynamics
- A **Post-AI economy**, incorporating AI-induced productivity growth

The goal is to understand how innovation shocks alter employment trajectories and wage dynamics over time.

## Methods

- Dynamic system modeling using differential equations
- Parameter estimation from empirical datasets
- Structural identifiability analysis to assess parameter recoverability
- Sensitivity analysis to evaluate model robustness under perturbations

## Repository Structure

- `pre_ai_model/`: Baseline labor market model and data before AI adoption
- `post_ai_model/`: AI-augmented model incorporating innovation-driven productivity growth
- `identifiability_analysis/`: Analysis of structural identifiability of model parameters
- `sensitivity_analysis/`: Sensitivity analysis of key parameters
- `model_datasets/`: Raw and processed datasets
- `results/`: Outputs and figures from analyses

## Key Results

- Identified parameters that are structurally identifiable under observed data constraints
- Demonstrated sensitivity of employment and wage dynamics to innovation intensity
- Highlighted regimes where AI adoption accelerates productivity without proportional labor displacement

## Technologies Used

- Python
- NumPy, SciPy, Pandas
- Differential equation solvers
- Data visualization and analysis tooling

## What I Learned

- Translating economic theory into executable computational models
- Designing experiments to validate model assumptions
- Evaluating robustness and identifiability in dynamic systems
- Communicating complex quantitative results through interpretable outputs
