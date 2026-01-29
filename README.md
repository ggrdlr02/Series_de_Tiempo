# Series_de_Tiempo
This repository contains materials (in Spanish) for a practical course on time series analysis, model selection using likelihood and information criteria, and volatility modeling for financial series using ARIMA/SARIMA and GARCH‑type models.

Note: All notebooks, slides, and explanations are in Spanish; this README is in English only.

1. Introduction to time series
Goal: build intuition about time series structure, typical plots, and simple baselines before using formal models.

Topics:

Time series definition: discrete observations indexed in time, univariate vs multivariate.
​

Components: trend, seasonality, cycles, irregular component; visual decomposition examples.
​

Stationarity concept (weak stationarity, mean and variance stability) and why most models assume it.
​

Basic plots and diagnostics: line plots, rolling mean/variance, simple transformations (log, differencing).
​

Simple forecasting baselines: naive, seasonal naive, moving average.
​

Deliverables (code in Python):

Example notebooks using pandas, matplotlib, and statsmodels.tsa to load, visualize, and decompose real series.

Simple baseline forecasts and error metrics (MAE, RMSE, MAPE).
​

2. Likelihood, AIC/BIC and ARIMA/SARIMA
Goal: fit ARIMA/SARIMA models, understand maximum log‑likelihood, and use information criteria (AIC, BIC) to select orders.

Topics:

Log‑likelihood for time‑series models: idea of maximizing the probability of observed data under the model.
​

Akaike Information Criterion (AIC), corrected AIC (AICc), Bayesian Information Criterion (BIC): formulas, interpretation as trade‑off between fit and complexity, “smaller is better”.

ARIMA 
(p,d,q): autoregressive part, differencing order, moving‑average part; conditions for stationarity and invertibility.

Seasonal ARIMA 
(p,d,q)×(P,D,Q) 
s: seasonal autoregressive and moving‑average terms, seasonal differencing, choice of season length 

ACF and PACF: definitions, interpretation for AR vs MA identification, how differencing changes these plots.
​

Model diagnostics: residual ACF, Ljung–Box tests, normality and heteroskedasticity checks.
​

Deliverables (code in Python):

Notebooks using statsmodels.tsa.SARIMAX for ARIMA/SARIMA fitting and forecasting.

Grid search or stepwise selection of 
(p,d,q,(P,D,Q) s) by AIC/AICc/BIC.
​

Examples of ACF/PACF plots and interpretation for model order selection.
​

Scripts computing and comparing log‑likelihood, AIC, BIC across candidate models.
​

3. Robust processes for volatility control
Goal: model time‑varying volatility (heteroskedasticity) in financial time series with GARCH‑type models and understand stylized facts like volatility clustering and leptokurtosis.

Topics:

Financial time series characteristics: heavy tails, volatility clustering, leverage effects.

Heteroskedasticity: conditional variance depending on past shocks; why this breaks constant‑variance assumptions in ARIMA.
​

ARCH and GARCH models: formulation of conditional variance, interpretation of parameters, persistence of volatility.
​

EGARCH: modeling log‑variance, asymmetric response to positive/negative shocks, automatic positivity of variance.
​

Other GARCH variants (GJR‑GARCH, TGARCH, etc.) and when to consider them.

Leptokurtic returns: modeling fat tails via Student‑t or other distributions instead of Gaussian.

Joint mean–variance modeling: ARIMA/SARIMA for the mean plus GARCH‑type models for the conditional variance.
​

Deliverables (code in Python):

Notebooks using a GARCH library (e.g., arch in Python) to fit GARCH and EGARCH models to real financial returns.
​

Diagnostics for volatility models: standardized residuals, Q‑statistics, tests for remaining ARCH effects.
​

Comparison of normal vs heavy‑tailed innovations for leptokurtic returns.

Tools and libraries
The course assumes intermediate Python and basic statistics.

Main libraries:

pandas and numpy for data handling.
​

matplotlib / seaborn for plotting.
​

statsmodels (tsa module, especially SARIMAX, ACF/PACF, and unit‑root tests).

arch (or similar) for GARCH/EGARCH and volatility models.
​

Optional:

scikit-learn for baseline regressors or feature‑based models.
​

prophet as an alternative for strong trend/seasonality when interpretability is more important than strict likelihood‑based modeling.
​

Structure of the repository
Suggested layout:

notebooks/ – Jupyter notebooks per module (intro, ARIMA/SARIMA, GARCH/EGARCH).

data/ – Sample datasets (macro series, stock indexes, FX rates).

src/ – Reusable Python modules (preprocessing, diagnostics, model selection).

figures/ – Exported plots for documentation.

README.md – This file (English).

LICENSE – License for the course material.

