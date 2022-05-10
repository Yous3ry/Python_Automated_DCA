# Python Automatic Decline Curve Analysis (DCA) For Petroleum Producing wells

## Table of Contents
1. [About the Project](#about-the-project)
2. [Workflow](#workflow)
3. [Dependencies](#dependencies)
4. [References](#references)

## About The Project
Decline Curve Analysis (DCA) is a widely accepted method for analyzing declining production rates and forecasting future performance of oil and gas wells. [[1]](#ref-1) For oil fields with hundreds or even thousands of wells updating the production forecast for each well based on DCA could be very time consuimg. Here's why:
1. Typically curve fitting to estimate the decline rate and initial rate is done graphically which could be very time consuming, and
2. Over the life of a conventional oil producing well, decline rates may change over time with changes in operating conditions, e.g. start of water injection, aritifical lift methodlogy or effciency changes, etc. Thus the history of the well could be divided into different distinct regions and DCA parameters should only be estimated based on the last stable producing conditions rather than average fitting of all the historical performance.

## Workflow
1. Connect to production database to read well data ([Python File here](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/DB_Connect.py))
2. Use R to detect change points assuming a piecewise linear trend using EnvCpt Package[[2]](#ref-2) as shown in example below ([R File here](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/Change_Detection.R))
![alt text](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/Sample_Well_1_ChangeDetection.png)
3. Finally an integrated Python file ([Final Python File here](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/Python_DCA.py)) is used to a) Load Well Data b) detect change points using R via Rpy2 c) fit exponential or Hyperbolic decline based on user input on the last stable period d) Forecast production rate. Note forecast is limited by user minimum rate and forecast period.
![alt_text](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/Sample_Well_1_Forecast_Results.png)

*NOTE* A compiled Python Script integrates all previous steps and loops through all the wells in the database and exports the total field/company forecast. ([Compiled Python File here](https://github.com/Yous3ry/Python_Automated_DCA/blob/main/Compiled_Script.py))

## Dependencies
Python Packages <- sqlalchemy, scipy, matplotlib, pandas, numpy, json, rpy2\
R Packages <- EnvCpt

<br>

## References
<a id="ref-1">[1]</a> 
https://petrowiki.spe.org/Production_forecasting_decline_curve_analysis#:~:text=Decline%20curve%20analysis%20(DCA)%20is,fluids%2C%20are%20usually%20the%20cause.
<br>
<a id="ref-1">[2]</a> 
https://rdrr.io/cran/EnvCpt/man/envcpt.html
