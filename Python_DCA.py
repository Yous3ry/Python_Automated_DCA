import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dateutil.relativedelta import relativedelta


def get_well_data():
    # Read dample well data CSV
    data = pd.read_csv("Sample_Well.csv")
    # ensure date format
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def detect_change(well_df):
    """
    Performs change detection of historical production using R
    :inputs: Well historical production dataframe
    :return: change points list
    """
    # import rpy2 needed packages
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages

    # Define Basic R packages
    base = rpackages.importr("base")
    stats = rpackages.importr("stats")
    utils = rpackages.importr("utils")
    # install packages if not installed
    if not rpackages.isinstalled("EnvCpt"):
        utils.install_packages("EnvCpt")
    EnvCpt = rpackages.importr("EnvCpt")

    # filter dataframe to desired columns only
    well_df = well_df[["Date", "OIL_RATE"]]

    # Convert pandas to r dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(well_df)

    # convert data to time series
    oil_ts = stats.ts(data=r_dataframe.rx2("OIL_RATE"), frequency=12)

    # define models to be used to find Change points
    models = ro.StrVector(["trendar1cpt", "trendar2cpt"])
    # Find Change points
    out = EnvCpt.envcpt(oil_ts, verbose=0, model=models, minseglen=int(0.05*len(well_df)))
    # find best model
    best_model = base.names(base.which_min(stats.AIC(out)))
    # find change points
    data = out.rx2(best_model)  # find the list named as model name
    # Try to catch error if no change was found
    try:
        points = data.slots['cpts']  # find change points slot
        points = points[:-1]  # remove last point
    except LookupError:
        return [0]
    return points


def exp_decline(time, initial_rate, decline_rate):
    """
    :return: exponential decline equation
    """
    return initial_rate * np.exp(-decline_rate * time)


def hyp_decline(time, initial_rate, decline_rate, b):
    """
    :return: Hyperbolic decline equation
    """
    return initial_rate / (1 + b * decline_rate * time)**(1/b)


def exp_decline_fit(well_df, max_di_limit):
    """
    Performs regression to fit historical production
    :inputs: Well historical production dataframe
    :return: initial rate fit and decline rate
    """
    # Perform history match for exponential decline
    lower_bounds = [min(well_df["OIL_RATE"]), 0]
    upper_bounds = [max(well_df["OIL_RATE"])*1.5, max_di_limit/12]  # Maximum fit rate 1.5 max actual rate & max Di A.n.
    params, pcov = curve_fit(exp_decline, well_df["t_actual"], well_df["OIL_RATE"], bounds=(lower_bounds, upper_bounds))
    errors = np.sqrt(np.diagonal(pcov))
    qi_fit = params[0]
    di_fit = params[1]
    # print fitting results
    print("Best Fit Annual Decline rate is: {}".format(round(di_fit * 12, 2)))
    print("Best Fit initial rate is: {}".format(round(qi_fit, 2)))
    return qi_fit, di_fit, errors


def hyp_decline_fit(well_df, max_di_limit):
    """
    Performs regression to fit historical production
    :inputs: Well historical production dataframe
    :return: initial rate fit and decline rate
    """
    # Perform history match for exponential decline
    lower_bounds = [min(well_df["OIL_RATE"]), 0, 0]
    # Maximum fit rate 1.5 max actual rate & max Di A.n. & max 1 b (harmonic)
    upper_bounds = [max(well_df["OIL_RATE"])*1.5, max_di_limit/12, 1]
    params, pcov = curve_fit(hyp_decline, well_df["t_actual"], well_df["OIL_RATE"], bounds=(lower_bounds, upper_bounds))
    errors = np.sqrt(np.diagonal(pcov))
    qi_fit = params[0]
    di_fit = params[1]
    b_fit = params[2]
    # print fitting results
    print("Best Fit Annual Decline rate is: {}".format(round(di_fit * 12, 2)))
    print("Best Fit b factor is: {}".format(round(b_fit, 2)))
    print("Best Fit initial rate is: {}".format(round(qi_fit, 2)))
    return qi_fit, di_fit, b_fit, errors


def exp_decline_forecast(q, d, tmax=1000, qmin=20):
    """
    Performs forecast using exponential decline
    :inputs: initial rate fit, decline rate, minimum oil rate and maximum duration for forecast in years
    :return: initial rate fit and decline rate
    """
    # Forecast well performance
    t_forecast = np.arange(tmax)
    q_forecast = exp_decline(t_forecast, q, d)
    # use cut-off rate
    q_forecast_cut = np.where(q_forecast < qmin, 0, q_forecast)
    return q_forecast_cut


def hyp_decline_forecast(q, d, b, tmax=1000, qmin=20):
    """
    Performs forecast using exponential decline
    :inputs: initial rate fit, decline rate, minimum oil rate and maximum duration for forecast in years
    :return: initial rate fit and decline rate
    """
    # Forecast well performance
    t_forecast = np.arange(tmax)
    q_forecast = hyp_decline(t_forecast, q, d, b)
    # use cut-off rate
    q_forecast_cut = np.where(q_forecast < qmin, 0, q_forecast)
    return q_forecast_cut


# Define user forecast constraints
minimum_oil_rate = 50  # Barrel Oil per Day
forecast_months = 200
max_di = 5  # A.n.
decline_method = "Hyperbolic"  # or Exponential

# get well data
well_data = get_well_data()
change_points = detect_change(well_data)

print(well_data)
# find trend change dates
change_dates = well_data.loc[change_points, "Date"].to_list()
# filter data for decline curve analysis fit
filtered_well_data = well_data[well_data["Date"] > change_dates[-1]].copy()
# reset index
filtered_well_data.reset_index(inplace=True, drop=True)
# duration of fitting region
fitting_time = (max(filtered_well_data["Date"]) - min(filtered_well_data["Date"])).days + 1
# Create filtered_well_data time
filtered_well_data["t_actual"] = range(0, len(filtered_well_data))
# correct forecast years to include fitting period
forecast_periods = forecast_months + len(filtered_well_data)

if decline_method == "Exponential":
    # Fit exponential decline
    qi, di, stds = exp_decline_fit(filtered_well_data, max_di)
    # Forecast production
    oil_forecast = exp_decline_forecast(qi, di, forecast_periods, minimum_oil_rate)
    oil_forecast_low = exp_decline_forecast(qi-stds[0], di-stds[1], forecast_periods, minimum_oil_rate)
    oil_forecast_high = exp_decline_forecast(qi+stds[0], di+stds[1], forecast_periods, minimum_oil_rate)
elif decline_method == "Hyperbolic":
    # Fit Hyperbolic decline
    qi, di, b, stds = hyp_decline_fit(filtered_well_data, max_di)
    # Forecast production
    oil_forecast = hyp_decline_forecast(qi, di, b, forecast_periods, minimum_oil_rate)
    oil_forecast_low = hyp_decline_forecast(qi-stds[0], di-stds[1], b-stds[2], forecast_periods, minimum_oil_rate)
    oil_forecast_high = hyp_decline_forecast(qi+stds[0], di+stds[1], b+stds[2], forecast_periods, minimum_oil_rate)

# prepare forecast dates
forecast_dates = pd.date_range(min(filtered_well_data["Date"]),
                               min(filtered_well_data["Date"]) + relativedelta(months=+forecast_periods),
                               freq='M')

# create forecast Dataframe
forecast_df = pd.DataFrame()
forecast_df["Date"] = forecast_dates
forecast_df["OIL_RATE"] = oil_forecast
forecast_df["OIL_RATE_low"] = oil_forecast_low
forecast_df["OIL_RATE_high"] = oil_forecast_high

# print results
print("cumulative production MSTB: {}".format(round(np.sum(well_data["OIL_RATE"])/1000, 0)))
print("Remaining reserves MSTB: {}".format(round(np.sum(oil_forecast)/1000, 0)))
print("EUR MSTB: {}".format(round((np.sum(oil_forecast)+np.sum(well_data["OIL_RATE"]))/1000, 0)))
print("----------------------------------------------------------")

# Plot Actual data
plt.scatter(well_data["Date"], well_data["OIL_RATE"], label="Actual", color="black", facecolors='none')
# Plot Forecast
plt.plot(forecast_df["Date"], forecast_df["OIL_RATE"], color="green", label="Base Forecast", linestyle="dashed")
plt.plot(forecast_df["Date"], forecast_df["OIL_RATE_low"], color="blue", label="Low Forecast", linestyle="dashed")
plt.plot(forecast_df["Date"], forecast_df["OIL_RATE_high"], color="orange", label="High Forecast", linestyle="dashed")
# plot change points
plt.vlines(change_dates, ymin=0, ymax=1e10, color="red", label="Change points")
plt.ylim([minimum_oil_rate*0.5, max(well_data["OIL_RATE"])*1.5])
plt.legend()
plt.grid()
plt.yscale("log")
plt.ylabel("OIL_RATE")
plt.title("Forecast using {} Decline with Base Case Di = {} and b = {}"
          .format(decline_method, round(di, 2), round(b, 2)))
plt.show()

