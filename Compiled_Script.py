import json
import pandas as pd
import numpy as np
from sqlalchemy.engine import URL
from sqlalchemy.engine import create_engine
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from dateutil.relativedelta import relativedelta
import warnings
from itertools import groupby


# Used to be able to capture optimizer warnings
warnings.simplefilter("error", OptimizeWarning)


def conn_DB(login):
    """
    Connects to production Database
    Input: login credentials dictionary
    :return: Connection engine
    """
    # Define Connection
    connection_string = '''DRIVER=SQL SERVER;SERVER={}; 
                        DATABASE={};
                        UID={};PWD={};
                        Trusted_Connection=no;'''.format(login["SERVER"], login["DB"], login["UID"], login["PWD"])
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    engine = create_engine(connection_url)
    return engine


def read_prod_data(connection, table, target_well):
    """
    Reads data from production database
    :input: database connection engine, table name in DB, target well name
    :return: Production & injection df and Well locations & types df
    """
    # read SQL as dataframe
    df = pd.read_sql('''SELECT START_DATETIME, SUM("ALLOCATED OIL") AS ALLOCATED_OIL 
    FROM {}
    WHERE lower(HEADERID) LIKE lower('{}%')
    Group by START_DATETIME;'''.format(table, target_well), connection)
    # prepare data
    # Converts date to end of month
    df["Date"] = pd.to_datetime(df["START_DATETIME"]) + pd.offsets.MonthEnd(0)
    # Converts Oil Volume to oil rate
    df["OIL_RATE"] = df["ALLOCATED_OIL"] / df["Date"].dt.day
    # ensure data is sorted by date
    df.sort_values(by="Date", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def get_operational_wells(connection, level, field):
    """
    Reads data from production database
    :input: database connection engine
    :return: list of wells that have been operational last 30 days
    """
    # read SQL as dataframe
    df = pd.read_sql('''SELECT DISTINCT(ITEM_NAME)
        FROM VT_TOTALS_DAY_en_US
        WHERE START_DATETIME >= (SELECT DATEADD(DAY, -30, MAX(START_DATETIME)) FROM VT_TOTALS_DAY_en_US)
        AND ACT_OIL_VOL > 0 AND ITEM_TYPE = '{}' AND ITEM_NAME LIKE '{}%';'''.format(level, field), connection)
    df.sort_values(by="ITEM_NAME", inplace=True)
    return df["ITEM_NAME"].to_list()


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
    from rpy2.rinterface_lib.embedded import RRuntimeError

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
    if int(0.05 * len(well_df)) >= 12:  # to avoid too small step sizes if dataset is very small
        try:
            out = EnvCpt.envcpt(oil_ts, verbose=0, model=models, minseglen=int(0.05 * len(well_df)))
        except RRuntimeError:
            out = EnvCpt.envcpt(oil_ts, verbose=0, model=models)
    else:
        out = EnvCpt.envcpt(oil_ts, verbose=0, model=models)
    # find best model
    best_model = base.names(base.which_min(stats.AIC(out)))
    # corrects if Best model is missing
    if type(best_model) is not str:
        best_model = "trendar1cpt"
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
    return initial_rate / (1 + b * decline_rate * time) ** (1 / b)


def exp_decline_fit(well_df, max_di_limit):
    """
    Performs regression to fit historical production
    :inputs: Well historical production dataframe
    :return: initial rate fit and decline rate
    """
    # Perform history match for exponential decline
    lower_bounds = [max(well_df["OIL_RATE"]), 0]  # Maximum oil rate as minimum for fitting
    upper_bounds = [max(well_df["OIL_RATE"]) * 1.5,
                    max_di_limit / 12]  # Maximum fit rate 1.5 max actual rate & max Di A.n.
    try:
        params, pcov = curve_fit(exp_decline, well_df["t_actual"], well_df["OIL_RATE"],
                                 bounds=(lower_bounds, upper_bounds))
    except OptimizeWarning:
        return "error"
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
    lower_bounds = [max(well_df["OIL_RATE"]), 0, 0]  # Maximum oil rate as minimum for fitting
    # Maximum fit rate 1.5 max actual rate & max Di A.n. & max 1 b (harmonic)
    upper_bounds = [max(well_df["OIL_RATE"]) * 1.5, max_di_limit / 12, 1]
    try:
        params, pcov = curve_fit(hyp_decline, well_df["t_actual"], well_df["OIL_RATE"],
                                 bounds=(lower_bounds, upper_bounds))
    except OptimizeWarning:
        return "error"
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
prod_table = "KPC_OFM_MTH_PRD"
target_field = "TUT"  # append empty string for all wells for a specific well, use well name
minimum_oil_rate = 10  # Barrel Oil per Day
forecast_months = 120
max_di = 5  # A.n.
forecast_level = "COMPLETION"  # ZONE or COMPLETION (well level)
decline_method = "Hyperbolic"  # Hyperbolic or Exponential
plot_results = True

# reading log in credentials from text file
with open("D:\\Logins.txt") as file:
    login_data = json.loads(file.read())  # Data stored as a JSON format (Dictionary)
# create connection
conn = conn_DB(login_data)

# get operational wells
wells = get_operational_wells(conn, forecast_level, target_field)
print("Number of operational wells in DB {}".format(len(wells)))

# Dataframes to store results
well_summary = pd.DataFrame(columns=["well", "di", "b", "oil_cum", "Rem_reserves", "EUR"])
all_wells_forecast_df = pd.DataFrame()
all_wells_actual_df = pd.DataFrame()

# loop through wells
for n, well in enumerate(wells):
    print(n, well)
    # returns desired well production dataframe
    well_data = read_prod_data(conn, prod_table, well)
    # Next if well history is empty
    if len(well_data) == 0 or sum(well_data["OIL_RATE"]) == 0:
        print("Insufficient history for {}".format(well))
        well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                0, np.nan, np.nan]
        continue
    # remove first rows if prod is zero
    if well_data["OIL_RATE"][0] == 0:
        first_prod_idx = np.where(well_data["OIL_RATE"] > 0)[0][0]
        well_data = well_data[first_prod_idx:]
        well_data.reset_index(inplace=True, drop=True)
    # Next well if well history < 12 months
    if len(well_data) < 12:
        print("Insufficient history for {}".format(well))
        well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0),
                                                np.nan, np.nan]
        continue
    # find change points
    change_points = list(detect_change(well_data))
    # find trend change dates
    change_dates = well_data.loc[change_points, "Date"].to_list()
    # filter data for decline curve analysis fit
    if len(change_dates) > 0:  # corrects for if no change occurred
        filtered_well_data = well_data[well_data["Date"] > change_dates[-1]].copy()
    else:
        filtered_well_data = well_data.copy()

    # EnvCpt has a problem with consecutive 0s
    # finding last consecutive zero values and appending it to zero points and updated filtered well data
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(well_data["OIL_RATE"])]
    cursor = 0
    result = []
    for k, l in groups:
        if int(k) == 0 and l >= 3:
            result.append([cursor, cursor + l - 1])
        cursor += l
    if len(result) > 0 and result[-1][1] not in change_points:
        if result[-1][1] not in change_points:
            change_points.append(result[-1][0])
            change_points.append(result[-1][1])
            change_dates = well_data.loc[change_points, "Date"].to_list()
            filtered_well_data = well_data[well_data["Date"] > change_dates[-1]].copy()
            filtered_well_data.reset_index(inplace=True, drop=True)

    # correct for too short last filtered data
    if len(filtered_well_data) < 4:
        if len(change_dates) > 1:
            filtered_well_data = well_data[well_data["Date"] > change_dates[-2]].copy()
        else:
            filtered_well_data = well_data.copy()
            change_dates = []
    # reset index
    filtered_well_data.reset_index(inplace=True, drop=True)
    # stop if no production in filtered well data
    if sum(filtered_well_data["OIL_RATE"]) == 0:
        print("Insufficient history for {}".format(well))
        well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0),
                                                np.nan, np.nan]
        continue
    # remove first rows if prod is zero
    if filtered_well_data["OIL_RATE"][0] == 0:
        first_prod_idx = np.where(filtered_well_data["OIL_RATE"] > 0)[0][0]
        filtered_well_data = filtered_well_data[first_prod_idx:]
        filtered_well_data.reset_index(inplace=True, drop=True)
    # removing leading any points < maximum oil rate
    max_rate_idx = np.where(filtered_well_data["OIL_RATE"] == max(filtered_well_data["OIL_RATE"]))[0][0]
    if max_rate_idx != 0:
        max_rate_idx += -1
    filtered_well_data = filtered_well_data[max_rate_idx:]
    filtered_well_data.reset_index(inplace=True, drop=True)
    # Create filtered_well_data time
    filtered_well_data["t_actual"] = range(0, len(filtered_well_data))
    # correct forecast years to include fitting period
    forecast_periods = forecast_months + len(filtered_well_data)

    # initialize parameters
    qi, di, b, stds = 0, 0, 0, 0

    # Fork based on user input
    if decline_method == "Exponential":
        # Fit exponential decline
        if exp_decline_fit(filtered_well_data, max_di) != "error":
            qi, di, stds = exp_decline_fit(filtered_well_data, max_di)
        else:
            print("Poor decline trend for well {}".format(well))
            well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                    round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0),
                                                    np.nan, np.nan]
            continue
        # Forecast production
        oil_forecast = exp_decline_forecast(qi, di, forecast_periods, minimum_oil_rate)
    elif decline_method == "Hyperbolic":
        # Fit Hyperbolic decline
        if hyp_decline_fit(filtered_well_data, max_di) != "error":
            qi, di, b, stds = hyp_decline_fit(filtered_well_data, max_di)
        else:
            print("Poor decline trend for well {}".format(well))
            well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                    round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0),
                                                    np.nan, np.nan]
            continue
        # Forecast production
        oil_forecast = hyp_decline_forecast(qi, di, b, forecast_periods, minimum_oil_rate)

    # Stop well if di == 0
    if di == 0:
        print("No Decline trend for {}".format(well))
        well_summary.loc[len(well_summary),] = [well, np.nan, np.nan,
                                                round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0),
                                                np.nan, np.nan]
        continue

    # prepare forecast dates
    forecast_dates = pd.date_range(min(filtered_well_data["Date"]),
                                   min(filtered_well_data["Date"]) + relativedelta(months=+forecast_periods),
                                   freq='M')
    # correct forecast dates if error
    if len(forecast_dates) != len(oil_forecast):
        forecast_dates = forecast_dates[:-1]
    # create forecast Dataframe
    forecast_df = pd.DataFrame()
    forecast_df["Date"] = forecast_dates
    forecast_df["OIL_RATE"] = oil_forecast

    # print results
    Cumulative = round(np.sum(well_data["OIL_RATE"]) / 1000 * 30.4, 0)
    Reserves = round(np.sum(oil_forecast[forecast_dates > max(filtered_well_data["Date"])]) / 1000 * 30.4, 0)
    EUR = Cumulative + Reserves
    print("cumulative production MSTB: {}".format(Cumulative))
    print("Remaining reserves MSTB: {}".format(Reserves))
    print("EUR MSTB: {}".format(EUR))
    print("----------------------------------------------------------")

    if plot_results:
        # Plot Actual data
        plt.scatter(well_data["Date"], well_data["OIL_RATE"], label="Actual")
        # Plot Forecast
        plt.plot(forecast_df["Date"], forecast_df["OIL_RATE"], color="black", label="Forecast", linestyle="dashed")
        # plot change points
        plt.vlines(change_dates, ymin=0, ymax=1e10, color="red", label="Change points")
        plt.ylim([minimum_oil_rate * 0.5, max(well_data["OIL_RATE"]) * 1.5])
        plt.hlines(minimum_oil_rate, xmin=min(well_data["Date"]), xmax=max(forecast_df["Date"]),
                   linestyles="dashed", color="green", label="Min Oil Rate")
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.ylabel("OIL_RATE")
        if decline_method == "Hyperbolic":
            plt.title("Forecast using {} Decline with Base Case Di = {} A.n. and b = {} for {}"
                      .format(decline_method, round(di * 12, 2), round(b, 2), well))
        else:
            plt.title("Forecast using {} Decline with Base Case Di = {} A.n. for {}"
                      .format(decline_method, round(di * 12, 2), well))
        plt.show()

    # Add data to wells dataframe
    well_summary.loc[len(well_summary),] = [well, di, b, Cumulative, Reserves, EUR]
    # prepare well forecast dataframe
    well_forecast_Df = pd.DataFrame()
    well_forecast_Df["Date"] = forecast_dates[forecast_dates > max(filtered_well_data["Date"])].copy()
    well_forecast_Df["well"] = well
    well_forecast_Df["OIL_RATE"] = oil_forecast[forecast_dates > max(filtered_well_data["Date"])].copy()
    # Add data to forecast dataframe
    if len(all_wells_forecast_df) == 0:
        all_wells_forecast_df = well_forecast_Df.copy()
    else:
        all_wells_forecast_df = pd.concat([all_wells_forecast_df, well_forecast_Df])
    # Add data to Actual Dataframe
    well_data["well"] = well
    if len(all_wells_actual_df) == 0:
        all_wells_actual_df = well_data.copy()
    else:
        all_wells_actual_df = pd.concat([all_wells_actual_df, well_data])

# Export Results
print(well_summary)
print("Total Reserves, ", np.sum(well_summary["Rem_reserves"]))
print("Total EUR, ", np.sum(well_summary["EUR"]))

# Plot total forecast
try:
    # aggregate production forecast
    total_forecast_df = all_wells_forecast_df[["Date", "OIL_RATE"]].groupby(by="Date", as_index=False).sum()
    total_actual_df = all_wells_actual_df[["Date", "OIL_RATE"]].groupby(by="Date", as_index=False).sum()
    # Plot results
    plt.plot(total_forecast_df["Date"], total_forecast_df["OIL_RATE"], label="Forecast", linestyle="dashed")
    plt.plot(total_actual_df["Date"], total_actual_df["OIL_RATE"], label="Actual")
    plt.grid()
    plt.legend()
    plt.title("Production Forecast")
    plt.show()
except KeyError:
    pass
