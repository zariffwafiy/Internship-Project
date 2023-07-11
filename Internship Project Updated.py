#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pycaret.regression import * 

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth as auth
from dash import Input, Output

file_path = "Data\\Updated_Merged_BIR_CLS.csv"
df = pd.read_csv(file_path)
df = df.drop(["BASE_DATE_1", "BOND_CODE_1"], axis = 1)
#print(df.head())
#print(list(df.columns))
#print(df.shape)


# # Data Cleaning

# ### Remove unnecessary columns
df = df[["BASE_DATE", "BOND_CODE", "PROD_CODE", "BOND_NAME", "FACILITY_NAME", "ISSUER_NAME", "TRADE_RECENCY", "TRADE_FREQUENCY", "TRADE_TURNOVER", "COMPOSITE_LIQUIDITY_SCORE", "NOTCH_VARIANCE"]]


# ### Rearrange for cleaner dataframe
df = df[["BASE_DATE", "BOND_CODE", "PROD_CODE", "BOND_NAME", "FACILITY_NAME", "ISSUER_NAME", "TRADE_RECENCY", "TRADE_FREQUENCY", "TRADE_TURNOVER", "COMPOSITE_LIQUIDITY_SCORE", "NOTCH_VARIANCE"]]


# ### Remove duplicate rows
df.drop_duplicates()


# ### Sort values by increasing date
df["BASE_DATE"] = pd.to_datetime(df["BASE_DATE"], format = '%d/%m/%Y')
df.sort_values(by = ["BASE_DATE"], ascending = True, inplace = True)


# Given that the latest base date available is 28/04/2023, the YTD will be from 28/04/2022 - 28/04/2023

# # Exploratory Data Analysis

# ### Dataset summary
df.describe()


# ### Dataset shape
df.shape


# ### Dataset general info
df.info()


# ### Dependant variable : Notch Variance
df.NOTCH_VARIANCE.unique()
df.NOTCH_VARIANCE.value_counts()

# ### Dependant variable : Composite Liquidity Score
df.COMPOSITE_LIQUIDITY_SCORE.unique()
df.COMPOSITE_LIQUIDITY_SCORE.value_counts()


# ### Boxplots
boxplot = df.boxplot(column = ["NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"])
plt.title("Boxplot of Notch Variance and Composite Liquidity Score")

boxplot_sns = sns.boxplot(data = df[["NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"]])
plt.title("Boxplot of Notch Variance and Composite Liquidity Score")
plt.xlabel("Features")
plt.ylabel("Values")
plt.xticks(rotation = 45)
sns.set_style("whitegrid")


# ### KDE plot
included_df = df[["NOTCH_VARIANCE", "TRADE_RECENCY", "TRADE_FREQUENCY", "TRADE_TURNOVER", "COMPOSITE_LIQUIDITY_SCORE"]]
for col in included_df.columns: 
    fig, ax = plt.subplots(figsize = (8, 6))
    
    sns.kdeplot(data = df[col], ax =ax
               )
    
    ax.set_title(f"KDE Plot - {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    
    #

# # Analytics

# ### Find correlation of variables, no dropping dependant variables (Correlation Matrix @ Correlation Heatmap)
plt.figure(figsize = (16, 6))
heatmap = sns.heatmap(df[["NOTCH_VARIANCE", "TRADE_RECENCY", "TRADE_FREQUENCY", "TRADE_TURNOVER", "COMPOSITE_LIQUIDITY_SCORE"]].corr(), vmin = -1, vmax = 1, annot = True, cmap = "BrBG")
heatmap.set_title("Correlation Heatmap", fontdict= {
    "fontsize" : 16
}, pad = 12)


# The notch variance does not show a strong correlation with the composite liquidity score 
# 
# **keep in note that :
#     * lower notch variance means that the BIR is better than credit rating
#     * higher notch variance means that the BIR is worse than credit rating**
# 

# choose the bond with the highest occurence
#print(df["BOND_CODE"].value_counts().head(1))

# chosen bond : bond code = PZ200004
chosen_bond_code = "PZ200004"
data = df.loc[df["BOND_CODE"] == chosen_bond_code]
plot_df = df[df["BOND_CODE"] == chosen_bond_code].sort_values(by = ["BASE_DATE"], ascending = True)
plot_df = plot_df[["BASE_DATE", "NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"]]

fig, ax = plt.subplots(figsize = (10, 5))
plot_df.plot(x = "BASE_DATE", y = ["NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"], kind = "line", ax = ax)
plt.xlabel("Values")
start_date = ""
plt.ylabel("Date")
plt.xticks(rotation = 45)
plt.title("Notch Variance and Composite Liquidity Score")
plt.tight_layout()
#

# observe the correlation between notch variance and composite liquidity score
#print(plot_df["NOTCH_VARIANCE"].corr(plot_df["COMPOSITE_LIQUIDITY_SCORE"]))


# ### Relation of dependant variable between multiple bonds
top_3_modes = df["BOND_CODE"].value_counts().head(3)
#print(top_3_modes)

bond_code_1 = "PZ200004"
bond_code_2 = "VZ120024"
bond_code_3 = "VK160112"

df1 = df[df["BOND_CODE"] == bond_code_1].sort_values(by = ["BASE_DATE"], ascending = True)
df2 = df[df["BOND_CODE"] == bond_code_2].sort_values(by = ["BASE_DATE"], ascending = True)
df3 = df[df["BOND_CODE"] == bond_code_3].sort_values(by = ["BASE_DATE"], ascending = True)


# ### Composite Liquidity Score
bond_code_1 = "PZ200004"
bond_code_2 = "VZ120024"
bond_code_3 = "VK160112"

df1 = df[df["BOND_CODE"] == bond_code_1].sort_values(by = ["BASE_DATE"], ascending = True)
df2 = df[df["BOND_CODE"] == bond_code_2].sort_values(by = ["BASE_DATE"], ascending = True)
df3 = df[df["BOND_CODE"] == bond_code_3].sort_values(by = ["BASE_DATE"], ascending = True)


# ### Notch Variance
plt.plot(df1["BASE_DATE"], df1["NOTCH_VARIANCE"], label= bond_code_1)
plt.plot(df2["BASE_DATE"], df2["NOTCH_VARIANCE"], label= bond_code_2)
plt.plot(df3["BASE_DATE"], df3["NOTCH_VARIANCE"], label= bond_code_3)
plt.ylabel("Values")
plt.xticks(rotation = 45)
plt.title("Notch Variance Score of 5 Different Bonds")
plt.tight_layout()
plt.legend()
#
df["BASE_DATE"] = pd.to_datetime(df["BASE_DATE"], format = '%d/%m/%Y')
df.sort_values(by = ["BASE_DATE"], ascending = True, inplace = True)

new_data = df.loc[df["BOND_CODE"] == chosen_bond_code]
new_data = new_data[["BASE_DATE", "NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"]]
new_data = new_data.set_index("BASE_DATE")


# # Data Modelling

# ### Split Training and Testing Data
train_data = new_data[:int(0.8 * len(new_data))]
test_data = new_data[int(0.8 * len(new_data)):]
step = len(test_data)
df_diff = train_data


# ### Autocorrelation Plot
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10))

# NOTCH_VARIANCE
plot_acf(new_data["NOTCH_VARIANCE"], lags = 20, ax = axes[0])
axes[0].set_title("ACF - NOTCH_VARIANCE")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Autocorrelation")

# COMPOSITE_LIQUIDITY_SCORE
plot_acf(new_data["COMPOSITE_LIQUIDITY_SCORE"], lags = 20, ax = axes[1])
axes[1].set_title("ACF - COMPOSITE_LIQUIDITY_SCORE")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Autocorrelation")

plt.tight_layout()
#

# Significance: Look for lag values that fall outside the shaded region or horizontal lines representing the confidence intervals. Lag values with autocorrelation outside these bounds are considered statistically significant and indicate a potential relationship between the current observation and the lagged values.
# 
# Decay in Correlation: Focus on the decay of autocorrelation values as the lag increases. If the autocorrelation values decay rapidly and approach zero, it suggests that the current observation is not strongly related to past observations at longer lags. On the other hand, slower decay or persistent autocorrelations indicate a stronger relationship between the current observation and past observations at specific lags.
# 
# Seasonality and Patterns: Pay attention to any repeating patterns or significant autocorrelation at specific lags. If there are clear spikes or clusters of autocorrelation values at regular intervals, it may suggest the presence of seasonality or repeating patterns in the data. These significant lags can guide the selection of the lag order.
# 
# While examining the ACF plot, it's not necessary to focus solely on the points where the autocorrelation is exactly zero. Instead, consider the overall pattern, significant lags outside the confidence intervals, and the rate of decay in correlation to make an informed decision about the lag order.
# 
# Additionally, it can be helpful to complement the ACF analysis with the partial autocorrelation function (PACF) plot, which shows the direct relationship between observations at specific lags while controlling for intermediate lags. The PACF plot can provide additional insights into the lag order selection process.
# 
# Overall, lag order selection is a combination of statistical significanc

# ### Check for Stationarity

# check for stationarity and make time series stationary
def check_stationarity(timeseries):
    column_name = timeseries.name
    
    # perform Augmented Dickey-Fuller test
    #print("checking stationarity for timeseries: " + column_name)
    result = adfuller(timeseries)
    #print("AD-F Statistics:", result[0])
    #print("p-value:", result[1])
    #print("Critical Values:")
    for key, value in result[4].items():
        print(key, ":", value)
    
    # check if differencinig is needed based on p-value
    # significance level
    alpha = 0.05 # significance level
    if result[1] > alpha:
        print("The time series is non-stationary.")
        print("Consider applying differencing to make it staionary.")
        
    else:
        print("The time series is stationary.")
        print("Differencing is not necessary")
    #print("")
    
check_stationarity(df_diff["NOTCH_VARIANCE"])
check_stationarity(df_diff["COMPOSITE_LIQUIDITY_SCORE"])


# As we can see, there is a need to differentiate NOTCH_VARIANCE, but not COMPOSITE_LIQUIDITY_SCORE

# ### Differentiate when needed

# differentiate only NOTCH_VARIANCE and not COMPOSITE_LIQUIDITY_SCORE
df_diff = train_data.copy()
df_diff["NOTCH_VARIANCE"] = df_diff["NOTCH_VARIANCE"].diff()
df_diff["NOTCH_VARIANCE"].fillna(df_diff["NOTCH_VARIANCE"].mode()[0], inplace = True)
#print(df_diff)

check_stationarity(df_diff["NOTCH_VARIANCE"])

# No need to differentiate NOTCH_VARIANCE further

# ### Selecting the suitable lag order
p = 10

for i in range(1, p+1):
    model = VAR(df_diff)
    results = model.fit(i)
    #print(f'VAR Order {i}')
    #print('AIC {}'.format(results.aic))
    #print('BIC {}'.format(results.bic))
    #print()


# We can see that the lag order 1 will give lowest values of BIC and AIC. Both AIC and BIC punishes overfitting by providing increasing values
# 
# Here are few items to take note:
# * BIC tend to penalize complex models, suggests a simpler model that still fits the data well.
# * AIC tend to favor them, indicates a relatively more complex model that provides a good fit to the data.
# 
# The AIC tends to favor more complex models, including models with higher lag orders, as it puts less penalty on the number of parameters. Therefore, if your primary goal is to have a more flexible and complex model that captures potential long-term dependencies and dynamics in the data, you may consider selecting the lag order that corresponds to the lowest AIC value.
# 
# However, it's important to note that a more complex model may not always lead to better forecasting performance. A higher lag order could also introduce noise or overfitting, especially if you have limited data or if the relationships among the variables are not well-defined.
# 
# The BIC, on the other hand, penalizes complex models more heavily by taking into account the sample size. It tends to prefer simpler models with fewer parameters. If your goal is to balance model complexity and overfitting while still obtaining good forecasting performance, you may consider selecting the lag order that corresponds to the lowest BIC value.
selected_order = model.select_order(maxlags = 30)
p = selected_order.bic
#print("Selected Order (P) of VAR model: ", p)
#print(selected_order.summary())


# We can see that lag order 1 holds the minimums for all the statistical measures 

# ### Build VAR model
lag_order = 10
results = model.fit(lag_order)
#results.summary()


# ### Check for serial correlation of residuals using Durban Watson Statistics
dw_statistic = durbin_watson(results.resid)
#print("Durbin-Watson Statistic:\n", dw_statistic)


# No serial correlation present for both timeseries

# ### Forecasting
forecast_input = train_data.values[-10:]
z = results.forecast(y = forecast_input, steps = step)
index = pd.date_range(start='29/4/2023',periods=step,freq='D')
df_forecast = pd.DataFrame(z, index = index, columns = df_diff.columns)


# ### Invert the differenced timeseries back if necessary
df_forecast['NOTCH_VARIANCE'] = (df['NOTCH_VARIANCE'].iloc[-1]) + df_forecast['NOTCH_VARIANCE'].cumsum()


# ### Plot Results
# Create subplots for NOTCH_VARIANCE and COMPOSITE_LIQUIDITY_SCORE
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot for NOTCH_VARIANCE
axes[0].plot(train_data.index, train_data["NOTCH_VARIANCE"], label="Train (Actual)", color='blue')
axes[0].plot(test_data.index, test_data["NOTCH_VARIANCE"], label = "Test (Actual)", color = "green")
axes[0].plot(df_forecast.index, df_forecast["NOTCH_VARIANCE"], label="Forecast", color='orange')
axes[0].set_xlabel("Date")
axes[0].set_ylabel("NOTCH_VARIANCE")
axes[0].set_title("Forecast vs Actuals - NOTCH_VARIANCE")
axes[0].legend()
axes[0].grid(True)

# Plot for COMPOSITE_LIQUIDITY_SCORE
axes[1].plot(train_data.index, train_data["COMPOSITE_LIQUIDITY_SCORE"], label="Train (Actual)", color='blue')
axes[1].plot(test_data.index, test_data["COMPOSITE_LIQUIDITY_SCORE"], label = "Test (Actual)", color = "green")
axes[1].plot(df_forecast.index, df_forecast["COMPOSITE_LIQUIDITY_SCORE"], label="Forecast", color='orange')
axes[1].set_xlabel("Date")
axes[1].set_ylabel("COMPOSITE_LIQUIDITY_SCORE")
axes[1].set_title("Forecast vs Actuals - COMPOSITE_LIQUIDITY_SCORE")
axes[1].legend()
axes[1].grid(True)

# Connect the lines between the last point of "Train (Actual)" and the first point of "Forecast"
axes[0].plot([new_data.index[-1], df_forecast.index[0]], [new_data["NOTCH_VARIANCE"].iloc[-1], df_forecast["NOTCH_VARIANCE"].iloc[0]], color='orange', linestyle='-')
axes[1].plot([new_data.index[-1], df_forecast.index[0]], [new_data["COMPOSITE_LIQUIDITY_SCORE"].iloc[-1], df_forecast["COMPOSITE_LIQUIDITY_SCORE"].iloc[0]], color='orange', linestyle='-')

plt.tight_layout()
# # Model Evaluation

# ## RMSE
# Evaluate the Forecasts
rmse_BIR = rmse(test_data["NOTCH_VARIANCE"], df_forecast["NOTCH_VARIANCE"])
rmse_CLS = rmse(test_data["COMPOSITE_LIQUIDITY_SCORE"], df_forecast["COMPOSITE_LIQUIDITY_SCORE"])

#print("Root Mean Squared Error (NOTCH_VARIANCE):", rmse_BIR)
#print("Root Mean Squared Error (COMPOSITE_LIQUIDITY_SCORE):", rmse_CLS)


# Lag order 10 returns the best(lowest) rmse score for both timeseries

# ## MAE
from sklearn.metrics import mean_absolute_error

mae_BIR = mean_absolute_error(test_data["NOTCH_VARIANCE"], df_forecast["NOTCH_VARIANCE"])
mae_CLS = mean_absolute_error(test_data["COMPOSITE_LIQUIDITY_SCORE"], df_forecast["COMPOSITE_LIQUIDITY_SCORE"])

#print("Mean Absolute Error (NOTCH_VARIANCE):", mae_BIR)
#print("Mean Absolute Error (COMPOSITE_LIQUIDITY_SCORE)", mae_CLS)


# ## R2 Score
from sklearn.metrics import r2_score

r2_BIR = r2_score(test_data["NOTCH_VARIANCE"], df_forecast["NOTCH_VARIANCE"])
r2_CLS = r2_score(test_data["COMPOSITE_LIQUIDITY_SCORE"], df_forecast["COMPOSITE_LIQUIDITY_SCORE"])

#print("r2 score (NOTCH_VARIANCE):", r2_BIR)
#print("r2 score (COMPOSITE_LIQUIDITY_SCORE):", r2_CLS)


# The r2 score is horrible, meaning that the model does not fit the data very well.
# 
# A negative r2 score indicatest that the model performs worse than a horizontal line(mean of data)

# ## Residuals
residuals_BIR = test_data["NOTCH_VARIANCE"].values - df_forecast["NOTCH_VARIANCE"].values
residuals_CLS = test_data["COMPOSITE_LIQUIDITY_SCORE"].values - df_forecast["COMPOSITE_LIQUIDITY_SCORE"].values

fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 12))

axes[0].scatter(df_forecast["NOTCH_VARIANCE"], residuals_BIR)
axes[0].axhline(y = 0, color = "r", linestyle = "--")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("NOTCH_VARIANCE Residual Plot")

axes[1].scatter(df_forecast["COMPOSITE_LIQUIDITY_SCORE"], residuals_CLS)
axes[1].axhline(y = 0, color = "r", linestyle = "--")
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")
axes[1].set_title("COMPOSITE_LIQUDIITY_SCORE Residual Plot")

#plt.show()


# All of the residuals for NOTCH_VARIANCE are negative, means the model is overestimating the data. Predicted values are much higher than the actual values. This indicates a bad model.
# 
# Some of the residuals for COMPOSITE_LIQUIDITY_SCORE are negative while some are positive, means the model is both overestimating and underestimating the data. The model underestimates at lower predicted values, and overestimates in higher predicted values. This also indicates a bad model. 

# ## Recommendation Section

# Retreive the best time for buyers and sellers of the bond
# 
# * buyer - high notch variance, high composite liquidity score
# * seller - low notch variance, high composite liquidity score
# 
# 
# Can be done by calculating a weighted score of both variables

# ### Buyer
weight_BIR_buyer = 0.5
weight_CLS_buyer = 0.5

df_forecast["COMBINED_SCORE_BUYER"] = (weight_BIR_buyer * df_forecast["NOTCH_VARIANCE"]) + (weight_CLS_buyer * df_forecast["COMPOSITE_LIQUIDITY_SCORE"])
# need to filter the df so that the maximum composite liquidity score should only be lesser than 5
df_forecast = df_forecast[df_forecast["COMPOSITE_LIQUIDITY_SCORE"] <= 5] 

score_buyer = df_forecast["COMBINED_SCORE_BUYER"].max()
date_score_buyer = df_forecast.loc[df_forecast["COMBINED_SCORE_BUYER"] == score_buyer].index[0]
BIR_buyer = df_forecast.loc[date_score_buyer, "NOTCH_VARIANCE"]
CLS_buyer = df_forecast.loc[date_score_buyer, "COMPOSITE_LIQUIDITY_SCORE"]
#print("The date of interest for buyer is: " +  str(date_score_buyer.strftime("%d/%m/%Y")) + " \nwith: \n   Notch Variance of: " + str(BIR_buyer) + "\n   Composite Liquidity Score of: " + str(CLS_buyer))

# ### Seller
weight_BIR_seller = -0.5
weight_CLS_seller = 0.5

df_forecast["COMBINED_SCORE_SELLER"] = (weight_BIR_seller * df_forecast["NOTCH_VARIANCE"]) + (weight_CLS_seller * df_forecast["COMPOSITE_LIQUIDITY_SCORE"])
# need to filter the df so that the maximum composite liquidity score should only be lesser than 5
df_forecast = df_forecast[df_forecast["COMPOSITE_LIQUIDITY_SCORE"] <= 5]

score_seller = df_forecast["COMBINED_SCORE_SELLER"].max()
date_score_seller = df_forecast.loc[df_forecast["COMBINED_SCORE_SELLER"] == score_seller].index[0]
BIR_seller = df_forecast.loc[date_score_seller, "NOTCH_VARIANCE"]
CLS_seller = df_forecast.loc[date_score_seller, "COMPOSITE_LIQUIDITY_SCORE"]
#print("The date of interest for seller is: " +  str(date_score_seller.strftime("%d/%m/%Y")) + " \nwith: \n   Notch Variance of: " + str(BIR_seller) + "\n   Composite Liquidity Score of: " + str(CLS_seller))

# for callbacks 
unique_bonds = sorted(df["BOND_NAME"].unique().tolist())

# dash app
app = dash.Dash(__name__)

app.title = "Analyzing Bond Dynamics"

app.layout = html.Div(
    children = [
        html.Div(
            children = [
                # can reduce a hierarchy, will get back to it later
                html.Div(
                    children = [
                        html.H1(
                            "Analyzing Bond Dynamics",
                            className= "header-title"
                        ),
                        html.P(
                            "Analyze the tradeability of your fixed income assests through this interactive dashboard",
                            className= "header-description"
                        )
                    ], 
                    className = "header-wrapper"
                ),
            ],
            className = "header"
        ),

        html.Div(
            children = [
                html.Div(
                    children = "Bond Filter",
                    className = "bond-filter-title"
                ),
                dcc.Dropdown(
                    id = "bond-filter",
                    options = [
                        {
                            "label": bond, "value": bond
                        } 
                        for bond in unique_bonds
                    ],
                    value = "TG EXCELLENCE SUKUK WAKALAH (TRANCHE 1)",
                    clearable = False,
                    className = "bond-filter"
                )
            ]
        ),

        # actual line chart 
        html.Div(
            children = [
                dcc.Graph(
                    id = "line-chart-1",
                    className= "line-chart-1"
                )
            ],
        ),

        # forecasting parameters
        html.Div(
            children =[
                html.Div(
                    children = "lag order",
                    className = "lag-order-filter-title"
                ),
                dcc.Slider(
                    id = "lag-order-filter",
                    min = 1,
                    max = 10,
                    step = 1,
                    value = 1,
                    marks={i: str(i) for i in range(1, 10 + 1)}
                )
            ],
            className= "lag-order-filter"
        ),

        html.Div(
            children = [
                html.Div(
                    children = "forecast horizon",
                    className = "forecast-horizon-filter-title"
                ),
                dcc.Slider(
                    id = "forecast-horizon-filter",
                    min = 1,
                    max = 30,
                    step = 1,
                    value = 1,
                    marks={i: str(i) for i in range(1, 30 + 1)}
                )
            ],
            className = "forecast-horizon-filter"
        ),

        # forecasted line chart
        html.Div(
            children = [
                dcc.Graph(
                    id = "line-chart-2",
                    className = "line-chart-2"
                )
            ]
        )

    ],
    className = "body"
)

@app.callback(
    Output("line-chart-1", "figure"),
    Output("line-chart-2", "figure"),
    Input("bond-filter", "value"),
    Input("lag-order-filter", "value"), 
    Input("forecast-horizon-filter", "value")
)

def update_charts(bond_name, lag_order, forecast_horizon): 
    # retrieve only bond code
    bond_code = df.loc[df["BOND_NAME"] == bond_name, "BOND_CODE"].iloc[0]
    # filter only chosen bond code, sort values by date
    new_data = df[df["BOND_CODE"] == bond_code].sort_values(by = ["BASE_DATE"], ascending = True)
    # include only necessary columns
    new_data = new_data[["BASE_DATE", "NOTCH_VARIANCE", "COMPOSITE_LIQUIDITY_SCORE"]].set_index("BASE_DATE")

    # data modelling
    train_data = new_data[:int(0.8 * len(new_data))]
    test_data = new_data[int(0.8 * len(new_data)):]

    model = VAR(train_data)
    results = model.fit(lag_order)

    forecast_input = train_data.values[-lag_order:]
    z = results.forecast(y = forecast_input, steps = forecast_horizon)
    index = pd.date_range(start='29/4/2023',periods=forecast_horizon,freq='D')
    df_forecast = pd.DataFrame(z, index = index, columns = df_diff.columns)

    line_chart_figure_1 = {
        "data": [
            # BIR
            go.Scatter(
                x = new_data.index,
                y = new_data["NOTCH_VARIANCE"],
                mode = "lines",
                line = dict(
                    color = "#101D6B", # blue
                    width = 2
                ),
                name = "BIR"
            ),
            # CLS
            go.Scatter(
                x = new_data.index,
                y = new_data["COMPOSITE_LIQUIDITY_SCORE"],
                mode = "lines",
                line = dict(
                    color = "#FDD128",
                    width = 2
                ),
                name = "CLS"
            )
        ],
        "layout": go.Layout(
            title = {
                "text" : "Bond: " + bond_name + " (" + bond_code + ")"
            },
            xaxis = dict(
                title = "Date"
            ),
            yaxis = dict(
                title = "Actual Values"
            )
        )
    }

    line_chart_figure_2 = {
        "data": [
            # NOTCH VARIANCE
            go.Scatter(
                x = train_data.index,
                y = train_data["NOTCH_VARIANCE"],
                mode = "lines",
                name = "BIR",
                line = dict(color = "#101D6B")
            ),
            go.Scatter(
                x=[train_data.index[-1], test_data.index[0]],
                y=[train_data["NOTCH_VARIANCE"].iloc[-1], test_data["NOTCH_VARIANCE"].iloc[0]],
                mode="lines",
                name = "BIR", 
                line=dict(color='#101D6B', dash='dash'),
                showlegend = False
            ),
            go.Scatter(
                x = test_data.index,
                y = test_data["NOTCH_VARIANCE"],
                mode = "lines",
                name = "BIR",
                line = dict(color = "#101D6B"),
                showlegend = False
            ),
            go.Scatter(
                x = df_forecast.index, 
                y = df_forecast["NOTCH_VARIANCE"],
                mode = "lines",
                name = "Forecasted BIR",
                line = dict(color = "#A020F0")
            ),
             go.Scatter(
                x=[test_data.index[-1], df_forecast.index[0]],
                y=[test_data["NOTCH_VARIANCE"].iloc[-1], df_forecast["NOTCH_VARIANCE"].iloc[0]],
                mode="lines",
                name = "Forecasted BIR", 
                line=dict(color='#A020F0', dash='dash'),
                showlegend = False
            ),

            # COMPOSITE_LIQUIDITY_SCORE
            go.Scatter(
                x = train_data.index,
                y = train_data["COMPOSITE_LIQUIDITY_SCORE"],
                mode = "lines",
                name = "CLS",
                line = dict(color = "#FDD128")
            ),
            go.Scatter(
                x=[train_data.index[-1], test_data.index[0]],
                y=[train_data["COMPOSITE_LIQUIDITY_SCORE"].iloc[-1], test_data["COMPOSITE_LIQUIDITY_SCORE"].iloc[0]],
                mode="lines",
                name = "CLS", 
                line=dict(color='#FDD128', dash='dash'),
                showlegend = False
            ),
            go.Scatter(
                x = test_data.index,
                y = test_data["COMPOSITE_LIQUIDITY_SCORE"],
                mode = "lines",
                name = "CLS",
                line = dict(color = "#FDD128"),
                showlegend = False
            ),
            go.Scatter(
                x = df_forecast.index,
                y = df_forecast["COMPOSITE_LIQUIDITY_SCORE"],
                mode = "lines",
                name = "Forecasted CLS",
                line = dict(color = "#FFA500")
            ),
            go.Scatter(
                x=[test_data.index[-1], df_forecast.index[0]],
                y=[test_data["COMPOSITE_LIQUIDITY_SCORE"].iloc[-1], df_forecast["COMPOSITE_LIQUIDITY_SCORE"].iloc[0]],
                mode="lines",
                name = "Forecasted CLS", 
                line=dict(color='#FFA500', dash='dash'),
                showlegend = False
            )
        ],
        "layout" : go.Layout(
            title = {
                "text" : "Forecasted Bond: " + bond_name + " (" + bond_code + ")"
            },
            xaxis = dict(
                title = "Date"
            ),
            yaxis = dict(
                title = "Forecasted Values"
            )
        )
    }

    return line_chart_figure_1, line_chart_figure_2

if __name__ == "__main__":
    app.run_server(debug = True)