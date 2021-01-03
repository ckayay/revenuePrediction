import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
import seaborn as sns
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

sns.set(rc={'figure.figsize': (20, 12)})
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)


def plot_data(str_title, str_column_x, str_column_y, data):
    ax = sns.lineplot(x=str_column_x, y=str_column_y, data=data)
    ax.set_title(str_title)
    ax.set(ylim=(1, 20000))
    plt.show()
    return


def prediction(train_data):
    # run hyper parameter and find the optimum parameters
    m = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10.0, daily_seasonality=True, yearly_seasonality=True)
    m.add_country_holidays(country_name='BR')
    # fit the model
    m.fit(train_data)

    # define the period for which we want a prediction
    future = m.make_future_dataframe(periods=30)
    # use the model to make a forecast
    forecast = m.predict(future)
    return forecast


def eval_prediction(forecast, data):
    # Cross validation
    # df_cv = cross_validation(forecast_model, initial='500 days', period='30 days', horizon= '30 days')
    # df_p = performance_metrics(df_cv)
    # fig = plot_cross_validation_metric(df_cv, metric='mape')
    # fig.show()
    py.plot([
        go.Scatter(x=data['ds'], y=data['y'], name='y'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
        go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
    ])
    return


def hyper_parameters(train):
    # Run hyper parameters to find the optimum ones
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        # define the model
        m = Prophet(**params)
        m.add_country_holidays(country_name='BR')
        # fit the model
        m.fit(train)
        df_cv = cross_validation(m, initial='500 days', period='30 days', horizon='30 days')
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print("Tuning results:")
    print(tuning_results)

    best_params = all_params[np.argmin(rmses)]
    print("Best results:")
    print(best_params)
    return
