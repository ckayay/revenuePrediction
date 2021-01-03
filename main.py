# ###################################################################################################################
# The Revenue has strong correlation on number of customers, upsell rate, conversion rate, total order value
# and price elasticity. If any of these metrics are down, the Revenue is down...
# Thus, predict these metrics and if any of their trend is down, the revenue will be down.
# TODO: Anamolies, add external events to explain peaks? (weather, etc), add tracers (actual vs predict, holidays)
# TODO: https://futurice.com/blog/business-forecasting-with-facebook-prophet
# ##################################################################################################################
import pandas as pd
from predict_prophet import *

################################################################
# Load datasets
################################################################
df = pd.read_csv("data/olist_order_metrics_dataset.csv")


################################################################
# Prep data & Predict TOTAL_ORDER_VALUE metric
################################################################
df_metric= df.copy()
df_metric.drop(df.columns.difference(['order_purchase_timestamp','total_order_value']), 1, inplace=True)
# Plot data
plot_data ('Sales/day' , 'order_purchase_timestamp', 'total_order_value', df_metric);

# rename columns to fit to prophet library (rename columns to ds & y
df_metric.columns = ['ds', 'y']
# remove rows with zero value
df_metric['y'] = df_metric['y'].fillna(0.0).astype(int)
df_metric= df_metric[df_metric['y'] != 0]

# Call prediction for 30 days
forecasted = prediction(df_metric);
# Evaluate results
eval_prediction (forecasted, df_metric)



################################################################
# Prep data & Predict NUMBER OF ITEMS SOLD metric
################################################################
df_metric= df.copy()
df_metric.drop(df.columns.difference(['order_purchase_timestamp','number_of_items_sold']), 1, inplace=True)
# Plot data
plot_data ('Items sold/day' , 'order_purchase_timestamp', 'number_of_items_sold', df_metric);

# rename columns to fit to prophet library (rename columns to ds & y
df_metric.columns = ['ds', 'y']
# remove rows with zero value
df_metric['y'] = df_metric['y'].fillna(0.0).astype(int)
df_metric= df_metric[df_metric['y'] != 0]

# Call prediction
forecasted = prediction(df_metric);
# Evaluate results
eval_prediction (forecasted, df_metric)


################################################################
# Prep data & Predict NUMBER OF CUSTOMERS metric
################################################################
df_metric= df.copy()
df_metric.drop(df.columns.difference(['order_purchase_timestamp','number_of_items_sold']), 1, inplace=True)
# Plot data
plot_data ('Items sold/day' , 'order_purchase_timestamp', 'number_of_items_sold', df_metric);

# rename columns to fit to prophet library (rename columns to ds & y
df_metric.columns = ['ds', 'y']
# remove rows with zero value
df_metric['y'] = df_metric['y'].fillna(0.0).astype(int)
df_metric= df_metric[df_metric['y'] != 0]

# Call prediction
forecasted = prediction(df_metric);
# Evaluate results
eval_prediction (forecasted, df_metric)