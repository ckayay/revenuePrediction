import pandas as pd

################################################################
# Join datasets
################################################################

df_orders = pd.read_csv("data/olist_orders_dataset.csv")
df_orderItems = pd.read_csv('data/olist_order_items_dataset.csv')
df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')
df_reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
df_categoryNames = pd.read_csv('data/product_category_name_translation.csv')
df_customerOrders = pd.merge(df_orders, df_customers, left_on='customer_id', right_on='customer_id', how='left')
df_customerOrderItems = pd.merge(df_customerOrders, df_orderItems, left_on='order_id', right_on='order_id', how='left')
df_customerOrderReviews = pd.merge(df_customerOrderItems, df_reviews, left_on='order_id', right_on='order_id', how='left')
df_customerOrderReviewsAndProducts = pd.merge(df_customerOrderReviews, df_products, left_on='product_id', right_on='product_id', how='left')
df = pd.merge(df_customerOrderReviewsAndProducts, df_categoryNames, left_on='product_category_name', right_on='product_category_name', how='left')

################################################################
# Data cleaning
################################################################
# Drop votes- containing how many votes each subcategory has received. The results are summarized in the most_voted._
votes_columns = [s for s in df.columns if "votes_" in s]
df.drop(votes_columns, axis=1, inplace=True)
df.drop(['order_delivered_carrier_date'], axis=1, inplace=True)
df.drop(['product_height_cm'], axis=1, inplace=True)
df.drop(['product_width_cm'], axis=1, inplace=True)
df.drop(['product_length_cm'], axis=1, inplace=True)
df.drop(['product_name_length'], axis=1, inplace=True)
df.drop(['product_weight_g'], axis=1, inplace=True)
df.drop(['product_description_length'], axis=1, inplace=True)
df.drop(['product_photos_qty'], axis=1, inplace=True)
df.drop(['seller_id'], axis=1, inplace=True)
df.drop(['review_comment_title'], axis=1, inplace=True)
df.drop(['review_comment_message'], axis=1, inplace=True)
df.drop(['product_category_name'], axis=1, inplace=True)
# df.drop(['review_answer_timestamp'], axis=1, inplace=True)
# df.drop(['review_creation_date'], axis=1, inplace=True)


# Convert datetime features to the correct format
df.order_delivered_customer_date = pd.to_datetime(df.order_delivered_customer_date, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df.order_purchase_timestamp = pd.to_datetime(df.order_purchase_timestamp, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df.order_approved_at = pd.to_datetime(df.order_approved_at, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df.order_estimated_delivery_date = pd.to_datetime(df.order_estimated_delivery_date, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df.review_creation_date = pd.to_datetime(df.review_creation_date, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df.review_answer_timestamp = pd.to_datetime(df.review_answer_timestamp, format='%d/%m/%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
df = df.sort_values('order_purchase_timestamp')
df.to_csv("data/olist_consolidated_dataset.csv")


# prepare order metrics - total order value, avg order value, number of items
df_temp1= df.copy()
df_temp1.drop(df.columns.difference(['order_purchase_timestamp','price', 'product_id', 'customer_id']), 1, inplace=True)
df_grouped_by_order_date = df_temp1.groupby(['order_purchase_timestamp'])
df_total_order_value = df_grouped_by_order_date['price'].sum().to_frame(name = 'total_order_value')
df_number_items_sold = df_grouped_by_order_date['product_id'].count().to_frame(name = 'number_of_items_sold')
df_number_customers = df_grouped_by_order_date['customer_id'].nunique().to_frame(name = 'number_of_customers')
df_merged = pd.merge(df_total_order_value, df_number_items_sold, left_on='order_purchase_timestamp', right_on='order_purchase_timestamp', how='left')
df_merged = pd.merge(df_merged, df_number_customers, left_on='order_purchase_timestamp', right_on='order_purchase_timestamp', how='left')
df_merged = df_merged.sort_values('order_purchase_timestamp')
df_merged.to_csv("data/olist_order_metrics_dataset.csv")


print("DONE")
################################################################
