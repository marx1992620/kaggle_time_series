import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# Statistical Analysis
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
# Feature Processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def show_sales_relation(merged_df):
    # Select numerical variables for correlation analysis
    numerical_vars = ['sales', 'transactions', 'dcoilwtico']

    # Plot scatter plot matrix
    sns.pairplot(merged_df[numerical_vars])
    plt.show()


def draw_trend(dataframe,x_value,y_value,title):
    # Create a time series plot with slider
    fig = px.line(dataframe, x=x_value, y=y_value)
    # fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title=title, title_x=0.5)
    fig.show()


def visualize_df(merged_df,df_oil):
    daily_sales = merged_df.groupby('date')['sales'].sum().reset_index()
    # draw_trend(daily_sales,'date','sales','Trend of Sales over Time')
    # draw_trend(df_oil,'date','dcoilwtico','Trend of Oil Price over Time')
    # show_sales_relation(merged_df)


def stationary_test(merged_df):
    print("-"*10,"start","-"*10)
    # Statistical Test of the 'sales' column in the merged_df using Adfuller
    sales_data = merged_df['sales']

    # Perform ADF test
    result = adfuller(sales_data)

    # Extract the test statistics and p-value from the result
    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Print the test statistics and critical values
    print(f"ADF Test Statistics: {test_statistic}")
    print(f"P-value: {p_value}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    # Check the p-value against a significance level (e.g., 0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis: The sales data is stationary.")
    else:
        print("Fail to reject the null hypothesis: The sales data is non-stationary.")


def t_test(merged_df):

    # Extract the relevant variables for the hypothesis test
    promo_sales = merged_df[merged_df['onpromotion'] == 1]['sales']
    non_promo_sales = merged_df[merged_df['onpromotion'] == 0]['sales']

    # Perform a two-sample t-test to compare sales between promotional and non-promotional periods
    t_statistic, p_value = ttest_ind(promo_sales, non_promo_sales)

    # Print the test result
    print("Hypothesis Testing for Promotional Activities:")
    print("Null Hypothesis (H0): The promotional activities do not have a significant impact on store sales.")
    print("Alternative Hypothesis (H1): The promotional activities have a significant impact on store sales.")
    print("-" * 30)
    print("Test Statistic:", t_statistic)
    print("P-value:", p_value)
    print("-" * 30)
    if p_value < 0.05:
        print("Reject the null hypothesis. Promotional activities have a significant impact on store sales at Corporation Favorita.")
    else:
        print("Fail to reject the null hypothesis. Promotional activities do not have a significant impact on store sales at Corporation Favorita.")


def check_sales_dates(merged_df):
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['year'] = merged_df['date'].dt.year

    lowest_sales_dates = merged_df.groupby('year')['date'].min()
    highest_sales_dates = merged_df.groupby('year')['date'].max()

    print("Dates with the lowest sales for each year:\n", lowest_sales_dates)
    print("="*30)
    print("Dates with the highest sales for each year:\n", highest_sales_dates)


def check_sales_relation(merged_df):
    # Calculate correlations between sales and promotions, oil prices, holidays
    corr_sales_promotions = merged_df['sales'].corr(merged_df['onpromotion'])
    corr_sales_oil = merged_df['sales'].corr(merged_df['dcoilwtico'])
    corr_sales_holidays = merged_df['sales'].corr(merged_df['holiday_type'] == 'Holiday')

    # Print the correlation values
    print(f"Correlation between Sales and Promotions: {corr_sales_promotions}")
    print(f"Correlation between Sales and Oil Prices: {corr_sales_oil}")
    print(f"Correlation between Sales and Holidays: {corr_sales_holidays}")


def extract_date(merged_df):
    # Extracting Date Components
    # Create a copy of the merged dataset
    merged_df_copy = merged_df.copy()
    merged_df_copy['date'] = pd.to_datetime(merged_df_copy['date'])
    merged_df_copy['year'] = merged_df_copy['date'].dt.year
    merged_df_copy['month'] = merged_df_copy['date'].dt.month
    merged_df_copy['day'] = merged_df_copy['date'].dt.day
    print(merged_df_copy.head())
    columns_to_drop = ['date', 'id', 'locale', 'locale_name', 'description', 'store_type', 'transferred', 'state']
    merged_df_copy = merged_df_copy.drop(columns=columns_to_drop)
    unique_families = merged_df_copy['family'].unique()
    print(unique_families)

    return merged_df_copy


def reset_category(merged_df_copy):
    # Define the category lists for each product category
    food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
    home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
    clothing_families = ['LINGERIE', 'LADYSWARE']
    grocery_families = ['GROCERY I', 'GROCERY II']
    stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
    cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
    hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']

    # Categorize the 'family' column based on the product categories
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(food_families), 'FOODS', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(home_families), 'HOME', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(clothing_families), 'CLOTHING', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(grocery_families), 'GROCERY', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(stationery_families), 'STATIONERY', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(cleaning_families), 'CLEANING', merged_df_copy['family'])
    merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(hardware_families), 'HARDWARE', merged_df_copy['family'])

    # Print the updated unique values
    unique_families = merged_df_copy['family'].unique()
    print(unique_families)


def feature_scaling(merged_df_copy):
    scaler = StandardScaler()
    num_cols = ['sales', 'transactions', 'dcoilwtico']
    merged_df_copy[num_cols] = scaler.fit_transform(merged_df_copy[num_cols])


def one_hot_encoding(merged_df_copy):
    # Define the categorical columns to encode
    categorical_columns = ["family", "city", "holiday_type"]

    # Perform one-hot encoding
    encoder = OneHotEncoder()
    one_hot_encoded_data = encoder.fit_transform(merged_df_copy[categorical_columns])

    # Create column names for the one-hot encoded data
    column_names = encoder.get_feature_names_out(categorical_columns)

    # Convert the one-hot encoded data to a DataFrame
    merged_df_encoded = pd.DataFrame(one_hot_encoded_data.toarray(), columns=column_names)

    # Concatenate the original dataframe with the one-hot encoded data
    merged_df_encoded = pd.concat([merged_df_copy, merged_df_encoded], axis=1)

    # Drop the original categorical columns
    merged_df_encoded.drop(categorical_columns, axis=1, inplace=True)

    # Print the head of the encoded DataFrame
    print(merged_df_encoded.head())

    return merged_df_encoded

