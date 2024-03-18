import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# Statistical Analysis
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


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


def stationarity_test(merged_df):
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