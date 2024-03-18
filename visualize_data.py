import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


def check_relation(merged_df):
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
    draw_trend(daily_sales,'date','sales','Trend of Sales over Time')
    draw_trend(df_oil,'date','dcoilwtico','Trend of Oil Price over Time')
    check_relation(merged_df)

