import plotly.graph_objects as go
import pandas as pd


def plot_stock(df, ticker, chart_type, predicted_price=None, future_date=None):
    """
    Create a Plotly chart for stock prices, optionally including a predicted price.

    Parameters:
    - df: DataFrame with Date index and Close column
    - ticker: Stock ticker symbol
    - chart_type: Type of chart (e.g., "Line")
    - predicted_price: Predicted stock price (float, optional)
    - future_date: Future date for prediction (datetime.date, optional)

    Returns:
    - Plotly Figure object
    """
    # Create figure
    fig = go.Figure()

    # Add historical close prices
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='#1E88E5')
        )
    )

    # Add predicted price if provided
    if predicted_price is not None and future_date is not None:
        # Convert future_date to datetime for plotting
        future_date = pd.to_datetime(future_date)

        # Get last historical point
        last_date = df.index[-1]
        last_price = df['Close'][-1]

        # Add connecting line from last historical point to predicted point
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_date],
                y=[last_price, predicted_price],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#D81B60', dash='dash'),
                marker=dict(size=10, symbol='circle')
            )
        )

        # Add annotation for predicted price
        fig.add_annotation(
            x=future_date,
            y=predicted_price,
            text=f'Predicted: ${predicted_price:.2f}',
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30,
            bgcolor='white',
            font=dict(size=12, color='#D81B60')
        )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )

    return fig