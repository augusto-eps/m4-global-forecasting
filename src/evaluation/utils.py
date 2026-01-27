import pandas as pd
import plotly.graph_objects as go

def plot_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    series_id: str,
    time_col: str="time_idx",
    value_col: str="value",
    pred_col: str="y_hat",
    horizon_col: str="horizon",
    n_train_points: int=31,
):
    #Filter one series
    train_s = train_df[train_df["M4id"] == series_id].sort_values(time_col)
    test_s = test_df[test_df["M4id"] == series_id].sort_values(time_col)
    pred_s = predictions[predictions["M4id"] == series_id].copy()

    #Last N train points
    train_tail = train_s.iloc[-n_train_points:]

    #Horizon
    horizon = pred_s[horizon_col].max()
    test_head = test_s.iloc[:horizon]

    #Recover absolute time index for predictions
    last_train_time = train_s[time_col].max()
    pred_s[time_col] = last_train_time + pred_s[horizon_col]

    #Plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_tail[time_col],
            y=train_tail[value_col],
            mode="lines",
            name="Train (last 2 months)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_head[time_col],
            y=test_head[value_col],
            mode="lines+markers",
            name="Actual (test)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred_s[time_col],
            y=pred_s[pred_col],
            mode="lines+markers",
            name="Prediction",
        )
    )

    fig.update_layout(
        title=f"Forecast vs Actuals — Series {series_id}",
        xaxis_title="Time index",
        yaxis_title="Value",
        legend_title="Series",
        template="plotly_white",
    )

    fig.show()

