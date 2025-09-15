# --- Plotly Dash Application ---

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from stock_market_analytics.config import config
from stock_market_analytics.inference.inference_steps import (
    download_and_load_model,
    get_inference_data,
    make_prediction_intervals,
    predict_quantiles,
)

# The app needs an external stylesheet.
# Dash automatically serves files from an 'assets' folder.
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Stock Return Predictions"
server = app.server

# --- App Layout ---
app.layout = html.Div(
    className="app-container",
    children=[
        # Header
        html.Div(
            className="header",
            children=[
                html.H1("Stock Log-Return Forecast Dashboard"),
                html.P(
                    "Visualizing multi-quantile predictions with calibrated intervals."
                ),
            ],
        ),
        # Controls
        html.Div(
            className="controls-container",
            children=[
                dcc.Input(
                    id="symbol-input",
                    type="text",
                    placeholder="Enter stock symbol (e.g., AAPL)...",
                    value="AAPL",
                    className="symbol-input",
                ),
                html.Button(
                    "Generate Forecast",
                    id="submit-button",
                    n_clicks=0,
                    className="submit-button",
                ),
            ],
        ),
        # Main content area
        dcc.Loading(
            id="loading-spinner",
            type="cube",
            color="#1f77b4",
            children=[
                html.Div(
                    id="output-container",
                    className="output-container",
                    children=[
                        # This Div will be populated by the callback
                    ],
                )
            ],
        ),
    ],
)


# --- Empty Layout for Initial State ---
def build_initial_layout():
    return html.Div(
        [
            html.Div(
                className="kpi-container",
                children=[
                    # KPIs will be populated here
                ],
            ),
            html.Div(
                className="plot-container",
                children=[
                    # Plots will be populated here
                ],
            ),
        ]
    )


# --- Callback to Update Dashboard ---
@app.callback(
    Output("output-container", "children"),
    Input("submit-button", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True,
)
def update_dashboard(symbol):  # type: ignore
    if not symbol:
        raise PreventUpdate

    # 1. Load Model (will be cached after first run)
    model = download_and_load_model()

    # 2. Get Data & Predictions
    features_df = get_inference_data(symbol)
    intervals_df = make_prediction_intervals(model, features_df)
    quantiles_df = predict_quantiles(model, intervals_df)

    # For easier plotting, merge dataframes
    # In a real case, ensure indices/keys are aligned for a proper merge
    merged_df = quantiles_df.copy()

    # --- Create Visualizations ---

    # KPI Cards
    latest_data = merged_df.dropna(subset=[config.modeling.target]).iloc[-1]
    median_pred = latest_data["pred_Q_50"]
    interval_width = (
        latest_data["pred_high_quantile"] - latest_data["pred_low_quantile"]
    )

    kpi_cards = html.Div(
        className="kpi-container",
        children=[
            html.Div(
                className="kpi-card",
                children=[
                    html.H3("Latest Actual Log-Return"),
                    html.P(
                        f"{latest_data[config.modeling.target]:.4f}",
                        className="kpi-value",
                    ),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.H3("Predicted Median (Q50)"),
                    html.P(f"{median_pred:.4f}", className="kpi-value"),
                ],
            ),
            html.Div(
                className="kpi-card",
                children=[
                    html.H3("Calibrated Interval Width"),
                    html.P(f"{interval_width:.4f}", className="kpi-value"),
                ],
            ),
        ],
    )

    # Plot 1: Prediction Intervals
    fig_intervals = go.Figure()
    fig_intervals.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["pred_high_quantile"],
            mode="lines",
            line={"width": 0},
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="Upper Bound",
            showlegend=False,
        )
    )
    fig_intervals.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df["pred_low_quantile"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="95% Prediction Interval",
        )
    )
    fig_intervals.add_trace(
        go.Scatter(
            x=merged_df["date"],
            y=merged_df[config.modeling.target],
            mode="lines",
            line={"color": "white", "width": 2},
            name="Actual Log-Return",
        )
    )
    fig_intervals.update_layout(
        title=f"Calibrated Prediction Intervals vs. Actual Log-Returns for {symbol.upper()}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Log-Return",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    # Plot 2: Quantile Fan Chart for the last 30 days
    df_tail = merged_df.tail(30)
    fig_fan = go.Figure()
    quantile_cols = [f"pred_Q_{q * 100:.0f}" for q in config.modeling.quantiles]

    # Add quantile prediction lines (from widest to narrowest)
    num_quantiles = len(config.modeling.quantiles)
    for i in range(num_quantiles // 2):
        upper_q_col = quantile_cols[-(i + 1)]
        lower_q_col = quantile_cols[i]

        fig_fan.add_trace(
            go.Scatter(
                x=df_tail["date"],
                y=df_tail[upper_q_col],
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig_fan.add_trace(
            go.Scatter(
                x=df_tail["date"],
                y=df_tail[lower_q_col],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba(214, 39, 40, {0.1 + (i * 0.05)})",
                name=f"Q{config.modeling.quantiles[i] * 100:.0f}-Q{config.modeling.quantiles[-(i + 1)] * 100:.0f}",
            )
        )

    # Add median line
    fig_fan.add_trace(
        go.Scatter(
            x=df_tail["date"],
            y=df_tail["pred_Q_50"],
            mode="lines",
            line={"color": "#ff7f0e", "width": 3},
            name="Median (Q50)",
        )
    )
    fig_fan.update_layout(
        title=f"Quantile Forecast Fan Chart (Last 30 Days) for {symbol.upper()}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Predicted Log-Return",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    # # Assemble the final layout
    # return html.Div([
    #     kpi_cards,
    #     html.Div(className="plot-container", children=[
    #         dcc.Graph(figure=fig_intervals),
    #         dcc.Graph(figure=fig_fan)
    #     ])
    # ])

    return html.Div(
        [
            kpi_cards,
            html.Div(
                className="plot-container",
                children=[
                    dcc.Graph(
                        figure=fig_intervals,
                        # FIX: Add explicit style to prevent resizing feedback loop
                        style={"height": "45vh", "width": "55vw"},
                    ),
                    dcc.Graph(
                        figure=fig_fan,
                        # FIX: Add explicit style to prevent resizing feedback loop
                        style={"height": "45vh", "width": "55vw"},
                    ),
                ],
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
