"""
Dash callbacks for the ML Backtester Dashboard.

Each tab is rendered lazily — data is loaded from S3 only when the tab is
activated (or the Refresh button is clicked).
"""

import json

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc

from ml_and_backtester_app.dashboard.s3_loader import (
    DATA,
    DYNAMIC_ALLOC_FIGURES,
    FORECASTING_FIGURES,
    FMP_FIGURES,
    load_parquet,
    presigned_url,
)


# ─── UI component builders ───────────────────────────────────────────────────


def _line_chart(df: pd.DataFrame, title: str, yaxis: str = "Value") -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def _datatable(df: pd.DataFrame) -> dash_table.DataTable:
    df_out = df.reset_index()
    df_out.columns = [str(c) for c in df_out.columns]
    for col in df_out.select_dtypes(include="number").columns:
        df_out[col] = df_out[col].round(4)
    return dash_table.DataTable(
        data=df_out.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_out.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "8px", "fontFamily": "monospace", "fontSize": "13px"},
        style_header={"backgroundColor": "#212529", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}],
        sort_action="native",
        page_size=20,
    )


def _chart_card(figure: go.Figure) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=figure, config={"displayModeBar": True})),
        className="shadow-sm",
    )


def _table_card(title: str, df: pd.DataFrame | None) -> dbc.Card:
    body = (
        [html.H6(title, className="card-title mb-3"), _datatable(df)]
        if df is not None
        else [html.P(f"{title} — data not available (run the pipeline first).", className="text-muted small")]
    )
    return dbc.Card(dbc.CardBody(body), className="shadow-sm")


def _png_card(title: str, key: str) -> dbc.Card:
    url = presigned_url(key)
    body = (
        [html.H6(title, className="card-title text-muted small"), html.Img(src=url, style={"width": "100%"})]
        if url
        else [html.P(f"{title} — image not available.", className="text-muted small")]
    )
    return dbc.Card(dbc.CardBody(body), className="shadow-sm")


def _section_header(text: str) -> html.H5:
    return html.H5(text, className="mt-4 mb-3 border-bottom pb-2 text-secondary")


# ─── Tab content builders ────────────────────────────────────────────────────


def _fmp_tab() -> html.Div:
    equity_df = load_parquet(DATA["fmp_equity_curves"])
    perf_df = load_parquet(DATA["fmp_performance"])

    rows: list = [_section_header("Factor Mimicking Portfolios")]

    # Interactive equity curves
    if equity_df is not None:
        fig = _line_chart(equity_df, "FMP Equity Curves", "Cumulative Return")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    # Interactive performance table
    rows.append(dbc.Row(dbc.Col(_table_card("FMP Performance Metrics", perf_df)), className="mb-4"))

    # Static PNGs in 2-column grid
    rows.append(_section_header("Detailed Analytics"))
    png_pairs = [
        ("Betas Distribution", "Betas Over Time"),
        ("R\u00b2 Over Time", "Significance Proportion"),
        ("Betas Summary", "R\u00b2 Summary"),
        ("Equity Curves (static)", "Performance Summary (static)"),
    ]
    for left, right in png_pairs:
        rows.append(
            dbc.Row(
                [
                    dbc.Col(_png_card(left, FMP_FIGURES[left]), width=6),
                    dbc.Col(_png_card(right, FMP_FIGURES[right]), width=6),
                ],
                className="mb-4",
            )
        )

    return html.Div(rows)


def _forecasting_tab() -> html.Div:
    val_score_df = load_parquet(DATA["best_val_score"])
    oos_rmse_df = load_parquet(DATA["oos_rmse_overtime"])
    rmse_table_df = load_parquet(DATA["oos_rmse_table"])
    sign_acc_df = load_parquet(DATA["oos_sign_accuracy"])

    rows: list = [_section_header("Forecasting — Model Performance")]

    # Interactive validation score chart
    if val_score_df is not None:
        fig = _line_chart(val_score_df, "Best Validation RMSE Over Time", "RMSE")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    # Interactive OOS RMSE chart
    if oos_rmse_df is not None:
        fig = _line_chart(oos_rmse_df, "Rolling OOS RMSE Over Time", "OOS RMSE")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    # Side-by-side interactive tables
    rows.append(
        dbc.Row(
            [
                dbc.Col(_table_card("Mean OOS RMSE by Model", rmse_table_df), width=6),
                dbc.Col(_table_card("OOS Sign Accuracy by Model", sign_acc_df), width=6),
            ],
            className="mb-4",
        )
    )

    # Static PNGs
    rows.append(_section_header("Detailed Analytics"))
    png_pairs = [
        ("Best Hyperparams", "Model Parameters"),
        ("Selected Features", "Mean Parameters"),
        ("Best Val Score (static)", "OOS RMSE Overtime (static)"),
        ("OOS RMSE Table (static)", "Sign Accuracy (static)"),
    ]
    for left, right in png_pairs:
        rows.append(
            dbc.Row(
                [
                    dbc.Col(_png_card(left, FORECASTING_FIGURES[left]), width=6),
                    dbc.Col(_png_card(right, FORECASTING_FIGURES[right]), width=6),
                ],
                className="mb-4",
            )
        )

    rows.append(dbc.Row(dbc.Col(_png_card("Features Sample", FORECASTING_FIGURES["Features Sample"])), className="mb-4"))

    return html.Div(rows)


def _dynamic_alloc_tab() -> html.Div:
    cum_returns_df = load_parquet(DATA["dynamic_alloc_cum_returns"])
    perf_df = load_parquet(DATA["dynamic_alloc_performance"])

    rows: list = [_section_header("Dynamic Allocation Strategy")]

    # Interactive cumulative returns
    if cum_returns_df is not None:
        fig = _line_chart(cum_returns_df, "Dynamic Allocation — Cumulative Returns", "Cumulative Return")
        for trace in fig.data:
            if "Bench" in str(trace.name):
                trace.line.dash = "dash"
                trace.line.color = "black"
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    # Interactive performance table
    rows.append(dbc.Row(dbc.Col(_table_card("Performance Table", perf_df)), className="mb-4"))

    # Static PNGs
    rows.append(_section_header("Static Charts"))
    rows.append(
        dbc.Row(
            [
                dbc.Col(
                    _png_card("Cumulative Returns (static)", DYNAMIC_ALLOC_FIGURES["Cumulative Returns (static)"]),
                    width=6,
                ),
                dbc.Col(
                    _png_card("Performance Table (static)", DYNAMIC_ALLOC_FIGURES["Performance Table (static)"]),
                    width=6,
                ),
            ],
            className="mb-4",
        )
    )

    return html.Div(rows)


# ─── Config & Run tab ────────────────────────────────────────────────────────

_STATUS_COLORS = {
    "idle": "secondary",
    "running": "warning",
    "done": "success",
    "error": "danger",
}


def _status_badge(status: str) -> dbc.Badge:
    return dbc.Badge(status.upper(), color=_STATUS_COLORS.get(status, "secondary"), className="fs-6")


def _config_tab() -> html.Div:
    from ml_and_backtester_app.dashboard.pipeline_runner import (
        get_output,
        get_status,
        load_config,
    )

    try:
        config_str = json.dumps(load_config(), indent=2)
    except Exception as exc:
        config_str = f"// Error loading config: {exc}"

    status = get_status()
    log_content = get_output() or "(no output yet — run the pipeline to see logs here)"

    return html.Div([
        _section_header("Config & Run Pipeline"),
        dbc.Row([

            # ── Left: JSON editor ────────────────────────────────────
            dbc.Col([
                html.Div(
                    "config/run_pipeline_config.json",
                    className="text-muted small font-monospace mb-2",
                ),
                dcc.Textarea(
                    id="config-textarea",
                    value=config_str,
                    style={
                        "width": "100%",
                        "height": "600px",
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "12px",
                        "border": "1px solid #dee2e6",
                        "borderRadius": "4px",
                        "padding": "10px",
                        "backgroundColor": "#f8f9fa",
                        "resize": "vertical",
                    },
                ),
                html.Div(id="config-validation-msg", className="mt-2 small"),
            ], width=6),

            # ── Right: Controls + Log ────────────────────────────────
            dbc.Col([
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Save & Run Pipeline",
                                id="btn-run-pipeline",
                                color="success",
                                size="sm",
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Stop",
                                id="btn-stop-pipeline",
                                color="danger",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Div(id="pipeline-status-badge", children=_status_badge(status)),
                            width="auto",
                            className="align-self-center",
                        ),
                    ],
                    className="mb-3 g-2 align-items-center",
                ),

                html.Pre(
                    log_content,
                    id="pipeline-log",
                    style={
                        "height": "560px",
                        "overflowY": "auto",
                        "backgroundColor": "#1e1e1e",
                        "color": "#d4d4d4",
                        "padding": "12px",
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "11px",
                        "borderRadius": "6px",
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-all",
                    },
                ),
            ], width=6),
        ]),

        # Interval component — lives inside the tab so it's only active when
        # the Config tab is open. Starts enabled only if pipeline is running.
        dcc.Interval(
            id="pipeline-interval",
            interval=2000,
            n_intervals=0,
            disabled=(status != "running"),
        ),
    ])


# ─── Callback registration ───────────────────────────────────────────────────


def register(app: dash.Dash) -> None:
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
        Input("btn-refresh", "n_clicks"),
    )
    def render_tab(active_tab: str, _n_clicks: int) -> html.Div:
        if active_tab == "tab-fmp":
            return _fmp_tab()
        if active_tab == "tab-forecasting":
            return _forecasting_tab()
        if active_tab == "tab-dynamic-alloc":
            return _dynamic_alloc_tab()
        if active_tab == "tab-config":
            return _config_tab()
        return html.P("Select a tab above.")

    # ── Save & Run ───────────────────────────────────────────────────────────

    @app.callback(
        Output("pipeline-interval", "disabled"),
        Output("config-validation-msg", "children"),
        Output("config-validation-msg", "className"),
        Input("btn-run-pipeline", "n_clicks"),
        State("config-textarea", "value"),
        prevent_initial_call=True,
    )
    def on_run(_n: int, config_str: str):
        from ml_and_backtester_app.dashboard.pipeline_runner import start

        ok, msg = start(config_str or "")
        if ok:
            return False, msg, "text-success small mt-2"
        return True, msg, "text-danger small mt-2"

    # ── Live output polling ──────────────────────────────────────────────────

    @app.callback(
        Output("pipeline-log", "children"),
        Output("pipeline-status-badge", "children"),
        Output("pipeline-interval", "disabled", allow_duplicate=True),
        Input("pipeline-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_output(_n: int):
        from ml_and_backtester_app.dashboard.pipeline_runner import get_output, get_status

        status = get_status()
        output = get_output() or "(no output yet)"
        return output, _status_badge(status), status != "running"

    # ── Stop ─────────────────────────────────────────────────────────────────

    @app.callback(
        Output("pipeline-log", "children", allow_duplicate=True),
        Output("pipeline-status-badge", "children", allow_duplicate=True),
        Input("btn-stop-pipeline", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_stop(_n: int):
        from ml_and_backtester_app.dashboard.pipeline_runner import get_output, stop

        msg = stop()
        output = get_output()
        suffix = f"\n\n[{msg}]"
        return (output + suffix) if output else suffix, _status_badge("idle")
