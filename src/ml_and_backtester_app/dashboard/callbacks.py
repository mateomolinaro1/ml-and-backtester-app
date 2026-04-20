import json
import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
import dash_bootstrap_components as dbc

# On ne garde que les fonctions de chargement, plus les dictionnaires statiques
from ml_and_backtester_app.dashboard.s3_loader import (
    load_parquet,
    presigned_url,
    S3PathManager, # Pour le typage
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

# ─── Tab content builders (MAINTENANT DYNAMIQUES) ───────────────────────────

def _fmp_tab(paths: S3PathManager) -> html.Div:
    equity_df = load_parquet(paths.DATA["fmp_equity_curves"])
    perf_df = load_parquet(paths.DATA["fmp_performance"])

    rows: list = [_section_header("Factor Mimicking Portfolios")]

    if equity_df is not None:
        fig = _line_chart(equity_df, "FMP Equity Curves", "Cumulative Return")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    rows.append(dbc.Row(dbc.Col(_table_card("FMP Performance Metrics", perf_df)), className="mb-4"))

    rows.append(_section_header("Detailed Analytics"))
    png_pairs = [
        ("Betas Distribution", "Betas Over Time"),
        ("R² Over Time", "Significance Proportion"),
        ("Betas Summary", "R² Summary"),
        ("Equity Curves (static)", "Performance Summary (static)"),
    ]
    for left, right in png_pairs:
        rows.append(
            dbc.Row(
                [
                    dbc.Col(_png_card(left, paths.FMP_FIGURES[left]), width=6),
                    dbc.Col(_png_card(right, paths.FMP_FIGURES[right]), width=6),
                ],
                className="mb-4",
            )
        )
    return html.Div(rows)

def _forecasting_tab(paths: S3PathManager) -> html.Div:
    val_score_df = load_parquet(paths.DATA["best_val_score"])
    oos_rmse_df = load_parquet(paths.DATA["oos_rmse_overtime"])
    rmse_table_df = load_parquet(paths.DATA["oos_rmse_table"])
    sign_acc_df = load_parquet(paths.DATA["oos_sign_accuracy"])

    rows: list = [_section_header("Forecasting — Model Performance")]

    if val_score_df is not None:
        fig = _line_chart(val_score_df, "Best Validation RMSE Over Time", "RMSE")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    if oos_rmse_df is not None:
        fig = _line_chart(oos_rmse_df, "Rolling OOS RMSE Over Time", "OOS RMSE")
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    rows.append(
        dbc.Row(
            [
                dbc.Col(_table_card("Mean OOS RMSE by Model", rmse_table_df), width=6),
                dbc.Col(_table_card("OOS Sign Accuracy by Model", sign_acc_df), width=6),
            ],
            className="mb-4",
        )
    )

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
                    dbc.Col(_png_card(left, paths.FORECASTING_FIGURES[left]), width=6),
                    dbc.Col(_png_card(right, paths.FORECASTING_FIGURES[right]), width=6),
                ],
                className="mb-4",
            )
        )

    rows.append(dbc.Row(dbc.Col(_png_card("Features Sample", paths.FORECASTING_FIGURES["Features Sample"])), className="mb-4"))
    return html.Div(rows)

def _dynamic_alloc_tab(paths: S3PathManager) -> html.Div:
    cum_returns_df = load_parquet(paths.DATA["dynamic_alloc_cum_returns"])
    perf_df = load_parquet(paths.DATA["dynamic_alloc_performance"])

    rows: list = [_section_header("Dynamic Allocation Strategy")]

    if cum_returns_df is not None:
        fig = _line_chart(cum_returns_df, "Dynamic Allocation — Cumulative Returns", "Cumulative Return")
        for trace in fig.data:
            if "Bench" in str(trace.name):
                trace.line.dash = "dash"
                trace.line.color = "black"
        rows.append(dbc.Row(dbc.Col(_chart_card(fig)), className="mb-4"))

    rows.append(dbc.Row(dbc.Col(_table_card("Performance Table", perf_df)), className="mb-4"))

    rows.append(_section_header("Static Charts"))
    rows.append(
        dbc.Row(
            [
                dbc.Col(_png_card("Cumulative Returns (static)", paths.DYNAMIC_ALLOC_FIGURES["Cumulative Returns (static)"]), width=6),
                dbc.Col(_png_card("Performance Table (static)", paths.DYNAMIC_ALLOC_FIGURES["Performance Table (static)"]), width=6),
            ],
            className="mb-4",
        )
    )
    return html.Div(rows)


def _backtest_tab(paths: S3PathManager) -> html.Div:
    # Ta liste de ratios financiers
    funda_cols = [
        'capei', 'be', 'bm', 'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 
        'ps', 'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa', 
        'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity', 
        'pretret_noa', 'pretret_earnat', 'gprof', 'equity_invcap', 'debt_invcap', 
        'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt', 
        'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt', 'curr_debt', 
        'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent', 
        'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio', 'intcov', 'intcov_ratio', 
        'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', 'inv_turn', 
        'at_turn', 'rect_turn', 'pay_turn', 'sale_invcap', 'sale_equity', 'sale_nwc', 
        'rd_sale', 'adv_sale', 'staff_sale', 'accrual', 'ret_crsp', 'mktcap', 
        'price', 'ptb', 'peg_trailing', 'divyield'
    ]

    # Construction des options (Momentum + Ratios)
    options = [{"label": " Momentum (Technical)", "value": "Momentum"}]
    options += [{"label": f" {c.upper()}", "value": c} for c in sorted(funda_cols)]

    return html.Div([
        _section_header("Strategy Backtester"),
        dbc.Row([
            # Panneau de contrôle (4 colonnes de large)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Parameters", className="fw-bold text-uppercase small"),
                    dbc.CardBody([
                        # 1. Sélection du facteur
                        html.Label("Factor / Signal", className="small mb-1"),
                        dcc.Dropdown(
                            id="bt-ratio-selector", 
                            options=options, 
                            value="Momentum",
                            className="mb-3"
                        ),
                        
                        # 2. Direction de la stratégie
                        html.Label("Strategy Logic", className="small mb-1"),
                        dbc.RadioItems(
                            id="bt-direction",
                            options=[
                                {"label": "Buy High Values (Growth/Mom)", "value": False},
                                {"label": "Buy Low Values (Value/Cheap)", "value": True},
                            ],
                            value=False,
                            className="mb-3 small",
                        ),

                        # 3. Date de début
                        html.Label("Start Date", className="small mb-1 d-block"),
                        dcc.DatePickerSingle(
                            id="bt-date-picker",
                            date="2010-01-01",
                            display_format="YYYY-MM-DD",
                            className="mb-3"
                        ),

                        # 4. Coûts et Rebalancement
                        dbc.Row([
                            dbc.Col([
                                html.Label("Costs (bps)", className="small"),
                                dbc.Input(id="bt-costs", type="number", value=10, size="sm"),
                            ]),
                            dbc.Col([
                                html.Label("Rebal. (days)", className="small"),
                                dbc.Input(id="bt-rebal", type="number", value=22, size="sm"),
                            ]),
                        ], className="mb-3"),

                        html.Label("Percentiles (Low / High)", className="small mb-1"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id="bt-p-low", type="number", value=10, min=0, max=49, size="sm"),
                                html.FormText("Bottom %", className="smaller text-muted"),
                            ]),
                            dbc.Col([
                                dbc.Input(id="bt-p-high", type="number", value=90, min=51, max=100, size="sm"),
                                html.FormText("Top %", className="smaller text-muted"),
                            ]),
                        ], className="mb-3"),

                        dbc.Button("Run Backtest", id="btn-run-backtest", color="primary", className="w-100 mt-2 shadow-sm"),
                    ])
                ], className="border-0 shadow-sm"),
                html.Div(id="backtest-status-msg", className="mt-3")
            ], width=4),
            
            # Affichage du résultat (8 colonnes de large)
            dbc.Col([
                _png_card("Strategy Performance", paths.BACKTEST_FIGURES["Cumulative Performance"])
            ], width=8),
        ])
    ])



# ─── Config & Run tab ────────────────────────────────────────────────────────

_STATUS_COLORS = {"idle": "secondary", "running": "warning", "done": "success", "error": "danger"}

def _status_badge(status: str) -> dbc.Badge:
    return dbc.Badge(status.upper(), color=_STATUS_COLORS.get(status, "secondary"), className="fs-6")

def _config_tab() -> html.Div:
    from ml_and_backtester_app.dashboard.pipeline_runner import get_output, get_status, load_config

    try:
        config_str = json.dumps(load_config(), indent=2)
    except Exception as exc:
        config_str = f"// Error loading config: {exc}"

    status = get_status()
    log_content = get_output() or "(no output yet — run the pipeline to see logs here)"

    return html.Div([
        _section_header("Config & Run Pipeline"),
        dbc.Row([
            dbc.Col([
                html.Div("config/run_pipeline_config.json", className="text-muted small font-monospace mb-2"),
                dcc.Textarea(id="config-textarea", value=config_str, style={"width": "100%", "height": "600px", "fontFamily": "'Courier New', monospace", "fontSize": "12px", "backgroundColor": "#f8f9fa"}),
            ], width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col(dbc.Button("Save & Run", id="btn-run-pipeline", color="success", size="sm"), width="auto"),
                    dbc.Col(dbc.Button("Stop", id="btn-stop-pipeline", color="danger", outline=True, size="sm"), width="auto"),
                    dbc.Col(html.Div(id="pipeline-status-badge", children=_status_badge(status)), width="auto"),
                ], className="mb-3 g-2"),
                html.Pre(log_content, id="pipeline-log", style={"height": "560px", "backgroundColor": "#1e1e1e", "color": "#d4d4d4", "fontSize": "11px"}),
            ], width=6),
        ]),
        dcc.Interval(id="pipeline-interval", interval=2000, n_intervals=0, disabled=(status != "running")),
    ])

# ─── Callback registration ───────────────────────────────────────────────────

def register(app: dash.Dash, paths: S3PathManager) -> None:
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
        Input("btn-refresh", "n_clicks"),
    )
    def render_tab(active_tab: str, _n_clicks: int) -> html.Div:
        # L'objet 'paths' est accessible ici grâce à la closure (portée de fonction)
        if active_tab == "tab-fmp":
            return _fmp_tab(paths)
        if active_tab == "tab-forecasting":
            return _forecasting_tab(paths)
        if active_tab == "tab-dynamic-alloc":
            return _dynamic_alloc_tab(paths)
        if active_tab == "tab-config":
            return _config_tab()
        if active_tab == "tab-backtest":
            return _backtest_tab(paths)
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
    
    @app.callback(
        Output("backtest-status-msg", "children"),
        Input("btn-run-backtest", "n_clicks"),
        State("bt-ratio-selector", "value"),
        State("bt-direction", "value"),
        State("bt-date-picker", "date"),
        State("bt-costs", "value"),
        State("bt-rebal", "value"),
        State("bt-p-low", "value"),
        State("bt-p-high", "value"),
        prevent_initial_call=True
    )
    def on_run_backtest(n, ratio, direction, start_date, costs, rebal, p_low, p_high):
        from ml_and_backtester_app.dashboard.pipeline_runner import start
        
        # On construit le dictionnaire de config dynamiquement
        # C'est ce que ton run_backtest.py va recevoir via SQS
        config_payload = {
            "ratio_name": ratio,
            "ascending": direction,  # True = Buy Low, False = Buy High
            "start_date": start_date,
            "transaction_costs": costs,
            "nb_period_to_exclude": rebal,
            "strategy_name": f"STRAT_{ratio.upper()}",
            "percentiles": [p_low, p_high]
        }
        
        # On envoie ça au Worker
        ok, msg = start(json.dumps(config_payload), task_type="backtest") 
        
        if ok:
            # Petit message sympa pour l'utilisateur
            return dbc.Alert(
                f"Backtest for {ratio} launched! Check the 'Config & Run' tab for logs.", 
                color="success", 
                className="py-2 small"
            )
        return dbc.Alert(f"Error: {msg}", color="danger", className="py-2 small")
