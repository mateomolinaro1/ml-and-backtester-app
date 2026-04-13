from dash import html, dcc
import dash_bootstrap_components as dbc


def create_layout() -> html.Div:
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("ML Backtester Dashboard", className="fw-bold fs-5"),
                dbc.Button(
                    "Refresh",
                    id="btn-refresh",
                    color="light",
                    size="sm",
                    className="ms-auto",
                    n_clicks=0,
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-4 px-3",
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(label="FMP Analytics", tab_id="tab-fmp"),
            dbc.Tab(label="Forecasting", tab_id="tab-forecasting"),
            dbc.Tab(label="Dynamic Allocation", tab_id="tab-dynamic-alloc"),
            dbc.Tab(label="Config & Run", tab_id="tab-config"),
        ],
        id="tabs",
        active_tab="tab-config",
        className="mb-3",
    )

    return html.Div(
        [
            navbar,
            dbc.Container(
                [
                    tabs,
                    dcc.Loading(
                        html.Div(id="tab-content"),
                        type="circle",
                        color="#0d6efd",
                    ),
                ],
                fluid=True,
            ),
        ]
    )
