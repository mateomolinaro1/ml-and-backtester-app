from dash import html, dcc
import dash_bootstrap_components as dbc
# Import nécessaire pour le typage (optionnel mais propre)
from ml_and_backtester_app.dashboard.s3_loader import S3PathManager

def create_layout(paths: S3PathManager) -> html.Div:
    # On peut maintenant utiliser paths.method pour personnaliser la barre
    method_label = paths.method.capitalize() # "Rolling" ou "Expanding"

    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [
                        html.Span("ML Backtester Dashboard "),
                        dbc.Badge(method_label, color="info", className="ms-2 fs-6")
                    ], 
                    className="fw-bold fs-5"
                ),
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