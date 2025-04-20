import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import pymongo

# Initialize Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Compliance Gap Analysis Dashboard"

# Layout will be dynamically created when the app starts
app.layout = html.Div([
    dbc.Container([
        html.H1("AI Regulatory Compliance Gap Analysis Dashboard", className="my-4"),
        
        dbc.Row([
            dbc.Col([
                html.H3("Project Selection"),
                dcc.Dropdown(
                    id="project-dropdown",
                    options=[],  # Will be populated with projects
                    value=None,
                    placeholder="Select a project"
                )
            ], width=4),
            
            dbc.Col([
                html.H3("Analysis Type"),
                dcc.RadioItems(
                    id="analysis-type",
                    options=[
                        {"label": "Jira Tasks Analysis", "value": "jira"},
                        {"label": "Document Analysis", "value": "document"},
                        {"label": "Comprehensive Analysis", "value": "comprehensive"}
                    ],
                    value="comprehensive",
                    inline=True,
                    className="mb-2"
                )
            ], width=4),
            
            dbc.Col([
                html.H3("Date Range"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    display_format="YYYY-MM-DD"
                )
            ], width=4)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Button(
                    "Refresh Data", 
                    id="refresh-button", 
                    className="btn btn-primary mt-4"
                )
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Overall Compliance Score", className="mt-4"),
                html.Div(id="overall-score-display", className="score-display")
            ], width=3),
            
            dbc.Col([
                html.H3("Total Gaps", className="mt-4"),
                html.Div(id="total-gaps-display", className="score-display")
            ], width=3),
            
            dbc.Col([
                html.H3("Critical Gaps", className="mt-4"),
                html.Div(id="critical-gaps-display", className="score-display")
            ], width=3),
            
            dbc.Col([
                html.H3("Resource Gaps", className="mt-4"),
                html.Div(id="resource-gaps-display", className="score-display")
            ], width=3)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Module Compliance Scores", className="mt-4"),
                dcc.Graph(id="module-scores-chart")
            ], width=6),
            
            dbc.Col([
                html.H3("Framework Compliance Scores", className="mt-4"),
                dcc.Graph(id="framework-scores-chart")
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Compliance Score Trend", className="mt-4"),
                dcc.Graph(id="compliance-trend-chart")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H3("Top Compliance Gaps", className="mt-4"),
                dash_table.DataTable(
                    id="top-gaps-table",
                    columns=[
                        {"name": "Module", "id": "moduleTitle"},
                        {"name": "Item", "id": "itemTitle"},
                        {"name": "Framework", "id": "frameworkId"},
                        {"name": "Gap", "id": "gap"},
                        {"name": "Score", "id": "score"},
                        {"name": "Criticality", "id": "criticality"}
                    ],
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{criticality} = "high"'},
                            'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                            'color': 'darkred'
                        },
                        {
                            'if': {'filter_query': '{criticality} = "medium"'},
                            'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                            'color': 'darkorange'
                        }
                    ],
                    page_size=5
                )
            ], width=12)
        ]),
        
        # Tabs for detailed views
        dbc.Tabs([
            dbc.Tab(label="Module Details", children=[
                # Module details content
                dbc.Row([
                    dbc.Col([
                        html.H3("Module Selection", className="mt-4"),
                        dcc.Dropdown(
                            id="module-dropdown",
                            options=[],  # Will be populated with modules
                            value=None,
                            placeholder="Select a module"
                        )
                    ], width=12)
                ]),
                
                # Module metrics
                dbc.Row([
                    dbc.Col([
                        html.H4("Module Compliance Score", className="mt-4"),
                        html.Div(id="module-score-display", className="score-display")
                    ], width=3),
                    
                    dbc.Col([
                        html.H4("Module Gaps", className="mt-4"),
                        html.Div(id="module-gaps-display", className="score-display")
                    ], width=3),
                    
                    dbc.Col([
                        html.H4("Critical Gaps", className="mt-4"),
                        html.Div(id="module-critical-gaps-display", className="score-display")
                    ], width=3),
                    
                    dbc.Col([
                        html.H4("Features Count", className="mt-4"),
                        html.Div(id="module-features-display", className="score-display")
                    ], width=3)
                ]),
                
                # Module charts
                dbc.Row([
                    dbc.Col([
                        html.H4("Feature Compliance Scores", className="mt-4"),
                        dcc.Graph(id="feature-scores-chart")
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Module Framework Coverage", className="mt-4"),
                        dcc.Graph(id="module-framework-coverage-chart")
                    ], width=6)
                ]),
                
                # Module gaps table
                dbc.Row([
                    dbc.Col([
                        html.H4("Module Compliance Gaps", className="mt-4"),
                        dash_table.DataTable(
                            id="module-gaps-table",
                            columns=[
                                {"name": "Feature", "id": "featureTitle"},
                                {"name": "Item", "id": "itemTitle"},
                                {"name": "Framework", "id": "frameworkId"},
                                {"name": "Gap", "id": "gap"},
                                {"name": "Score", "id": "score"},
                                {"name": "Criticality", "id": "criticality"}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{criticality} = "high"'},
                                    'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                                    'color': 'darkred'
                                },
                                {
                                    'if': {'filter_query': '{criticality} = "medium"'},
                                    'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                                    'color': 'darkorange'
                                }
                            ],
                            page_size=10
                        )
                    ], width=12)
                ])
            ]),

            dbc.Tab(label="Resource Analysis", children=[
                # Resource analysis content
                dbc.Row([
                    dbc.Col([
                        html.H4("Resource Gaps by Role", className="mt-4"),
                        dcc.Graph(id="resource-gaps-chart")
                    ], width=6),

                    dbc.Col([
                        html.H4("Skills Gaps", className="mt-4"),
                        dcc.Graph(id="skills-gaps-chart")
                    ], width=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        html.H4("Resource Allocation by Module", className="mt-4"),
                        dcc.Graph(id="module-resource-allocation-chart")
                    ], width=12)
                ]),

                dbc.Row([
                    dbc.Col([
                        html.H4("Training Needs", className="mt-4"),
                        dash_table.DataTable(
                            id="training-needs-table",
                            columns=[
                                {"name": "Skill/Certification", "id": "skill"},
                                {"name": "Roles Affected", "id": "rolesAffected"},
                                {"name": "Required", "id": "required"},
                                {"name": "Available", "id": "available"},
                                {"name": "Gap", "id": "gap"},
                                {"name": "Priority", "id": "priority"}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{priority} = "high"'},
                                    'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                                    'color': 'darkred'
                                },
                                {
                                    'if': {'filter_query': '{priority} = "medium"'},
                                    'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                                    'color': 'darkorange'
                                }
                            ],
                            page_size=10
                        )
                    ], width=12)
                ])
            ]),

            dbc.Tab(label="Framework Analysis", children=[
                # Framework analysis content
                dbc.Row([
                    dbc.Col([
                        html.H3("Framework Selection", className="mt-4"),
                        dcc.Dropdown(
                            id="framework-dropdown",
                            options=[
                                {"label": "EU AI Act", "value": "euAiAct"},
                                {"label": "ISO 42001", "value": "iso42001"},
                                {"label": "FDA SaMD", "value": "fdaSaMD"},
                                {"label": "GMLP", "value": "gmlp"}
                            ],
                            value="euAiAct",
                            placeholder="Select a framework"
                        )
                    ], width=12)
                ]),

                # Framework metrics
                dbc.Row([
                    dbc.Col([
                        html.H4("Framework Compliance Score", className="mt-4"),
                        html.Div(id="framework-score-display", className="score-display")
                    ], width=3),

                    dbc.Col([
                        html.H4("Framework Gaps", className="mt-4"),
                        html.Div(id="framework-gaps-display", className="score-display")
                    ], width=3),

                    dbc.Col([
                        html.H4("Articles Covered", className="mt-4"),
                        html.Div(id="framework-articles-display", className="score-display")
                    ], width=3),

                    dbc.Col([
                        html.H4("Articles with Issues", className="mt-4"),
                        html.Div(id="framework-issues-display", className="score-display")
                    ], width=3)
                ]),

                # Framework charts
                dbc.Row([
                    dbc.Col([
                        html.H4("Framework Article Coverage", className="mt-4"),
                        dcc.Graph(id="framework-article-coverage-chart")
                    ], width=12)
                ]),

                # Framework issues table
                dbc.Row([
                    dbc.Col([
                        html.H4("Articles with Compliance Issues", className="mt-4"),
                        dash_table.DataTable(
                            id="framework-issues-table",
                            columns=[
                                {"name": "Article", "id": "articleId"},
                                {"name": "Title", "id": "title"},
                                {"name": "Module", "id": "moduleTitle"},
                                {"name": "Gap Count", "id": "gapCount"},
                                {"name": "Average Score", "id": "avgScore"},
                                {"name": "Status", "id": "status"}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{status} = "Critical"'},
                                    'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                                    'color': 'darkred'
                                },
                                {
                                    'if': {'filter_query': '{status} = "Warning"'},
                                    'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                                    'color': 'darkorange'
                                },
                                {
                                    'if': {'filter_query': '{status} = "Good"'},
                                    'backgroundColor': 'rgba(0, 128, 0, 0.2)',
                                    'color': 'darkgreen'
                                }
                            ],
                            page_size=10
                        )
                    ], width=12)
                ])
            ])
        ], className="mt-4")

    ], fluid=True)
])

# Callback to populate project dropdown
@app.callback(
    Output("project-dropdown", "options"),
    Input("refresh-button", "n_clicks")
)
def populate_projects(n_clicks):
    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return []

    try:
        projects = gap_analyzer.jira_connector.get_projects()
        return [{"label": project["name"], "value": project["key"]} for project in projects]
    except:
        return []

# Callback to update overall metrics
@app.callback(
    [
        Output("overall-score-display", "children"),
        Output("total-gaps-display", "children"),
        Output("critical-gaps-display", "children"),
        Output("resource-gaps-display", "children")
    ],
    [
        Input("project-dropdown", "value"),
        Input("analysis-type", "value")
    ]
)
def update_overall_metrics(project_id, analysis_type):
    if not project_id:
        return "N/A", "N/A", "N/A", "N/A"

    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return "N/A", "N/A", "N/A", "N/A"

    db = gap_analyzer.db

    # Get analysis results
    results = list(db.gapAnalysisResults.find({
        "projectId": project_id,
        "analysisType": analysis_type
    }))

    if not results:
        return "N/A", "N/A", "N/A", "N/A"

    # Calculate metrics
    scores = [result.get("score", 0) for result in results]
    overall_score = int(sum(scores) / len(scores)) if scores else 0

    total_gaps = sum(len(result.get("gaps", [])) for result in results)
    critical_gaps = sum(len(result.get("gaps", [])) for result in results if result.get("criticality") == "high")
    resource_gaps = 3  # Placeholder - would be calculated based on resource analysis

    # Format the displays
    overall_score_html = html.Div([
        html.Span(f"{overall_score}%", className="big-number"),
        html.Div(get_score_color(overall_score), className="score-indicator")
    ])

    total_gaps_html = html.Div([
        html.Span(str(total_gaps), className="big-number"),
        html.Div("Issues", className="metric-label")
    ])

    critical_gaps_html = html.Div([
        html.Span(str(critical_gaps), className="big-number"),
        html.Div("Critical", className="metric-label")
    ])

    resource_gaps_html = html.Div([
        html.Span(str(resource_gaps), className="big-number"),
        html.Div("Needed", className="metric-label")
    ])

    return overall_score_html, total_gaps_html, critical_gaps_html, resource_gaps_html

# Helper function to get score color
def get_score_color(score):
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    else:
        return "red"

# Callback to update module scores chart
@app.callback(
    Output("module-scores-chart", "figure"),
    [Input("project-dropdown", "value"), Input("analysis-type", "value")]
)
def update_module_scores_chart(project_id, analysis_type):
    if not project_id:
        return go.Figure()

    gap_analyzer = app.config.get("gap_analyzer")
    if not gap_analyzer:
        return go.Figure()

    db = gap_analyzer.db

    # Get analysis results
    results = list(db.gapAnalysisResults.find({
        "projectId": project_id,
        "analysisType": analysis_type
    }))

    if not results:
        return go.Figure()

    # Group by module and calculate scores
    module_scores = {}
    for result in results:
        module_id = result.get("moduleId", "unknown")
        module_title = result.get("moduleTitle", "Unknown")
        score = result.get("score", 0)

        if module_id not in module_scores:
            module_scores[module_id] = {"moduleTitle": module_title, "scores": [], "total": 0, "count": 0}

        module_scores[module_id]["scores"].append(score)
        module_scores[module_id]["total"] += score
        module_scores[module_id]["count"] += 1

    # Calculate average scores
    for module_id in module_scores:
        module_scores[module_id]["avg_score"] = (
            module_scores[module_id]["total"] / module_scores[module_id]["count"]
            if module_scores[module_id]["count"] > 0 else 0
        )

    # Create dataframe for plotting
    df = pd.DataFrame([
        {
            "module": data["moduleTitle"],
            "score": data["avg_score"]
        }
        for module_id, data in module_scores.items()
    ])

    if df.empty:
        return go.Figure()

    # Create bar chart
    fig = px.bar(
        df,
        x="module",
        y="score",
        color="score",
        color_continuous_scale=["red", "yellow", "green"],
        range_color=[0, 100],
        labels={"module": "Module", "score": "Compliance Score (%)"},
        text="score"
    )

    fig.update_layout(
        title="Module Compliance Scores",
        xaxis_title="Module",
        yaxis_title="Compliance Score (%)",
        yaxis_range=[0, 100]
    )

    return fig

# Add more callbacks for other charts and tables
# These would be similar to the above callback for module scores
