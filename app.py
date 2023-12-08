import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('celeba_buffalo_l.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2('Menu', style={'color': 'white'}),
                html.Br(),
                dcc.Link('Load Dataset', href='/load-dataset', style={'color': 'white'}),
                html.Br(),
                dcc.Link('Exploratory Analysis', href='/exploratory-analysis', style={'color': 'white'}),
                html.Br(),
                dcc.Link('Clustering and Dimension Reduction', href='/clustering-dimension-reduction', style={'color': 'white'}),
            ], style={'background-color': 'black', 'height': '100vh'}),
            width=3
        ),
        dbc.Col([
            html.H1(id='page-content-header', style={'color': 'white'}),
            html.Div(id='page-content', style={'background-color': 'blue', 'color': 'white', 'justify-content': 'center', 'align-items': 'center', 'height': '100vh'})
        ], width=9)
    ])
])


# Home Page Content
home_page_content = html.Div([
    html.H2('Welcome to the Home Page!', style={'text-align': 'center'}),
    html.P('This is the center-aligned content of the home page.'),
    html.P('You can customize this page as needed.'),
])
..
# Load Dataset Page Content
load_dataset_page_content = html.Div([
    html.H2('Load Dataset', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        multiple=False
    ),
    html.Button('Upload and Display', id='upload-button', n_clicks=0, style={'margin-top': '10px'}),
    dcc.Slider(
        id='rows-slider',
        min=1,
        max=100,
        step=1,
        marks={i: str(i) for i in range(1, 101)},
        value=10,
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),
    html.Div(id='output-data-upload'),
])

# Exploratory Analysis Page Content
exploratory_analysis_page_content = html.Div([
    html.H2('Exploratory Analysis', style={'text-align': 'center'}),

    # Missing Values
    html.Div([
        html.H3('Missing Values'),
        html.Div(id='missing-values-output')
    ]),

    # Categorical Bar Plots
    html.Div([
        html.H3('Categorical Bar Plots'),
        dcc.Dropdown(
            id='categorical-variable-dropdown',
            options=[
                {'label': col, 'value': col} for col in df.select_dtypes(include='object').columns
            ],
            multi=False,
            value=df.select_dtypes(include='object').columns[0]
        ),
        dcc.Graph(id='categorical-bar-plot')
    ]),

    # Embeddings Heatmap
    html.Div([
        html.H3('Embeddings Heatmap'),
        dcc.Graph(id='embeddings-heatmap')
    ]),

    # Scatter Plot of Embeddings
    html.Div([
        html.H3('Scatter Plot of Embeddings'),
        dcc.Graph(id='scatter-plot-embeddings')
    ]),
])
def display_dataset(df, selected_rows):
    # Filter and display selected number of rows
    df_display = df.head(selected_rows)

    # Display the dataset content
    return html.Div([
        html.H5('Dataset Content:'),
        dcc.Markdown(f'```\n{df_display.to_string(index=False)}\n```'),
    ])

def check_missing_values(df):
    missing_values_html = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        # Body
        [html.Tr([html.Td(df[col].isnull().sum()) for col in df.columns])]
    )
    return missing_values_html

def plot_categorical_bar(df, categorical_variable):
    bar_plot_fig = px.bar(df, x=categorical_variable, title=f'Bar Plot of {categorical_variable}')
    return bar_plot_fig

def plot_embeddings_heatmap(df):
    embeddings_heatmap_fig = sns.heatmap(df[['Embedding1', 'Embedding2']].corr(),
                                         annot=True,
                                         cmap='coolwarm',
                                         fmt=".2f",
                                         annot_kws={"size": 10},
                                         linewidths=.5,
                                         square=True).get_figure()
    plt.close(embeddings_heatmap_fig)
    return go.Figure(go.Heatmap(z=[[0, 0], [0, 0]], colorscale='Viridis'))

def plot_scatter_embeddings(df):
    scatter_plot_fig = px.scatter(df, x='Embedding1', y='Embedding2', title='Scatter Plot of Embeddings')
    return scatter_plot_fig

# Callback to update content on load dataset page
@app.callback([Output('output-data-upload', 'children')],
              Input('upload-button', 'n_clicks'),
              State('upload-data', 'contents'),
              State('rows-slider', 'value'))
def update_load_dataset_page(n_clicks, contents, selected_rows):
    if n_clicks > 0 and contents is not None:
        # Decode and read the uploaded file
        content_string = contents.split(',')[1]
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Display the dataset
        dataset_content = display_dataset(df, selected_rows)
        return dataset_content,

    return html.Div(),

# Callback to update content on exploratory analysis page
@app.callback([Output('missing-values-output', 'children'),
               Output('categorical-bar-plot', 'figure'),
               Output('embeddings-heatmap', 'figure'),
               Output('scatter-plot-embeddings', 'figure')],
              Input('upload-button', 'n_clicks'),
              State('upload-data', 'contents'),
              State('rows-slider', 'value'))
def update_exploratory_analysis(n_clicks, contents, selected_rows):
    if n_clicks > 0 and contents is not None:
        # Decode and read the uploaded file
        content_string = contents.split(',')[1]
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Use exploratory analysis functions
        missing_values_html = check_missing_values(df)
        categorical_variable = df.select_dtypes(include='object').columns[0]
        bar_plot_fig = plot_categorical_bar(df, categorical_variable)
        embeddings_heatmap_fig = plot_embeddings_heatmap(df)
        scatter_plot_fig = plot_scatter_embeddings(df)
        return missing_values_html, bar_plot_fig, embeddings_heatmap_fig, scatter_plot_fig

    return html.Div(), go.Figure(), go.Figure(), go.Figure()


@app.callback([Output('page-content', 'children'), Output('page-content-header', 'children')],
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/load-dataset':
        return load_dataset_page_content, 'Load Dataset'
    elif pathname == '/exploratory-analysis':
        return exploratory_analysis_page_content, 'Exploratory Analysis'
    elif pathname == '/clustering-dimension-reduction':
        return 'This is Clustering and Dimension Reduction content.', 'Clustering and Dimension Reduction'
    else:
        return home_page_content, 'Home'

if __name__ == '__main__':
    app.run_server(debug=True)