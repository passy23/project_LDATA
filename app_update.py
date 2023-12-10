import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the datasets
df1 = pd.read_csv('celeba_buffalo_l.csv')
df2 = pd.read_csv('celeba_buffalo_s.csv')


# Initialize the Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Face Recognition Latent Space Analysis Dashboard"),

    # Dropdown to select dataset
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Dataset 1', 'value': 'dataset1'},
            {'label': 'Dataset 2', 'value': 'dataset2'},
        ],
        value='dataset1',
        style={'width': '50%'}
    ),

    # Scatter plot for latent space analysis
    dcc.Graph(id='latent-space-scatter'),

    # Dropdown to select X-axis variable
    dcc.Dropdown(
        id='x-variable-dropdown',
        style={'width': '50%'},
        placeholder="Select embedding 1 variable"
    ),

    # Dropdown to select Y-axis variable
    dcc.Dropdown(
        id='y-variable-dropdown',
        style={'width': '50%'},
        placeholder="Select embedding 1 variable"
    ),
    # Bar plot for categorical variable
    dcc.Graph(id='categorical-bar-plot'),

    # Dropdown to select categorical variable for bar plot
    dcc.Dropdown(
        id='categorical-variable-dropdown',
        style={'width': '50%'},
        placeholder="Select categorical variable to plot"
    ),

    # Correlation heatmap
    dcc.Graph(id='correlation-heatmap'),

    # Dropdown for variable selection for heatmap
    dcc.Dropdown(
        id='heatmap-variable-dropdown',
        options=[],  # Will be populated dynamically
        multi=True,  # Allow multiple variable selection
        style={'width': '50%'}
    ),

])

# Define callback to update dropdown options based on selected dataset
@app.callback(
     [Output('x-variable-dropdown', 'options'),
     Output('y-variable-dropdown', 'options'),
     Output('categorical-variable-dropdown', 'options'),
     Output('heatmap-variable-dropdown', 'options')],
    [Input('dataset-dropdown', 'value')]
)
def update_dropdown_options(selected_dataset):
    if selected_dataset == 'dataset1':
        df = df1
    elif selected_dataset == 'dataset2':
        df = df2
  
    # Get the list of available numeric variables
    categorical_variable = [{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns]
    scatter_variable = [{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns]
    heatmap_variable = [{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns]

    return categorical_variable, scatter_variable, scatter_variable, heatmap_variable

# Define callback to update scatter plot based on selected dataset and variables
@app.callback(
    Output('latent-space-scatter', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('x-variable-dropdown', 'value'),
     Input('y-variable-dropdown', 'value')]
)
def update_scatter_plot(selected_dataset, x_variable, y_variable):
    if selected_dataset == 'dataset1':
        df = df1
    elif selected_dataset == 'dataset2':
        df = df2
  
    # Check if selected variables are numeric
    if x_variable not in df.select_dtypes(include='number').columns or \
            y_variable not in df.select_dtypes(include='number').columns:
        # Handle non-numeric variables
        return px.scatter(title='Select embeddings variables for the scatter plot')

    # Create scatter plot
    fig = px.scatter(df, x=x_variable, y=y_variable, color='Big_Nose', size_max=10,
                     labels={x_variable: f'{x_variable}', y_variable: f'{y_variable}'},
                     title='Scatter plot')

    return fig

# Define callback to update bar plot based on selected dataset and categorical variable
@app.callback(
    Output('categorical-bar-plot', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('categorical-variable-dropdown', 'value')]
)
def update_categorical_bar_plot(selected_dataset, categorical_variable):
    if selected_dataset == 'dataset1':
        df = df1
    elif selected_dataset == 'dataset2':
        df = df2
    
    # Check if selected variable is categorical
    if categorical_variable not in df.select_dtypes(include='number').columns:
        # Handle non-categorical variables
        return px.bar(title='BAR PLOT: Select variable to plot')

    # Create bar plot
    fig = px.bar(df, x=categorical_variable, color='Big_Nose', title='Categorical Variable Analysis')
    fig.show()
    return fig

# Define callback to update correlation heatmap based on selected dataset and variables
@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('heatmap-variable-dropdown', 'value')]
)
def update_correlation_heatmap(selected_dataset, selected_variables):
    if selected_dataset == 'dataset1':
        df = df1
    elif selected_dataset == 'dataset2':
        df = df2  
  

     # Handle the case where selected_variables is None or empty
    if not selected_variables:
        # Provide a default correlation heatmap
        default_heatmap = go.Figure(data=go.Heatmap(z=[[0]], colorscale='Viridis'))
        default_heatmap.update_layout(title='Correlation Heatmap : Select variables')
        return default_heatmap

    # Subset the DataFrame based on selected variables
    subset_df = df[selected_variables]

    # Create a correlation matrix
    correlation_matrix = subset_df.corr()

    # Create the correlation heatmap
    heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))
    heatmap.update_traces(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index)
    heatmap.update_layout(title='Correlation Heatmap')

    return heatmap

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)