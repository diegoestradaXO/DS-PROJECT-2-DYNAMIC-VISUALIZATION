import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import datetime
from dash.dependencies import Input, Output, State
from tensorflow import keras




myModel = keras.models.load_model("model.h5") 

# ================== App logic =======================
UVG_LOGO = 'https://altiplano.uvg.edu.gt/admisiones/images/logo_uvgadmin.png'
app = dash.Dash(
    external_stylesheets=[dbc.themes.DARKLY]
)

navbar = dbc.Navbar(
    dbc.Container(
        [
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=UVG_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Detección de COVID19 con radiografías del tórax", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
           
        ],
        
    ),
    color="primary",
    dark=True,
)

app.layout = html.Div([
    navbar,
    html.H1('Modelo Xception de Keras', style={'text-align':'center','margin-top':'10px'}),
    html.Div([ # Horizontal flexbox va aquí
        dcc.Upload( #Elemento para subir imagenes
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File', style={'text-decoration': 'underline'})
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'min-width':'216px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
        html.Div([
            html.H5('Tu imagen', style={'text-align':'center'}),
            html.Hr(),
            html.Div(id='output-image-upload',style={'max-width':'100%','max-height':'100%'}), #Mostrar la imagen subida
        ],style={'display':'flex','flex-direction':'column'}),
        html.Div([
            html.H5('prediction here!'),
            html.Hr()
        ])
        ],
         style={'width':'100%',
                'display':'flex',
                'flex-direction':'row',
                'flex-wrap':'wrap',
                'justify-content':'space-around',
                'margin-top':'50px'
         }
        )
    
])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == "__main__":
    app.run_server()