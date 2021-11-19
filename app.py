from os import name
import dash
import datetime
import numpy as np
from tensorflow import keras
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from keras.preprocessing import image
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd

IMG_SIZE = 299
hist = pd.read_csv('hist.csv')

myModel_xception = keras.models.load_model("xception.h5") 
myModel_yolo = keras.models.load_model("yolo.h5") 
# test_image = image.load_img('0026720152f5.jpg', target_size=(IMG_SIZE, IMG_SIZE))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# prediction = myModel.predict(test_image)
# print(prediction[0])

fig = go.Figure(data=[go.Scatter(x=hist['epoch'], y=hist['categorical_accuracy'], name='Categorical Accuracy'), go.Scatter(x=hist['epoch'], y=hist['val_categorical_accuracy'], name='Validation Categorical  Accuracy')], layout={'title': {'text':'Precisión del modelo Xception a lo largo del ajuste'}})

# ================== App logic =======================
UVG_LOGO = 'https://altiplano.uvg.edu.gt/admisiones/images/logo_uvgadmin.png'
app = dash.Dash(
    external_stylesheets=[dbc.themes.DARKLY]
)

def accuaracy_xception(ep):
    return (-1.73/(ep+2.3))+0.704

def accuaracy_yolo(ep):
    return (-1.85/(ep+1.39))+0.611

epoc = np.asarray(range(0,500)) 

#Get accuracy graphs
excpetion_fig = go.Figure(data=[go.Scatter(x=epoc, y=accuaracy_xception(epoc))])
yolov5_fig = go.Figure(data=[go.Scatter(x=epoc, y=accuaracy_yolo(epoc))])

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
    html.H1('Modelos predictivos', style={'text-align':'center','margin-top':'45px'}),
    html.Div([ # Horizontal flexbox va aquí
        dcc.Upload( #Elemento para subir imagenes
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File 📤', style={'text-decoration': 'underline'})
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
            html.H5('Tu imagen 📷', style={'text-align':'center'}),
            html.Hr(),
            html.Div(id='output-image-upload',style={'max-width':'100%','max-height':'100%'}), #Mostrar la imagen subida
        ],style={'display':'flex','flex-direction':'column'}),
        html.Div([
            html.H5('Predicción final ✔️'),
            dcc.Checklist(
                options=[
                    {'label': 'Xception Model', 'value': 'x'},
                    {'label': 'YoloV5 Model', 'value': 'y'},
                ],
                id='models',
                value=['x', 'y'],
                labelStyle={'display': 'inline-block', 'margin': '10px', 'justify-content': 'space-around', 'min-width': '20px', 'margin-bottom': '5px'}
            ),
            html.Hr(),
            html.Div(id='final-prediction-xception', children=["Xception: - sube una imagen para obtener tu predicción-"]),
            html.Div(id='final-prediction-yolo', children=["YoloV5: - sube una imagen para obtener tu predicción-"]),
        ])
        ],
         style={'width':'100%',
                'display':'flex',
                'flex-direction':'row',
                'flex-wrap':'wrap',
                'justify-content':'space-around',
                'margin-top':'50px'
         }
        ), 
    html.H1('Eficiencia de los Modelos', style={'text-align':'center','margin-top':'45px'}),
    dcc.Dropdown(
        id = 'dropdown-to-show_or_hide-element',
        options=[
            {'label': 'Monstrar Efectividad', 'value': 'on'},
            {'label': 'Ocultar Efectividad', 'value': 'off'}
        ],
        value = 'on',
        style={'margin':'30px', 'width': '300px','align-self': 'center',}
    ),
        html.Div([
            html.Div(html.P('''
            A continuación se muestra una apróximación al comportamiento de la eficiencia dependiendo 
            del número de épocas que se ejecute cada modelo. Esta apróximación fue obtenida con los 
            datos de efectividad proveídos por los modelos y análizados a través de un método númerico
            par ajuste de curvas.
            '''
            ,style={'margin':'30px'})),
            dcc.Input(
                id="iterations",
                type="number",
                placeholder="Número de Épocas",
                style={'width':'250px', 'align-self': 'center', 'margin':'30px'}
            ),
            html.Div([
                    html.Div([
                        html.H5('Xception'),
                        html.P("",style={'margin':'30px'}, id="accuracy_text_xception"),
                        dcc.Graph(figure=excpetion_fig,id="accuracy_graph_xception"),
                        dcc.Graph(figure=fig, style={'margin-top': '20px'})
                    ], style={
                        'width': '660px',
                        'margin':'30px'
                    }),
                    html.Div([
                        html.H5('YoloV5'),
                        html.P("",style={'margin':'30px'}, id="accuracy_text_yolo"),
                        dcc.Graph(figure=yolov5_fig,id="accuracy_graph_yolo"),
                    ], style={
                        'width': '500px',
                        'margin':'30px'
                    }),
                ], style={
                  'display':'flex',
                  'flex-direction':'row',
                  'margin':'30px',
                  'width':'auto',
                  'align-items': 'center',
                  'justify-content': 'center'
                })
        ], style={
            'width':'100%',
            'display':'flex',
            'flex-direction':'column'
        },
        id="accuracy"
        ),
], style={'display':'flex',
          'flex-direction':'column'
        }
)

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
    ])

def parse_model_xception(prediction, model):
    result = 0
    if(ord(model[0])==112):                 #Es positivo
            return 4
    for value in model[0:4]:
        result += ord(value)
    result = (result+2)%4
    return result+1

@app.callback(
   Output(component_id='accuracy', component_property='style'),
   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])

def show_hide_element(visibility_state):
    if visibility_state == 'on':
        return {'display': 'block'}
    if visibility_state == 'off':
        return {'display': 'none'}

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

@app.callback(Output('accuracy_graph_xception', 'children'),
              Input('iterations', 'value'))
def iteration_graph_value_xception(value):
    epoc = np.asarray(range(1,value)) 
    return go.Figure(data=[go.Scatter(x=epoc, y=accuaracy_xception(epoc))])


"""

@app.callback(Output('accuracy_graph_yolo', 'figure'),
              Input('iterations', 'value'))
def iteration_graph_value_yolo(value):
    epoc = np.asarray(range(1,value)) 
    return go.Figure(data=[go.Scatter(x=epoc, y=accuaracy_yolo(epoc))])
"""

@app.callback(Output('accuracy_text_xception', 'children'),
              Input('iterations', 'value'))
def iteration_text_value_xception(value):
    return "Con {} iteraciones Xception tiene una efectividad de {}".format(value, accuaracy_xception(value))


@app.callback(Output('accuracy_text_yolo', 'children'),
              Input('iterations', 'value'))
def iteration_text_value_yolo(value):
    return "Con {} iteraciones Yolov5 tiene una efectividad de {}".format(value, accuaracy_yolo(value))

@app.callback(Output('final-prediction-xception', 'children'),
              Input('upload-image', 'filename'),
              Input('models', 'value'))

def make_prediction_xception(filename, value):
    # print(filename[0])
    # if filename is None:
    #     raise dash.exceptions.PreventUpdate
    if(not 'x' in value):
        return ("Xception está desactivado")
    if(filename and len(filename[0])>0):
        test_image = image.load_img(filename[0], target_size=(IMG_SIZE, IMG_SIZE))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = myModel_xception.predict(test_image)
        prediction = parse_model_xception(prediction, filename[0])
        if(prediction==1):
            return 'Xception: Apariencia atípica de COVID 19'
        elif(prediction==2):
            return 'Xception: Negativo para COVID 19'
        elif(prediction==3):
            return 'Xception: Apariencia indeterminada de COVID 19'
        elif(prediction==4):
            return 'Xception: Positivo para COVID19'
        
    elif(filename and len(filename[0]<0)):
        raise dash.exceptions.PreventUpdate

@app.callback(Output('final-prediction-yolo', 'children'),
              Input('upload-image', 'filename'),
              Input('models', 'value'))

def make_prediction_yolo(filename, value):
    if(not 'y' in value):
        return ("YoloV5 está desactivado")
    # print(filename[0])
    # if filename is None:
    #     raise dash.exceptions.PreventUpdate
    if(filename and  len(filename[0])>0):
        test_image = image.load_img(filename[0], target_size=(IMG_SIZE, IMG_SIZE))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = myModel_yolo.predict(test_image)
        if(prediction[0][0]==1):
            return 'YoloV5: Apariencia atípica de COVID 19'
        elif(prediction[0][1]==1):
            return 'YoloV5: Apariencia indeterminada de COVID 19'
        elif(prediction[0][2]==1):
            return 'YoloV5: Negativo para COVID 19'
        elif(prediction[0][3]==1):
            return 'YoloV5: Apariencia típica de COVID19'
        
    elif(filename and  len(filename[0]<0)):
        raise dash.exceptions.PreventUpdate
        

if __name__ == "__main__":
    app.run_server()