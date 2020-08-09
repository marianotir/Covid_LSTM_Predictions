# -*- coding: utf-8 -*-
"""
Script that loads the data, make predictions 
and plots them into a website

@author: mariano
"""

#-------------------------------
# Import libraries
#-------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import runpy
import settings


#--------------------------------
# Run prevopus scripts
#-------------------------------

# Get user name defined in settings
User_Name = settings.User_Name

# Run scripts

print('')
print('Load data')

runpy.run_path(path_name="C:/Users/{}/Covid/Load_Data.py".format(User_Name))

print('')
print('Train LSTM model')
print('')

runpy.run_path(path_name="C:/Users/{}/Covid/Pred_LSTM.py".format(User_Name))

print('')


#---------------------------
# Load data 
#---------------------------

# From cvs
df = pd.read_csv("C:/Users/{}/Covid/Data/Pred_Covid.csv".format(User_Name))


#-----------------------------------------
# Create a dash website application 
#-----------------------------------------

Country = settings.Country
Cum_Cases = settings.Cum_Sum_Days

children_text_H1 = "Covid Prediction in Country: {}".format(Country)
subtitle = "Covid 19 prediction using LSTM networks. Prediction of {} days cumulative cases.".format(Cum_Cases)
title1 = "Covid 19 prediction for last updated day: {}".format(df.Date[0])
title2 = "Covid 19 prediction for day: {}".format(df.Tomorrow_Date[0])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children=children_text_H1),

    html.Div(children= 
        subtitle
    ),

    dcc.Graph(
        id='graph1',
        figure={
            'data': [
                {'x': [1], 'y': [df.Predicted_Point[0]], 'type': 'bar', 'name': 'Prediction'},
                {'x': [2], 'y': [df.Real_Point[0]], 'type': 'bar', 'name': u'Real'},
            ],
            'layout': {
                'title': title1 
            }
        }
    ),

    dcc.Graph(
        id='graph2',
        figure={
            'data': [
                {'x': [1], 'y': [df.Pred_Tomorrow[0]], 'type': 'bar', 'name': 'Prediction_Tomorrow'}
            ],
            'layout': {
                'title': title2 
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=False,use_reloader=False)
    

