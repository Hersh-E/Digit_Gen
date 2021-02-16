
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import load_model
import pathlib
import tensorflow as tf
from keras import backend as K

dir(K)

from funcs.latent_gens import generate_latent_points

DATA_PATH = pathlib.Path(__file__).parent.joinpath("Models").resolve()

# global gen_model, upc_model, sess, graph

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session(graph = graph)
K.set_session(sess)

# Initialize all variables
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)

gen_model = load_model(DATA_PATH.joinpath('generator_model2_100.h5'), compile=False)
upc_model = load_model(DATA_PATH.joinpath('upc_model.h5'), compile=False)

def gen_image(noise):#, gen_model=gen_model, upc_model=upc_model):
	global gen_model, upc_model, sess, graph
	with sess.as_default():
		with graph.as_default():
			number = gen_model.predict(noise)
	number = number * 255
	with sess.as_default():
		with graph.as_default():
			big_number = upc_model.predict(number).reshape(56,56)
	return big_number

control1 = dbc.Card([
	dbc.FormGroup([
		dbc.Label("Click to Generate Random Inputs"),
		html.Button('Randomness Generator', id='button_gen1'),
	]),
], body=True)

app = dash.Dash("Digit Gen", external_stylesheets=[dbc.themes.SLATE])

server = app.server

app.layout = dbc.Container([
	dbc.Row([

		dbc.Col([
			dcc.Graph(
				id='randomness_plot',
				style={"margin": "15px"},

			),
		# ], md='4'),
		# dbc.Col([
			dcc.Graph(
				id='generated_number',
				style={"margin": "15px"},
			),

		], md='8'),

		dbc.Col([
			control1,
		], md='4'),

	], align='center'),
	# dbc.Row([
	#
	#     dbc.Col([
	#         # control2,
	#     ], md='3'),
	#
	#     dbc.Col([
	#     ], md='9'),
	#
	# ], align='center'),

	# Hidden div inside the app that stores the intermediate value
	html.Div(id='generated_values', style={'display': 'none'})
], fluid=True)

@app.callback(
	Output('generated_values', 'children'),
	[Input('button_gen1', 'n_clicks')],
)
def generate_values(n_clicks):
	 # some expensive clean data step
	x = generate_latent_points(100, 1)
	x2 = pd.DataFrame(x[0], columns=['rand_values'])
	return x2.to_json(date_format='iso', orient='split')

@app.callback(
	Output('randomness_plot', 'figure'),
	[Input('generated_values', 'children')]
)
def clean_data(jsonified_cleaned_data):
	 # some expensive clean data step
	dff = pd.read_json(jsonified_cleaned_data, orient='split').reset_index()

	fig = px.bar(dff, x="index", y="rand_values", labels={'index':'','rand_values':''})
	# Set custom x-axis labels
	fig.update_xaxes(ticktext=[], tickvals=[],)

	return fig

@app.callback(
	Output('generated_number', 'figure'),
	[Input('generated_values', 'children')]
)
def genImage(jsonified_cleaned_data):
	dff = pd.read_json(jsonified_cleaned_data, orient='split')

	big_number = gen_image(dff.T.to_numpy())

	fig = px.imshow(big_number, color_continuous_scale='gray_r')

	return fig

if __name__ == '__main__':

	app.run_server(debug=True, processes=1, threaded=True, host='127.0.0.1',port=8050, use_reloader=False)
	# app.run_server
