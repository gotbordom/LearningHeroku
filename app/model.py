import sqlite3
from tabulate import tabulate
from flask import Flask, render_template
from werkzeug.exceptions import abort
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

#def get_db_connection():
#    conn = sqlite3.connect('database.db')
#    conn.row_factory = sqlite3.Row
#    return conn
#
#def get_prediction(prediction_id):
#    conn = get_db_connection()
#    prediction = conn.execute('SELECT * FROM predictions').fetchall()
#    conn.close()
#    if prediction is None:
#        abort(404)
#    return prediction
#
app = Flask(__name__)
#
#@app.route('/')
#def index():
#    conn = get_db_connection()
#    predictions = conn.execute('SELECT * FROM predictions').fetchall()
#    conn.close()
#    if prediction is None:
#       abort(404)
#    return prediction
#
#@app.route('/prediction_id')
#def prediction(prediction_id):
#    prediction = get_prediction(prediction_id)
#    return render_template('prediction.html', prediction=predictions)
#

 #model = lr.proba
#with open('model.pkl', 'wb') as f:
# Pickle the 'data' dictionary using the highest protocol available.
#with open('model.pkl', 'wb') as f:
#    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

#with open('model.pkl', 'rb') as g:
#     pickle.load(g)

#import tabulate
#table = tabulate.table(g)
data = pd.read_csv("https://github.com/Nick-Milliken/deploy-ml/raw/main/ramen/ramen-ratings.csv")

X = data.Stars

X = X.replace('Unrated', np.NaN)
X = pd.to_numeric(X)
X = np.nan_to_num(X)
X = X.reshape(-1, 1)

y = data.Country

X, y = X, y
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('lr', LogisticRegression())])
pipe.score(X_test, y_test)

lr = LogisticRegression(random_state = 42)
lr.fit = lr.fit(X_train, y_train)
lr.fit = lr.predict(X)
lr.proba = lr.predict_proba(X)
#print(lr.fit)
#print(lr.proba)


@app.route('/')
def index():
    return "<h1>Welcome to my NightMare </h1>"
#     return render_template("../index.html")

@app.route('/table')
def table():
    model = lr.proba
    model = tabulate(model, tablefmt='html')
    print(model) 

