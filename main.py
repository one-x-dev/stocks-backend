import pandas as pd
import numpy as np
import tensorflow as tf

import get_prices as hist
from preprocessing import DataProcessing


import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
fix.pdr_override()

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def return_data(name, start, end):
    data = pdr.get_data_yahoo(name, start, end)
    temp = []
    for index, row in data.iterrows():
      temp.append({
          'Date': index,
          "High": row['High'],
          "Low": row['Low'],
          "Open": row['Open'],
          "Close": row['Close'],
          "Volume": row['Volume'],
          "Adj Close": row['Adj Close']
      })
    return temp

def return_predict(name, start_predict, end_predict):
    start = "2006-01-01"
    end = "2020-01-01"

    hist.get_stock_data(name, start_date=start, end_date=end)
    process = DataProcessing("stock_prices.csv", 0.9)
    process.gen_test(10)
    process.gen_train(10)

    X_train = process.X_train.reshape((process.X_train.shape[0], 10, 1)) / 200
    Y_train = process.Y_train / 200

    X_test = process.X_test.reshape(process.X_test.shape[0], 10, 1) / 200
    Y_test = process.Y_test / 200

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(10, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, Y_train, epochs=50)

    print(model.evaluate(X_test, Y_test))

    data = pdr.get_data_yahoo(name, start_predict, end_predict)
    stock = data["Adj Close"]
    X_predict = np.array(stock).reshape((1, np.array(stock).shape[0], 1)) / 200

    result = model.predict(X_predict)*200;
    print(result)
    return { 'predict': str(result[0][0]) }

@app.route('/predict')
def predict():
    data = request.args
    start = data['start']
    end = data['end']
    name = data['name']

    temp = return_predict(name, start, end)
    print(temp)
    return jsonify(temp)

@app.route('/test')
def test():
    data = request.args
    start = data['start']
    end = data['end']
    name = data['name']
    temp = return_data(name, start, end)
    return jsonify(temp)
    
if __name__ == '__main__':
    app.run()
