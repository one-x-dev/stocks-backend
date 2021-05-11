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

def return_predict():
    start = "2005-12-01"
    end = "2020-12-01"

    hist.get_stock_data("AAPL", start_date=start, end_date=end)
    process = DataProcessing("stock_prices.csv", 0.9)
    process.gen_test(10)
    process.gen_train(10)

    X_train = process.X_train.reshape((3379, 10, 1)) / 200
    Y_train = process.Y_train / 200

    X_test = process.X_test.reshape(359, 10, 1) / 200
    Y_test = process.Y_test / 200

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(10, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, Y_train, epochs=100)
    new_data = []
    time_data = []
    print(model.evaluate(X_test, Y_test))
    first_day = 5
    last_day = 19
    for i in range(7):
        data1 = "2021-01-" + str(first_day + i)
        data2 = "2021-02-" + str(last_day + i if 1 + i >= 10 else "0" + str(2 + i))
        time = "2021-01-" + str(last_day + i + 1 if last_day+ i + 1 >= 10 else "0" + str(last_day + i + 1))
        time_data.append(time)
        data = pdr.get_data_yahoo("AAPL", data1, data2)
        print(data1, data2)
        stock = data["Adj Close"]
        print(stock.shape)
        X_predict = np.array(stock).reshape((1, stock.shape[0], 1)) / 200
        s = model.predict(X_predict)*200
        print(s[0])
        new_data.append(s[0])
    result = []
    for i in range(7):
        print(i, new_data[i][0])
        result.append({ "predict" : float(new_data[i][0]), "date" : time_data[i] })
    return result

@app.route('/predict')
def predict():
    data = request.args
    
    temp = return_predict()
    print(temp)
    return jsonify(temp)

@app.route('/test')
def test():
    data = request.args
    start = data['start']
    end = data['end']
    name = data['name']
    print("test end: ", end)
    temp = return_data(name, start, end)
    return jsonify(temp)
    
if __name__ == '__main__':
    app.run()
