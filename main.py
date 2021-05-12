from flask_cors import CORS
from flask import request
from flask import jsonify
from flask import Flask
import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime, timedelta

import get_prices as hist
from preprocessing import DataProcessing


import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
fix.pdr_override()


app = Flask(__name__)
CORS(app)


def return_predict(name, today):
    start = "2014-01-01"

    today = datetime(
        int(today.split('-')[0]), int(today.split('-')
                                      [1]), int(today.split('-')[2])
    )
    train_month = today - timedelta(days=30)
    predict_week = today + timedelta(days=7)

    hist.get_stock_data(name, start_date=start, end_date=train_month.date())
    process = DataProcessing("stock_prices.csv", 0.9)
    process.gen_test(10)
    process.gen_train(10)

    X_train = process.X_train.reshape((process.X_train.shape[0], 10, 1)) / 200
    Y_train = process.Y_train / 200

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        20, input_shape=(10, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, Y_train, epochs=100)

    data = pd.read_csv('stock_prices.csv')

    result = []

    date = today

    for i in range(1, 8):
        stock = data["Adj Close"][-30:]

        X_predict = np.array(stock).reshape((1, stock.shape[0], 1)) / 200

        predict = model.predict(X_predict)*200

        date = today + timedelta(days=i)

        data = data.append({"Date": str(date.date()), "Adj Close":
                            float(predict[0])}, ignore_index=True)

    data.to_csv('final_file.csv', index=False)

    for index, row in data[-21:].iterrows():
        result.append({
            'date': row["Date"],
            'predict': row["Adj Close"]
        })
    return result


@app.route('/predict')
def predict():
    data = request.args
    name = data["name"]
    today = data['today']

    temp = return_predict(name, today)
    
    return jsonify(temp)


if __name__ == '__main__':
    app.run()
