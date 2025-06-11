from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tf_keras import Sequential
from tf_keras.src.layers import LSTM, Dense


def prepare_data(df, look_back=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    df_close = df.iloc[look_back:]  # odpowiadający zakres dla y

    return X.reshape((X.shape[0], X.shape[1], 1)), y, scaler, df_close

def train_model(X, y, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history