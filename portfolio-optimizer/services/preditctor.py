import numpy as np
import pandas as pd


def predict_future(model, X, scaler, last_date, forecast_days=1):
    # Ekstrapolacja predykcji o N dni
    last_input = X[-1]
    future_predictions = []
    current_input = last_input

    for _ in range(forecast_days):
        next_pred = model.predict(current_input.reshape(1, -1, 1), verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[1:], next_pred[0, 0])

    # Odskalowanie i tworzenie zakresu dat
    predicted_values = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    predicted_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')  # B = dni robocze

    return predicted_dates, predicted_values