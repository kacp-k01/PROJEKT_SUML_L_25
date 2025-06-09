import numpy as np

def predict_future(model, X, y, scaler, forecast_days=1):
    predicted_scaled = model.predict(X)
    scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Ekstrapolacja predykcji o N dni
    last_input = X[-1]
    future_predictions = []
    current_input = last_input

    for _ in range(forecast_days):
        next_pred = model.predict(current_input.reshape(1, -1, 1))
        future_predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[1:], next_pred[0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten(), actual.flatten()