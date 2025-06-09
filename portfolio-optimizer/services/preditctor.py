def predict_future(model, X, y, scaler):
    predicted_scaled = model.predict(X)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    return predicted.flatten(), actual.flatten()