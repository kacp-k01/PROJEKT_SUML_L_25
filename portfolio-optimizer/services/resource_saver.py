import json
import os

def save_results(ticker, predictions, output_dir="resources"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{ticker}_predictions.json")
    with open(filename, "w") as f:
        json.dump({"ticker": ticker, "predictions": predictions.tolist()}, f)
