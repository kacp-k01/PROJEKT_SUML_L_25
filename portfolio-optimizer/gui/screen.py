import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from services.data_fetcher import fetch_data as fet
from services.model import prepare_data, train_model
from services.preditctor import predict_future
from services.resource_saver import save_results

def launch_app():
    app = tk.Tk()
    app.title("Portfolio Optimizer")
    app.geometry("600x400")

    ticker_label = ttk.Label(app, text="Podaj ticker (np. AAPL):")
    ticker_label.pack()

    ticker_entry = ttk.Entry(app)
    ticker_entry.pack()

    def run_prediction():
        ticker = ticker_entry.get()
        if not ticker:
            messagebox.showerror("Błąd", "Musisz podać ticker!")
            return
        try:
            df = fet(ticker)
            X, y, scaler = prepare_data(df)
            model = train_model(X, y)
            predicted, actual = predict_future(model, X, y, scaler)
            save_results(ticker, predicted)

            # Wykres
            plt.figure(figsize=(10, 5))
            plt.plot(actual, label="Rzeczywiste")
            plt.plot(predicted, label="Predykcja")
            plt.legend()
            plt.title(f"Predykcja cen dla {ticker}")
            plt.show()

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    predict_button = ttk.Button(app, text="Uruchom predykcję", command=run_prediction)
    predict_button.pack(pady=10)

    app.mainloop()
