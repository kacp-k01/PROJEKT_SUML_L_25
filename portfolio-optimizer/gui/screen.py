import tkinter as tk
from tkinter import ttk, messagebox
from services.data_fetcher import fetch_data
from services.model import prepare_data, train_model
from services.preditctor import predict_future
from services.resource_saver import save_results
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def launch_app():
    app = tk.Tk()
    app.title("Portfolio Optimizer")
    app.geometry("600x450")

    ticker_label = ttk.Label(app, text="Podaj ticker (np. AAPL):")
    ticker_label.pack()

    ticker_entry = ttk.Entry(app)
    ticker_entry.pack()

    forecast_label = ttk.Label(app, text="Wybierz horyzont czasowy predykcji (dni):")
    forecast_label.pack()

    forecast_options = [1, 5, 30, 60, 100]
    forecast_var = tk.IntVar(value=1)
    forecast_menu = ttk.Combobox(app, textvariable=forecast_var, values=forecast_options, state="readonly")
    forecast_menu.pack()

    def run_prediction():
        ticker = ticker_entry.get()
        days_forward = forecast_var.get()

        if not ticker:
            messagebox.showerror("Błąd", "Musisz podać ticker!")
            return

        try:
            df = fetch_data(ticker)
            X, y, scaler, df_close = prepare_data(df)
            model = train_model(X, y)
            predicted_dates, predicted_values = predict_future(model, X, scaler, df_close.index[-1], days_forward)
            save_results(ticker, predicted_values)

            # Wykres z datami
            plt.figure(figsize=(10, 5))
            plt.plot(df_close.index, df_close["Close"], label="Rzeczywiste", color="blue")
            plt.plot(predicted_dates, predicted_values, label=f"Predykcja ({days_forward} dni)", color="orange")
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.xticks(rotation=45)
            plt.title(f"Predykcja cen dla {ticker} ($)")
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    predict_button = ttk.Button(app, text="Uruchom predykcję", command=run_prediction)
    predict_button.pack(pady=10)

    app.mainloop()
