import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from services.data_fetcher import fetch_data
from services.model import prepare_data, train_model
from services.preditctor import predict_future
from services.resource_saver import save_results
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def launch_app():
    # Główne okno
    app = tk.Tk()
    app.title("Portfolio Optimizer")
    app.geometry("600x600")
    app.configure(bg="#e6f0ff")  # jasnoniebieskie tło

    # Stylizacja czcionek i stylów
    style = ttk.Style()
    style.configure("TLabel", font=("Segoe UI", 12), background="#e6f0ff")
    style.configure("TButton", font=("Segoe UI", 11), padding=6)
    style.configure("TEntry", font=("Segoe UI", 11))
    style.configure("TCombobox", font=("Segoe UI", 11))

    # Nagłówek
    header = ttk.Label(app, text="Portfolio Optimizer", font=("Segoe UI", 16, "bold"))
    header.pack(pady=10)

    # Pole na ticker
    ticker_label = ttk.Label(app, text="Podaj ticker (np. AAPL):")
    ticker_label.pack()
    ticker_entry = ttk.Entry(app, width=20)
    ticker_entry.pack(pady=5)

    # Dropdown do wyboru horyzontu
    forecast_label = ttk.Label(app, text="Wybierz horyzont czasowy predykcji (dni):")
    forecast_label.pack()
    forecast_options = [1, 5, 30, 60, 100]
    forecast_var = tk.IntVar(value=1)
    forecast_menu = ttk.Combobox(app, textvariable=forecast_var, values=forecast_options, state="readonly", width=10)
    forecast_menu.pack(pady=5)

    # Ramka na wykres
    chart_frame = ttk.LabelFrame(app, text="Wykres predykcji", padding=(10, 10))
    chart_frame.pack(padx=20, pady=15, fill="both", expand=True)

    # Puste płótno na wykres
    canvas_widget = None

    def run_prediction():
        nonlocal canvas_widget  # pozwala nadpisać poprzedni wykres

        ticker = ticker_entry.get().upper()
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

            # Tworzenie wykresu
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            ax.plot(df_close.index, df_close["Close"], label="Rzeczywiste", color="blue")
            ax.plot(predicted_dates, predicted_values, label=f"Predykcja ({days_forward} dni)", color="orange")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f"Predykcja cen dla {ticker}")
            ax.grid(True)
            ax.legend()

            # Usuwanie poprzedniego wykresu jeśli istnieje
            if canvas_widget:
                canvas_widget.get_tk_widget().destroy()

            # Osadzenie wykresu w GUI
            canvas_widget = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    # Przycisk do uruchomienia predykcji
    predict_button = ttk.Button(app, text="Uruchom predykcję", command=run_prediction)
    predict_button.pack(pady=10)

    app.mainloop()
