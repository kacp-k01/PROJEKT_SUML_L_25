import tkinter as tk
from tkinter import ttk, messagebox

from tkhtmlview import HTMLScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from services.data_fetcher import fetch_data, fetch_company_info
from services.model import prepare_data, train_model
from services.preditctor import predict_future
from services.resource_saver import save_report_and_plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def launch_app():
    app = tk.Tk()
    app.report_html = None
    app.title("Portfolio Optimizer")
    app.geometry("1400x700")
    app.configure(bg="#e6f0ff")

    style = ttk.Style()
    style.configure("TLabel", font=("Segoe UI", 11, "bold"), background="#e6f0ff", foreground="#333333")
    style.configure("TButton", font=("Segoe UI", 11), padding=6)
    style.configure("TEntry", font=("Segoe UI", 11))
    style.configure("TCombobox", font=("Segoe UI", 11))

    # Layout
    main_frame = tk.Frame(app, bg="#e6f0ff")
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    left_panel = tk.Frame(main_frame, width=250, bg="#e6f0ff")
    left_panel.pack(side="left", fill="y", padx=10)

    center_panel = ttk.LabelFrame(main_frame, text="Wizualizacja predykcji", padding=(10, 10))
    center_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    right_panel = ttk.LabelFrame(main_frame, text="Raport", padding=(10, 10), width=350)
    right_panel.pack(side="right", fill="y", padx=10, pady=5)
    right_panel.pack_propagate(False)

    # ========== POLA WYBORU ==========
    ttk.Label(left_panel, text="Ticker:").pack(anchor="w", pady=(10, 0))
    tk.Label(left_panel, text="Oznaczenie danego instrumentu finansowego.",
             font=("Segoe UI", 8), fg="#333333", bg="#e6f0ff", wraplength=200, justify="left").pack(anchor="w")
    ticker_entry = ttk.Entry(left_panel)
    ticker_entry.pack(fill="x", pady=5)

    ttk.Label(left_panel, text="Horyzont predykcji (dni):").pack(anchor="w", pady=(10, 0))
    forecast_var = tk.IntVar(value=30)
    forecast_menu = ttk.Combobox(left_panel, textvariable=forecast_var, values=[1, 5, 30, 60, 100], state="readonly")
    forecast_menu.pack(fill="x", pady=5)

    ttk.Label(left_panel, text="Epoki treningu:").pack(anchor="w", pady=(10, 0))
    tk.Label(left_panel, text="Liczba epok – ile razy model przejrzy dane. Zbyt mało - niedotrenowanie. Zbyt dużo - przetrenowanie.",
             font=("Segoe UI", 8), fg="#333333", bg="#e6f0ff", wraplength=200, justify="left").pack(anchor="w")
    epochs_var = tk.IntVar(value=10)
    epochs_entry = ttk.Entry(left_panel, textvariable=epochs_var)
    epochs_entry.pack(fill="x", pady=5)

    ttk.Label(left_panel, text="Wielkość grupy:").pack(anchor="w", pady=(10, 0))
    tk.Label(left_panel, text="Wielkość grupy danych (batch) przy uczeniu. Im większa, tym większe tempo uczenia modelu i mniejsza dokładność.",
             font=("Segoe UI", 8), fg="#333333", bg="#e6f0ff", wraplength=200, justify="left").pack(anchor="w")
    batch_var = tk.IntVar(value=32)
    batch_entry = ttk.Entry(left_panel, textvariable=batch_var)
    batch_entry.pack(fill="x", pady=5)

    # ========== PRZYCISK ==========
    def run_prediction():
        ticker = ticker_entry.get().upper()
        days_forward = forecast_var.get()
        epochs = epochs_var.get()
        batch_size = batch_var.get()

        if not ticker:
            messagebox.showerror("Błąd", "Musisz podać ticker!")
            return

        try:
            df = fetch_data(ticker)
            info = fetch_company_info(ticker)
            X, y, scaler, df_close = prepare_data(df)
            model, history = train_model(X, y, epochs=epochs, batch_size=batch_size)
            predicted_dates, predicted_values = predict_future(model, X, scaler, df_close.index[-1], days_forward)
            predicted_train = model.predict(X, verbose=0)
            predicted_train_inv = scaler.inverse_transform(predicted_train)
            y_inv = scaler.inverse_transform(y.reshape(-1, 1))

            mse = mean_squared_error(y_inv, predicted_train_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_inv, predicted_train_inv)

            company_name = info.get("longName", "Brak nazwy")
            industry = info.get("industry", "Brak branży")
            country = info.get("country", "Brak kraju")
            currency = info.get("currency", "Brak podanej waluty")
            previous_close = info.get("previousClose", "N/A")
            current_price = info.get("currentPrice", "N/A")
            # ========== RAPORT ==========
            report_text = f"""
                       <h3 style='margin-bottom: 10px;'>Informacje o spółce</h3>
                       <b>Spółka:</b> {company_name}<br>
                       <b>Branża:</b> {industry}<br>
                       <b>Kraj:</b> {country}<br>
                       <b>Waluta:</b> {currency}<br><br>

                       <h3 style='margin-bottom: 10px;'>Parametry predykcji</h3>
                       <b>Ticker:</b> {ticker}<br>
                       <b>Epoki treningowe:</b> {epochs}<br>
                       <b>Wielkość grupy:</b> {batch_size}<br>
                       <b>Dni do przodu:</b> {days_forward}<br>
                       <b>Liczba rekordów:</b> {len(df)}<br>
                       <b>Aktualna cena:</b> {current_price}<br>
                       <b>Poprzednie zamknięcie:</b> {previous_close}<br><br>

                       <h3 style='margin-bottom: 10px;'>Wyniki modelu</h3>
                       <b>Strata końcowa (loss):</b> {round(history.history['loss'][-1], 6)}<br>
                       <b>MSE:</b> {mse:.2f}<br>
                       <b>RMSE:</b> {rmse:.2f}{currency}<br>
                       <b>MAE:</b> {mae:.2f}{currency}
                       """

            raport_pure_text = (f"""\
Informacje o spółce:
Spółka: {company_name}
Branża: {industry}
Kraj: {country}
Waluta: {currency}

Parametry predykcji:
Ticker: {ticker}
Epoki treningowe: {epochs}
Wielkość grupy: {batch_size}
Dni do przodu: {days_forward}
Liczba rekordów: {len(df)}
Aktualna cena: {current_price}
Poprzednie zamknięcie: {previous_close}

Wyniki modelu
Strata końcowa (loss): {round(history.history['loss'][-1], 6)}
MSE: {mse:.2f}
RMSE: {rmse:.2f}{currency}
MAE: {mae:.2f}{currency}
""")

            # ========== RAPORT BOX ==========
            if app.report_html:
                app.report_html.set_html(report_text)
            else:
                app.report_html = HTMLScrolledText(right_panel, html="", width=40)
                app.report_html.pack(fill="both", expand=True)
                app.report_html.configure(background="#f8faff", font=("Segoe UI", 10))
                app.report_html.set_html(report_text)

            # ========== WYKRES ==========
            fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
            ax.plot(df_close.index, df_close["Close"], label="Rzeczywiste", color="blue")
            ax.plot(predicted_dates, predicted_values, label=f"Predykcja ({days_forward} dni)", color="orange")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f"Predykcja cen dla {ticker}")
            ax.grid(True)
            ax.legend()

            if hasattr(app, 'canvas_widget') and app.canvas_widget:
                app.canvas_widget.get_tk_widget().destroy()

            app.canvas_widget = FigureCanvasTkAgg(fig, master=center_panel)
            app.canvas_widget.draw()
            app.canvas_widget.get_tk_widget().pack(fill="both", expand=True)

            app.latest_figure = fig
            app.latest_ticker = ticker
            app.latest_pure_text = raport_pure_text

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    ttk.Button(left_panel, text="Uruchom predykcję", command=run_prediction).pack(pady=(20, 0), fill="x")

    # ========== OPCJE ZAPISU ==========
    save_plot_var = tk.BooleanVar(value=True)
    save_text_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(left_panel, text="Zapisz wykres (.png)", variable=save_plot_var).pack(anchor="w", pady=(10, 0))
    ttk.Checkbutton(left_panel, text="Zapisz raport (.txt)", variable=save_text_var).pack(anchor="w")

    def save_outputs():
        from tkinter import filedialog
        if not hasattr(app, 'latest_figure') or not hasattr(app, 'latest_pure_text') or not hasattr(app, 'latest_ticker'):
            messagebox.showwarning("Brak danych", "Najpierw uruchom predykcję.")
            return

        directory = filedialog.askdirectory(title="Wybierz folder zapisu")
        if directory:
            save_report_and_plot(
                app.latest_ticker,
                app.latest_figure,
                app.latest_pure_text,
                directory,
                save_plot=save_plot_var.get(),
                save_text=save_text_var.get()
            )
            messagebox.showinfo("Zapisano", "Pliki zostały zapisane.")

    ttk.Button(left_panel, text="Zapisz wyniki", command=save_outputs).pack(pady=(10, 0), fill="x")

    ttk.Button(left_panel, text="Zamknij program", command=app.destroy).pack(pady=(10, 0), fill="x")

    app.mainloop()
