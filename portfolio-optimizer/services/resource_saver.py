import os
from datetime import datetime


def save_report_and_plot(ticker, fig, report_text, save_dir, save_plot=True, save_text=True):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if save_plot:
        fig_path = os.path.join(save_dir, f"{ticker}_wykres_{timestamp}.png")
        fig.savefig(fig_path)

    if save_text:
        text_path = os.path.join(save_dir, f"{ticker}_raport_{timestamp}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(report_text)