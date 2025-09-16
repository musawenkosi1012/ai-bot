# src/gui.py
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import time

class BotGUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("ICT M1 Scalper - Python (ML)")
        self.status_label = tk.Label(self.root, text="Stopped", fg="red")
        self.status_label.pack()
        self.start_btn = tk.Button(self.root, text="Start", command=self.start)
        self.stop_btn  = tk.Button(self.root, text="Stop", command=self.stop, state='disabled')
        self.start_btn.pack(); self.stop_btn.pack()
        self.log = ScrolledText(self.root, height=20, width=80)
        self.log.pack()
        self.update_metrics()

    def start(self):
        self.log.insert(tk.END, "Starting engine...\n")
        self.status_label.config(text="Running", fg="green")
        self.start_btn.config(state='disabled'); self.stop_btn.config(state='normal')
        t = threading.Thread(target=self.engine.run, daemon=True)
        t.start()

    def stop(self):
        self.log.insert(tk.END, "Stopping engine...\n")
        self.status_label.config(text="Stopped", fg="red")
        self.start_btn.config(state='normal'); self.stop_btn.config(state='disabled')
        self.engine.stop()

    def log_message(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def update_metrics(self):
        # update live metrics e.g. P/L, accepted rate
        # call again
        self.root.after(1000, self.update_metrics)

    def run(self):
        self.root.mainloop()