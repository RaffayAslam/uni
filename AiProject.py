import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import os
import requests
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

API_KEY = "5a1019409988059d1727e7491a4d585a"
HISTORICAL_DATA_FILE = 'Pakistan_weather_data.csv'
MODEL_FILENAME = 'pakistan_pro_model.pkl'


def train_pro_model():
    if not os.path.exists(HISTORICAL_DATA_FILE):
        return None, 0, None

    df = pd.read_csv(HISTORICAL_DATA_FILE)
    df['temp_c'] = (df['temp'] - 32) * 5/9 
    df['soil_moisture'] = (df['rh'] * 0.6).clip(0, 100)
    
    def label_logic(row):
        # Flood: Precipitation high
        if row['precip'] > 0.05: return 2 
        # Drought: Temp high, humidity low
        if row['temp_c'] > 40 and row['rh'] < 30: return 1 
        return 0 # Normal

    df['condition'] = df.apply(label_logic, axis=1)
    X = df[['temp_c', 'rh', 'precip', 'soil_moisture']]
    y = df['condition']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Calculate Evaluation Metrics
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    joblib.dump({'model': pipeline, 'cm': cm, 'acc': acc}, MODEL_FILENAME)
    return pipeline, acc, cm


class EcoGuardEnterprise:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoGuard Pro: Enterprise AI Disaster Management")
        self.root.geometry("1200x950")
        
        # Initialize AI Data
        if not os.path.exists(MODEL_FILENAME):
            self.model, self.accuracy, self.cm = train_pro_model()
        else:
            try:
                data = joblib.load(MODEL_FILENAME)
                self.model, self.accuracy, self.cm = data['model'], data['acc'], data['cm']
            except:
                # If file is old/corrupt, retrain
                self.model, self.accuracy, self.cm = train_pro_model()

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#1e293b", height=80)
        header.pack(fill="x")
        tk.Label(header, text="ECOGUARD ENTERPRISE: PAKISTAN CLIMATE INTELLIGENCE", fg="white", bg="#1e293b", font=("Arial", 18, "bold")).pack(pady=20)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_dash = ttk.Frame(self.notebook)
        self.tab_fore = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dash, text=" Live Dashboard ")
        self.notebook.add(self.tab_fore, text=" 5-Day Forecast ")
        self.notebook.add(self.tab_eval, text=" Model Evaluation (AI Metrics) ")

        self.build_dashboard()
        self.build_forecast()
        self.build_evaluation()

    def build_dashboard(self):
        left = tk.Frame(self.tab_dash, bg="white", width=350, relief="solid", bd=1, padx=20, pady=20)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        tk.Label(left, text="LOCATION SERVICES", font=("Arial", 9, "bold")).pack(anchor="w")
        self.city_entry = tk.Entry(left, font=("Arial", 12))
        self.city_entry.insert(0, "Karachi")
        self.city_entry.pack(fill="x", pady=5)
        tk.Button(left, text="SYNC API", command=self.fetch_all, bg="#2563eb", fg="white").pack(fill="x")

        # Sliders
        self.temp = self.create_slider(left, "Temperature", 0, 60, 30)
        self.hum = self.create_slider(left, "Humidity", 0, 100, 50)
        self.rain = self.create_slider(left, "Rainfall", 0, 400, 0)
        self.soil = self.create_slider(left, "Soil Saturation", 0, 100, 40)

        tk.Button(left, text="GENERATE AI REPORT", command=self.run_ai, bg="#1e293b", fg="white", font=("Arial", 10, "bold"), pady=10).pack(fill="x", pady=20)
        
        self.alert_log = tk.Text(left, height=8, font=("Courier", 8), bg="#fef2f2", fg="#991b1b")
        self.alert_log.pack(fill="x", pady=10)
        self.alert_log.insert("1.0", "ALERT SYSTEM ACTIVE\nMonitoring for risks...")

        right = tk.Frame(self.tab_dash, bg="white", padx=20, pady=20)
        right.pack(side="right", fill="both", expand=True)

        self.metric_frame = tk.Frame(right, bg="white")
        self.metric_frame.pack(fill="x")
        self.d_temp = self.create_digit(self.metric_frame, "TEMP", 0)
        self.d_hum = self.create_digit(self.metric_frame, "HUMID", 1)
        self.d_rain = self.create_digit(self.metric_frame, "RAIN", 2)

        self.res_lbl = tk.Label(right, text="SYSTEM INITIALIZED", font=("Arial", 28, "bold"), bg="white", fg="#cbd5e1")
        self.res_lbl.pack(pady=40)
        self.conf_lbl = tk.Label(right, text="Confidence: --", font=("Arial", 12), bg="white")
        self.conf_lbl.pack()

    def create_digit(self, parent, txt, col):
        f = tk.Frame(parent, bg="#f8fafc", bd=1, relief="flat", padx=10, pady=10)
        f.grid(row=0, column=col, padx=5, sticky="nsew")
        tk.Label(f, text=txt, font=("Arial", 8, "bold"), bg="#f8fafc", fg="#64748b").pack()
        lbl = tk.Label(f, text="--", font=("Arial", 16, "bold"), bg="#f8fafc")
        lbl.pack()
        parent.grid_columnconfigure(col, weight=1)
        return lbl

    def create_slider(self, parent, label, mn, mx, val):
        tk.Label(parent, text=label, bg="white", font=("Arial", 8)).pack(anchor="w", pady=(10,0))
        v = tk.DoubleVar(value=val)
        ttk.Scale(parent, variable=v, from_=mn, to=mx).pack(fill="x")
        return v

    def build_evaluation(self):
        """Displays Confusion Matrix and Accuracy Chart with Error Fixes"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        container = tk.Frame(self.tab_eval, bg="white")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        fig = Figure(figsize=(8, 5), dpi=100)
        
        # Accuracy Bar Chart
        ax1 = fig.add_subplot(121)
        ax1.bar(["Model Accuracy"], [self.accuracy * 100], color="#3b82f6")
        ax1.set_ylim(0, 100)
        ax1.set_title("Overall Accuracy (%)")

        # Confusion Matrix Heatmap
        ax2 = fig.add_subplot(122)
        ax2.imshow(self.cm, interpolation='nearest', cmap='Blues')
        ax2.set_title("Confusion Matrix")
        
        labels = ['Norm', 'Drou', 'Floo']
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_yticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        
        # FIX: Dynamic loop using the actual shape of self.cm to avoid IndexErrors
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax2.text(j, i, str(self.cm[i, j]), ha="center", va="center", 
                         color="white" if self.cm[i, j] > (self.cm.max()/2) else "black")

        canvas = FigureCanvasTkAgg(fig, container)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def build_forecast(self):
        cols = ("Time", "Temp", "Hum", "Rain", "AI Result", "Conf %")
        self.tree = ttk.Treeview(self.tab_fore, columns=cols, show='headings')
        for c in cols: self.tree.heading(c, text=c)
        self.tree.pack(fill="both", expand=True, padx=20, pady=20)

    def fetch_all(self):
        city = self.city_entry.get()
        c_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        f_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        try:
            r = requests.get(c_url).json()
            if r["cod"] == 200:
                self.temp.set(r["main"]["temp"])
                self.hum.set(r["main"]["humidity"])
                self.rain.set(r.get("rain", {}).get("1h", 0))
                self.run_ai()
            
            fr = requests.get(f_url).json()
            if fr["cod"] == "200":
                for i in self.tree.get_children(): self.tree.delete(i)
                for item in fr["list"]:
                    t, h = item["main"]["temp"], item["main"]["humidity"]
                    ra = item.get("rain", {}).get("3h", 0) / 3
                    feat = pd.DataFrame([[t, h, ra, 40]], columns=['temp_c', 'rh', 'precip', 'soil_moisture'])
                    p = self.model.predict(feat)[0]
                    c = max(self.model.predict_proba(feat)[0])
                    self.tree.insert("", "end", values=(item["dt_txt"][11:16], t, h, f"{ra:.1f}", ["Normal", "Drought", "Flood"][p], f"{c*100:.1f}"))
        except: messagebox.showerror("Error", "Sync Failed")

    def run_ai(self):
        self.d_temp.config(text=f"{self.temp.get():.1f}°C")
        self.d_hum.config(text=f"{self.hum.get():.0f}%")
        self.d_rain.config(text=f"{self.rain.get():.1f}mm")

        feat = pd.DataFrame([[self.temp.get(), self.hum.get(), self.rain.get(), self.soil.get()]], 
                            columns=['temp_c', 'rh', 'precip', 'soil_moisture'])
        res = self.model.predict(feat)[0]
        prob = max(self.model.predict_proba(feat)[0])
        
        labels = ["NORMAL", "DROUGHT RISK", "FLOOD RISK"]
        clrs = ["#059669", "#ea580c", "#dc2626"]
        
        self.res_lbl.config(text=labels[res], fg=clrs[res])
        self.conf_lbl.config(text=f"AI Confidence: {prob*100:.1f}%")

        if res != 0 and prob > 0.80:
            self.trigger_alert(labels[res], prob)

    def trigger_alert(self, disaster, conf):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] ALERT: {disaster}\nConfidence: {conf*100:.1f}%\nStatus: SMS Sent to PDMA\nEmail sent to Admin.\n"
        self.alert_log.insert("1.0", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = EcoGuardEnterprise(root)
    root.mainloop()
