"""
AI-Driven Fraud Detection Desktop Application
University of the West of Scotland — MSc Project
Student: Evans Polley | B01823633
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

from fraud_pipeline import FraudDataPipeline, ModelManager

C = {
    "bg": "#0D1117", "panel": "#161B22", "card": "#1C2128",
    "accent": "#2563EB", "accent2": "#7C3AED", "success": "#10B981",
    "warn": "#F59E0B", "danger": "#EF4444", "text": "#E6EDF3",
    "muted": "#8B949E", "border": "#30363D", "header_bg": "#131922",
}
FF = "Segoe UI" if os.name == "nt" else "Helvetica"


class FraudDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Fraud Detector — UWS MSc | Evans Polley B01823633")
        self.configure(bg=C["bg"])
        try: self.state("zoomed")
        except Exception:
            try: self.attributes("-zoomed", True)
            except: self.geometry("1400x860")

        self.pipeline = FraudDataPipeline()
        self.models = ModelManager()
        self.current_file = None
        self._df_display = None
        self._build_ui()

    def _build_ui(self):
        self._style()
        self._build_header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)
        self.nb = ttk.Notebook(body, style="Dark.TNotebook")
        self.nb.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
        self._build_tab_dashboard()
        self._build_tab_table()
        self._build_tab_metrics()
        self._build_tab_charts()
        self._build_right_panel(body)
        self._build_statusbar()

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=C["bg"])
        s.configure("Dark.TNotebook", background=C["bg"], borderwidth=0)
        s.configure("Dark.TNotebook.Tab", background=C["panel"], foreground=C["muted"],
                    padding=[16,8], font=(FF, 10, "bold"))
        s.map("Dark.TNotebook.Tab",
              background=[("selected", C["accent"]), ("active", C["card"])],
              foreground=[("selected","white"), ("active", C["text"])])
        s.configure("Treeview", background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], rowheight=26, font=(FF, 9))
        s.configure("Treeview.Heading", background=C["header_bg"],
                    foreground=C["text"], font=(FF, 9, "bold"))
        s.map("Treeview", background=[("selected", C["accent"])])
        s.configure("TScrollbar", background=C["panel"], troughcolor=C["bg"])

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["header_bg"], height=56)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🛡", bg=C["header_bg"], fg=C["accent"],
                 font=(FF,22)).pack(side="left", padx=(16,4))
        tk.Label(hdr, text="AI Fraud Detector", bg=C["header_bg"], fg=C["text"],
                 font=(FF,16,"bold")).pack(side="left")
        tk.Label(hdr, text=" | UWS MSc Project — Evans Polley  B01823633",
                 bg=C["header_bg"], fg=C["muted"], font=(FF,10)).pack(side="left")
        tk.Button(hdr, text="⊕  Load Data", bg=C["accent"], fg="white",
                  font=(FF,10,"bold"), relief="flat", padx=14, pady=6,
                  cursor="hand2", command=self._load_file
                  ).pack(side="right", padx=16, pady=10)

    def _build_tab_dashboard(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📊  Dashboard  ")
        self.welcome_frame = tk.Frame(tab, bg=C["bg"])
        self.welcome_frame.pack(expand=True, fill="both")
        tk.Label(self.welcome_frame, text="🛡", bg=C["bg"], fg=C["accent"],
                 font=(FF,64)).pack(pady=(80,10))
        tk.Label(self.welcome_frame, text="AI-Driven Fraud Detection System",
                 bg=C["bg"], fg=C["text"], font=(FF,20,"bold")).pack()
        tk.Label(self.welcome_frame,
                 text="Load a CSV, XLSX or JSON file to begin.\n"
                      "The system preprocesses data, trains ML models and visualises fraud patterns.",
                 bg=C["bg"], fg=C["muted"], font=(FF,11), justify="center").pack(pady=8)
        tk.Button(self.welcome_frame, text="⊕  Open File", bg=C["accent"], fg="white",
                  font=(FF,12,"bold"), relief="flat", padx=24, pady=10,
                  cursor="hand2", command=self._load_file).pack(pady=20)

        self.dash_content = tk.Frame(tab, bg=C["bg"])
        self.kpi_frame = tk.Frame(self.dash_content, bg=C["bg"])
        self.kpi_frame.pack(fill="x", padx=10, pady=(10,0))
        self.kpi_vars = {}
        for label, key, colour in [
            ("Total Transactions","total_tx",C["accent"]),
            ("Fraud Detected","fraud_tx",C["danger"]),
            ("Fraud Rate","fraud_pct",C["warn"]),
            ("Best F1 Score","best_f1",C["success"]),
            ("Best AUC-ROC","best_auc",C["accent2"])]:
            card = tk.Frame(self.kpi_frame, bg=C["card"],
                            highlightbackground=colour, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            tk.Label(card, text=label, bg=C["card"], fg=C["muted"], font=(FF,9)).pack(pady=(10,2))
            v = tk.StringVar(value="—"); self.kpi_vars[key] = v
            tk.Label(card, textvariable=v, bg=C["card"], fg=colour,
                     font=(FF,18,"bold")).pack(pady=(0,10))
        self.dash_fig = plt.Figure(figsize=(12,6), facecolor=C["bg"])
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=self.dash_content)
        self.dash_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_tab_table(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  🗄️  Data Table  ")
        toolbar = tk.Frame(tab, bg=C["panel"], height=42)
        toolbar.pack(fill="x"); toolbar.pack_propagate(False)
        tk.Label(toolbar, text="🔍 Search:", bg=C["panel"], fg=C["text"],
                 font=(FF,9)).pack(side="left", padx=(12,4), pady=10)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_table)
        tk.Entry(toolbar, textvariable=self.search_var, bg=C["card"], fg=C["text"],
                 insertbackground=C["text"], relief="flat", font=(FF,9), width=30
                 ).pack(side="left", pady=8)
        for label, cmd, colour in [
            ("➕ Add",self._crud_add,C["success"]),
            ("✏️ Edit",self._crud_edit,C["accent"]),
            ("🗑️ Delete",self._crud_delete,C["danger"])]:
            tk.Button(toolbar, text=label, bg=colour, fg="white",
                      font=(FF,9,"bold"), relief="flat", padx=10, pady=2,
                      cursor="hand2", command=cmd).pack(side="left", padx=4, pady=8)
        self.row_count_var = tk.StringVar(value="No data loaded")
        tk.Label(toolbar, textvariable=self.row_count_var, bg=C["panel"], fg=C["muted"],
                 font=(FF,9)).pack(side="right", padx=12)
        frame = tk.Frame(tab, bg=C["bg"])
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y"); hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("fraud", background="#3B0A0A", foreground="#FCA5A5")
        self.tree.tag_configure("legit", background=C["card"], foreground=C["text"])

    def _build_tab_metrics(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📈  Model Metrics  ")
        canvas = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); canvas.pack(fill="both", expand=True)
        self.metrics_inner = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0,0), window=self.metrics_inner, anchor="nw")
        self.metrics_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        tk.Label(self.metrics_inner, text="Run fraud detection to see model results.",
                 bg=C["bg"], fg=C["muted"], font=(FF,12)).pack(pady=60)

    def _build_tab_charts(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📉  Charts  ")
        self.charts_fig = plt.Figure(figsize=(12,8), facecolor=C["bg"])
        self.charts_canvas = FigureCanvasTkAgg(self.charts_fig, master=tab)
        self.charts_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=C["panel"], width=270)
        panel.pack(side="right", fill="y", padx=(0,8), pady=8)
        panel.pack_propagate(False)

        def section(t):
            tk.Label(panel, text=t, bg=C["panel"], fg=C["accent"],
                     font=(FF,10,"bold")).pack(anchor="w", padx=14, pady=(14,4))
            tk.Frame(panel, bg=C["border"], height=1).pack(fill="x", padx=12, pady=(0,8))

        section("FILE OPERATIONS")
        self._rb(panel, "⊕  Load File (CSV / XLSX / JSON)", self._load_file, C["accent"])
        self._rb(panel, "🔄  Reload Current File", self._reload_file, C["card"])
        self._rb(panel, "💾  Export Results as CSV", self._export_csv, C["card"])

        section("DETECTION SETTINGS")
        tk.Label(panel, text="Algorithm:", bg=C["panel"], fg=C["muted"],
                 font=(FF,9)).pack(anchor="w", padx=14)
        self.model_var = tk.StringVar(value="All Models")
        ttk.Combobox(panel, textvariable=self.model_var,
                     values=["All Models"]+list(ModelManager.MODELS.keys()),
                     state="readonly", font=(FF,9)).pack(fill="x", padx=12, pady=4)
        tk.Label(panel, text="Test Set Size:", bg=C["panel"], fg=C["muted"],
                 font=(FF,9)).pack(anchor="w", padx=14, pady=(6,0))
        self.test_size_var = tk.DoubleVar(value=0.2)
        tk.Scale(panel, from_=0.1, to=0.4, resolution=0.05, orient="horizontal",
                 variable=self.test_size_var, bg=C["panel"], fg=C["text"],
                 troughcolor=C["card"], highlightthickness=0, font=(FF,8)
                 ).pack(fill="x", padx=12)
        self.smote_var = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="Use SMOTE Resampling", variable=self.smote_var,
                       bg=C["panel"], fg=C["text"], selectcolor=C["accent"],
                       activebackground=C["panel"], font=(FF,9)
                       ).pack(anchor="w", padx=12, pady=4)
        self._rb(panel, "▶  Run Fraud Detection", self._run_detection, C["success"])

        section("CHART OPTIONS")
        self.chart_type_var = tk.StringVar(value="All Charts")
        ttk.Combobox(panel, textvariable=self.chart_type_var,
                     values=["All Charts","Line Graph Only",
                             "Donut Chart Only","Confusion Matrix"],
                     state="readonly", font=(FF,9)).pack(fill="x", padx=12, pady=4)
        self._rb(panel, "🔃  Refresh Charts", self._refresh_charts, C["card"])

        section("DATASET INFO")
        self.info_vars = {}
        for label, key in [("Rows","rows"),("Columns","cols"),
                            ("Fraud","fraud"),("Legit","legit"),("Source","source")]:
            row = tk.Frame(panel, bg=C["panel"])
            row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=label+":", bg=C["panel"], fg=C["muted"],
                     font=(FF,9), width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="—"); self.info_vars[key] = v
            tk.Label(row, textvariable=v, bg=C["panel"], fg=C["text"],
                     font=(FF,9,"bold")).pack(side="left")

        section("PROGRESS")
        self.progress_var = tk.StringVar(value="Ready — load a dataset to begin")
        tk.Label(panel, textvariable=self.progress_var, bg=C["panel"], fg=C["muted"],
                 font=(FF,9), wraplength=240).pack(padx=12)
        self.progress_bar = ttk.Progressbar(panel, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=12, pady=8)

        section("ABOUT")
        tk.Label(panel,
                 text="University of the West of Scotland\n"
                      "MSc Computer Science\n"
                      "Evans Polley | B01823633\n\n"
                      "Models: Logistic Regression,\n"
                      "Decision Tree, Random Forest, SVM\n\n"
                      "Metrics: Accuracy, Precision,\n"
                      "Recall, F1-Score, AUC-ROC",
                 bg=C["panel"], fg=C["muted"], font=(FF,8), justify="left"
                 ).pack(anchor="w", padx=14, pady=(0,14))

    def _rb(self, parent, text, cmd, colour):
        tk.Button(parent, text=text, bg=colour, fg="white",
                  font=(FF,9,"bold"), relief="flat", padx=10, pady=6,
                  cursor="hand2", command=cmd).pack(fill="x", padx=12, pady=3)

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=C["header_bg"], height=24)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready — load a dataset to begin")
        tk.Label(sb, textvariable=self.status_var, bg=C["header_bg"], fg=C["muted"],
                 font=(FF,8)).pack(side="left", padx=10)

    # ── File Ops ──

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Open Dataset",
            filetypes=[("Supported","*.csv *.xlsx *.xls *.json"),
                       ("CSV","*.csv"),("Excel","*.xlsx *.xls"),
                       ("JSON","*.json"),("All","*.*")])
        if not path: return
        self.current_file = path
        self._do_load(path)

    def _reload_file(self):
        if self.current_file: self._do_load(self.current_file)
        else: messagebox.showinfo("No File","Load a file first.")

    def _do_load(self, path):
        def worker():
            self._set_status(f"Loading {os.path.basename(path)}…")
            self.after(0, self.progress_bar.start)
            self.after(0, lambda: self.progress_var.set("Loading and preprocessing…"))
            try:
                df = self.pipeline.load(path)
                self.after(0, lambda: self._on_data_loaded(df, path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load Error", str(e)))
            finally:
                self.after(0, self.progress_bar.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_data_loaded(self, df, path):
        self.welcome_frame.pack_forget()
        self.dash_content.pack(fill="both", expand=True)
        n = len(df)
        nf = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
        pct = nf/n*100 if n else 0
        self.info_vars["rows"].set(f"{n:,}")
        self.info_vars["cols"].set(str(len(df.columns)))
        self.info_vars["fraud"].set(f"{nf:,}")
        self.info_vars["legit"].set(f"{n-nf:,}")
        self.info_vars["source"].set(os.path.basename(path)[:22])
        self.kpi_vars["total_tx"].set(f"{n:,}")
        self.kpi_vars["fraud_tx"].set(f"{nf:,}")
        self.kpi_vars["fraud_pct"].set(f"{pct:.2f}%")
        self._populate_table(df)
        self._draw_dashboard_charts()
        self.progress_var.set(f"Loaded {n:,} records successfully.")
        self._set_status(f"Loaded {n:,} records from {os.path.basename(path)}")

    def _export_csv(self):
        if self.pipeline.df is None:
            messagebox.showinfo("No Data","Load data first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV","*.csv")])
        if path:
            self.pipeline.df.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"Saved to:\n{path}")

    # ── Table ──

    def _populate_table(self, df):
        self.tree.delete(*self.tree.get_children())
        prio = ["trans_date_trans_time","merchant","category","amt","city","state","is_fraud"]
        show = [c for c in prio if c in df.columns] or list(df.columns[:10])
        self.tree["columns"] = show
        for col in show:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=max(80, min(220, len(col)*11)), anchor="w")
        self._df_display = df[show].copy().reset_index(drop=True)
        self._load_tree_rows(self._df_display)

    def _load_tree_rows(self, df):
        self.tree.delete(*self.tree.get_children())
        for i, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float): vals.append(f"{v:.4f}" if abs(v)<1e6 else f"{v:.2e}")
                else: vals.append(str(v)[:60])
            tag = "fraud" if str(row.get("is_fraud","0")) in ("1","1.0") else "legit"
            self.tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))
        nf = int((df.get("is_fraud", pd.Series([0]*len(df)))==1).sum())
        self.row_count_var.set(f"{len(df):,} rows  |  {nf} fraud rows highlighted")

    def _filter_table(self, *_):
        if self._df_display is None: return
        q = self.search_var.get().lower()
        if not q:
            self._load_tree_rows(self._df_display); return
        mask = self._df_display.astype(str).apply(
            lambda col: col.str.lower().str.contains(q, na=False)).any(axis=1)
        self._load_tree_rows(self._df_display[mask])

    def _sort_by(self, col):
        if self._df_display is None: return
        asc = not getattr(self, f"_s_{col}", True)
        setattr(self, f"_s_{col}", asc)
        self._df_display = self._df_display.sort_values(col, ascending=asc).reset_index(drop=True)
        self._load_tree_rows(self._df_display)

    def _get_idx(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection","Select a row first."); return None
        return int(sel[0])

    def _crud_add(self):
        if self._df_display is None:
            messagebox.showinfo("No Data","Load data first."); return
        self._row_editor(mode="add")

    def _crud_edit(self):
        idx = self._get_idx()
        if idx is not None: self._row_editor(mode="edit", idx=idx)

    def _crud_delete(self):
        idx = self._get_idx()
        if idx is not None and messagebox.askyesno("Delete","Delete selected record?"):
            self._df_display = self._df_display.drop(index=idx).reset_index(drop=True)
            self._load_tree_rows(self._df_display)

    def _row_editor(self, mode="add", idx=None):
        win = tk.Toplevel(self)
        win.title("Add Record" if mode=="add" else "Edit Record")
        win.configure(bg=C["bg"]); win.geometry("540x520"); win.grab_set()
        canvas = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cols = list(self._df_display.columns)
        entries = {}
        for i, col in enumerate(cols):
            tk.Label(inner, text=col, bg=C["bg"], fg=C["muted"],
                     font=(FF,9), anchor="w", width=28
                     ).grid(row=i, column=0, sticky="w", padx=16, pady=3)
            v = tk.StringVar()
            if mode=="edit" and idx is not None:
                v.set(str(self._df_display.at[idx, col]))
            tk.Entry(inner, textvariable=v, bg=C["card"], fg=C["text"],
                     insertbackground=C["text"], relief="flat", font=(FF,9), width=36
                     ).grid(row=i, column=1, padx=10, pady=3)
            entries[col] = v
        def save():
            nr = {col: v.get() for col,v in entries.items()}
            if mode=="add":
                self._df_display = pd.concat(
                    [self._df_display, pd.DataFrame([nr])], ignore_index=True)
            else:
                for col,v in entries.items(): self._df_display.at[idx,col] = v.get()
            self._load_tree_rows(self._df_display); win.destroy()
        tk.Button(inner, text="💾  Save Record", bg=C["success"], fg="white",
                  font=(FF,10,"bold"), relief="flat", padx=20, pady=6, command=save
                  ).grid(row=len(cols), column=0, columnspan=2, pady=16)

    # ── Detection ──

    def _run_detection(self):
        if self.pipeline.df is None:
            messagebox.showinfo("No Data","Load a dataset first."); return
        def worker():
            self.after(0, self.progress_bar.start)
            self.after(0, lambda: self.progress_var.set("Splitting & resampling…"))
            try:
                self.pipeline.split_and_resample(
                    test_size=self.test_size_var.get(),
                    use_smote=self.smote_var.get())
                sel = self.model_var.get()
                def cb(msg):
                    self.after(0, lambda: self.progress_var.set(msg))
                    self.after(0, lambda: self._set_status(msg))
                if sel == "All Models":
                    self.models.train_all(
                        self.pipeline.X_train, self.pipeline.y_train,
                        self.pipeline.X_test, self.pipeline.y_test, progress_cb=cb)
                else:
                    cb(f"Training {sel}…")
                    self.models.train_single(sel,
                        self.pipeline.X_train, self.pipeline.y_train,
                        self.pipeline.X_test, self.pipeline.y_test)
                self.after(0, self._on_detection_done)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
            finally:
                self.after(0, self.progress_bar.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_detection_done(self):
        valid = {k:v for k,v in self.models.results.items() if "f1" in v}
        if valid:
            self.kpi_vars["best_f1"].set(f"{max(v['f1'] for v in valid.values()):.3f}")
            self.kpi_vars["best_auc"].set(f"{max(v['auc'] for v in valid.values()):.3f}")
        self._update_metrics_tab()
        self._draw_charts()
        self.progress_var.set("✅ Detection complete!")
        self._set_status("Detection complete — see Metrics and Charts tabs")
        self.nb.select(2)

    def _update_metrics_tab(self):
        for w in self.metrics_inner.winfo_children(): w.destroy()
        results = self.models.results
        if not results:
            tk.Label(self.metrics_inner, text="No results", bg=C["bg"],
                     fg=C["muted"], font=(FF,12)).pack(pady=40); return
        tk.Label(self.metrics_inner, text="Model Performance Comparison",
                 bg=C["bg"], fg=C["text"], font=(FF,14,"bold")).pack(pady=(16,4))
        tk.Label(self.metrics_inner,
                 text=f"Train: {len(self.pipeline.y_train):,}  |  "
                      f"Test: {len(self.pipeline.y_test):,}  |  "
                      f"SMOTE: {'Yes' if self.smote_var.get() else 'No'}",
                 bg=C["bg"], fg=C["muted"], font=(FF,9)).pack(pady=(0,12))
        metrics = [("Accuracy","accuracy",C["accent"]),
                   ("Precision","precision",C["warn"]),
                   ("Recall","recall",C["success"]),
                   ("F1-Score","f1",C["accent2"]),
                   ("AUC-ROC","auc",C["danger"])]
        for mname, res in results.items():
            if "error" in res:
                card = tk.Frame(self.metrics_inner, bg=C["card"],
                                highlightbackground=C["danger"], highlightthickness=1)
                card.pack(fill="x", padx=16, pady=6)
                tk.Label(card, text=f"{mname} — ERROR: {res['error']}",
                         bg=C["card"], fg=C["danger"], font=(FF,9)).pack(padx=12, pady=8)
                continue
            card = tk.Frame(self.metrics_inner, bg=C["card"],
                            highlightbackground=C["border"], highlightthickness=1)
            card.pack(fill="x", padx=16, pady=8)
            hdr = tk.Frame(card, bg=C["header_bg"])
            hdr.pack(fill="x")
            tk.Label(hdr, text=f"  {mname}", bg=C["header_bg"], fg=C["text"],
                     font=(FF,11,"bold")).pack(side="left", pady=8)
            cm = res.get("cm")
            if cm is not None and cm.shape==(2,2):
                tn,fp,fn,tp = cm.ravel()
                tk.Label(hdr, text=f"  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}",
                         bg=C["header_bg"], fg=C["muted"], font=(FF,9)
                         ).pack(side="right", padx=12)
            row = tk.Frame(card, bg=C["card"])
            row.pack(fill="x", padx=12, pady=10)
            for label, key, colour in metrics:
                val = res.get(key, 0)
                cf = tk.Frame(row, bg=C["card"])
                cf.pack(side="left", expand=True)
                tk.Label(cf, text=label, bg=C["card"], fg=C["muted"], font=(FF,8)).pack()
                tk.Label(cf, text=f"{val:.3f}", bg=C["card"], fg=colour,
                         font=(FF,15,"bold")).pack()
                bb = tk.Frame(cf, bg=C["border"], height=5, width=80)
                bb.pack(pady=(2,0)); bb.pack_propagate(False)
                tk.Frame(bb, bg=colour, height=5, width=max(2,int(val*80))
                         ).place(x=0, y=0)

    # ── Charts ──

    def _draw_dashboard_charts(self):
        df = self.pipeline.df
        if df is None: return
        self.dash_fig.clear(); self.dash_fig.patch.set_facecolor(C["bg"])
        gs = GridSpec(1,2, figure=self.dash_fig, wspace=0.4)
        ax1 = self.dash_fig.add_subplot(gs[0]); ax1.set_facecolor(C["card"])
        self._draw_line_chart(ax1, df)
        ax2 = self.dash_fig.add_subplot(gs[1]); ax2.set_facecolor(C["bg"])
        self._draw_donut(ax2, df)
        ax2.set_title("Transaction vs Fraud — Twin Donut", color=C["text"], fontsize=9, pad=8)
        self.dash_canvas.draw()

    def _draw_line_chart(self, ax, df):
        ax.set_facecolor(C["card"])
        if "trans_datetime" in df.columns and df["trans_datetime"].notna().any():
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["trans_datetime"], utc=True, errors="coerce").dt.date
            daily = tmp.groupby(["date","is_fraud"]).size().unstack(fill_value=0)
            x = range(len(daily))
            if 0 in daily.columns:
                ax.plot(x, daily[0], color=C["accent"], lw=1.5, label="Legitimate")
                ax.fill_between(x, daily[0], alpha=0.12, color=C["accent"])
            if 1 in daily.columns:
                ax.plot(x, daily[1], color=C["danger"], lw=2, label="Fraud",
                        marker="o", markersize=3)
                ax.fill_between(x, daily[1], alpha=0.35, color=C["danger"])
            step = max(1, len(daily)//6)
            ax.set_xticks(list(range(0, len(daily), step)))
            ax.set_xticklabels([str(d) for d in daily.index[::step]],
                                rotation=30, fontsize=6, color=C["muted"])
        else:
            ax.text(0.5,0.5,"No datetime column", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes)
        ax.set_title("Fraud Instances Over Time", color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7)

    def _draw_donut(self, ax, df):
        if "category" in df.columns and "is_fraud" in df.columns:
            ct = df.groupby("category").size()
            cf = df[df["is_fraud"]==1].groupby("category").size().reindex(ct.index, fill_value=0)
            labels = ct.index.tolist()
            outer_vals = ct.values.astype(float)
            inner_vals = cf.values.astype(float)
        else:
            nleg = int((df["is_fraud"]==0).sum()); nfr = int((df["is_fraud"]==1).sum())
            labels = ["Legitimate","Fraud"]
            outer_vals = np.array([nleg, nfr], dtype=float)
            inner_vals = np.array([0, nfr], dtype=float)
        cmap = plt.cm.get_cmap("tab20", max(len(labels),2))
        colours = [cmap(i) for i in range(len(labels))]
        fraud_c = [(*c[:3],0.5) for c in colours]
        ax.pie(outer_vals, radius=1.0, colors=colours, startangle=90,
               wedgeprops=dict(width=0.38, edgecolor=C["bg"], linewidth=1.5))
        if inner_vals.sum()>0:
            ax.pie(inner_vals, radius=0.60, colors=fraud_c, startangle=90,
                   wedgeprops=dict(width=0.30, edgecolor=C["bg"], linewidth=1))
        ft = int(inner_vals.sum())
        ax.text(0,0.10,f"{ft:,}", ha="center",va="center",
                fontsize=13,fontweight="bold",color=C["danger"])
        ax.text(0,-0.14,"fraud", ha="center",va="center",fontsize=8,color=C["muted"])
        ax.text(0,-0.30,"outer=all  inner=fraud", ha="center",va="center",
                fontsize=6,color=C["muted"])
        patches = [mpatches.Patch(color=colours[i], label=labels[i][:18])
                   for i in range(min(10,len(labels)))]
        ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5,-0.32),
                  ncol=2, fontsize=6.5, facecolor=C["bg"], edgecolor=C["border"],
                  labelcolor=C["text"], framealpha=0.8)

    def _draw_model_bar(self, ax, results):
        valid = {k:v for k,v in results.items() if "f1" in v}
        if not valid:
            ax.text(0.5,0.5,"Run detection first",ha="center",va="center",
                    color=C["muted"],transform=ax.transAxes); return
        mets = ["accuracy","precision","recall","f1","auc"]
        mlbls = ["Acc","Prec","Rec","F1","AUC"]
        cols = [C["accent"],C["warn"],C["success"],C["accent2"],C["danger"]]
        x = np.arange(len(valid)); w = 0.14
        for i,(met,col) in enumerate(zip(mets,cols)):
            vals = [v.get(met,0) for v in valid.values()]
            ax.bar(x+i*w, vals, w, label=mlbls[i], color=col, alpha=0.85)
        ax.set_xticks(x+w*2)
        ax.set_xticklabels(list(valid.keys()), fontsize=7, color=C["muted"], rotation=8)
        ax.set_ylim(0,1.15)
        ax.set_title("Model Metrics Comparison", color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.set_facecolor(C["card"])
        ax.legend(facecolor=C["card"],edgecolor=C["border"],
                  labelcolor=C["text"],fontsize=6,ncol=5)

    def _draw_cm(self, ax, cm, title):
        ax.set_facecolor(C["card"])
        ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
        ax.set_title(title, color=C["text"], fontsize=8, pad=4)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legit","Fraud"], color=C["muted"], fontsize=7)
        ax.set_yticklabels(["Legit","Fraud"], color=C["muted"], fontsize=7)
        ax.tick_params(colors=C["muted"])
        for sp in ax.spines.values(): sp.set_color(C["border"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                        color="white" if cm[i,j]>cm.max()/2 else C["text"],
                        fontsize=11,fontweight="bold")
        ax.set_xlabel("Predicted",color=C["muted"],fontsize=7)
        ax.set_ylabel("Actual",color=C["muted"],fontsize=7)

    def _draw_charts(self):
        self.charts_fig.clear(); self.charts_fig.patch.set_facecolor(C["bg"])
        df = self.pipeline.df; results = self.models.results
        ct = self.chart_type_var.get()
        if ct == "Line Graph Only":
            ax = self.charts_fig.add_subplot(111); self._draw_line_chart(ax,df)
        elif ct == "Donut Chart Only":
            ax = self.charts_fig.add_subplot(111)
            self._draw_donut(ax,df)
            ax.set_title("Transactions vs Fraud (Twin Donut)",color=C["text"],fontsize=11)
        elif ct == "Confusion Matrix":
            valid = {k:v for k,v in results.items() if "cm" in v}
            n = len(valid)
            if n==0: return
            cn,rn = min(n,2),(n+1)//2
            for i,(mname,res) in enumerate(valid.items()):
                ax = self.charts_fig.add_subplot(rn,cn,i+1)
                self._draw_cm(ax,res["cm"],mname)
        else:
            gs = GridSpec(2,2,figure=self.charts_fig,hspace=0.45,wspace=0.4)
            ax0 = self.charts_fig.add_subplot(gs[0,0]); ax0.set_facecolor(C["card"])
            self._draw_line_chart(ax0,df)
            ax1 = self.charts_fig.add_subplot(gs[0,1]); ax1.set_facecolor(C["bg"])
            self._draw_donut(ax1,df)
            ax1.set_title("Transactions vs Fraud (Twin Donut)",color=C["text"],fontsize=9,pad=8)
            ax2 = self.charts_fig.add_subplot(gs[1,0]); ax2.set_facecolor(C["card"])
            self._draw_model_bar(ax2,results)
            ax3 = self.charts_fig.add_subplot(gs[1,1]); ax3.set_facecolor(C["card"])
            valid = {k:v for k,v in results.items() if "cm" in v}
            if valid:
                best = max(valid,key=lambda k: valid[k].get("f1",0))
                self._draw_cm(ax3,valid[best]["cm"],f"Best Model: {best}")
        self.charts_canvas.draw()

    def _refresh_charts(self):
        if self.pipeline.df:
            if self.models.results: self._draw_charts()
            else: self._draw_dashboard_charts()

    def _set_status(self, msg):
        self.status_var.set(msg)


if __name__ == "__main__":
    app = FraudDetectorApp()
    app.mainloop()
