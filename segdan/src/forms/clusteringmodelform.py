import tkinter as tk
from tkinter import ttk

from src.utils.confighandler import ConfigHandler

class ClusteringModelForm():
    def __init__(self, parent, allow_grid=False, models=ConfigHandler.CONFIGURATION_VALUES["clustering_models"]):
            

            self.top = tk.Toplevel(parent)
            self.allow_grid = allow_grid

            self.top.title("Clustering model selection")

            self.top.geometry("450x275")

            self.model_selected = tk.StringVar(value="")
            self.grid_search = tk.BooleanVar(value=False)

            self.config_data = {}
            self.model_name = ""
            self.models = models
            self.linkages = ConfigHandler.CONFIGURATION_VALUES["linkages"]

            self.top.grid_columnconfigure(0, weight=1)
            self.top.grid_columnconfigure(1, weight=1)
            self.top.grid_columnconfigure(2, weight=1)
            self.top.grid_columnconfigure(3, weight=1)

            self.model_frame = tk.LabelFrame(self.top, text="Hyperparameters", padx=10, pady=10)
            self.model_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")
            
            self.create_controls()
            
            self.on_model_change(None)

    def create_controls(self):
          
        tk.Label(self.top, text="Clustering model").grid(row=0, column=0, sticky="w", padx=10)
        model_dropdown = ttk.Combobox(self.top, textvariable=self.model_selected, values=self.models, state="readonly")
        model_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)

        if self.allow_grid:
            tk.Label(self.top, text="Use grid search").grid(row=0, column=2, sticky="w", padx=10)
            grid_search_checkbox = tk.Checkbutton(self.top, variable=self.grid_search, command= lambda: self.on_grid_search_toggle())
            grid_search_checkbox.grid(row=1, column=2, padx=10, pady=5)

        
        tk.Button(self.top, text="Cancel", command=self.clear_and_close_form).grid(row=10, column=1, pady=10, sticky="ew")
        tk.Button(self.top, text="Add model", command=self.top.destroy).grid(row=10, column=2, pady=10, sticky="ew")

    def create_model_frame(self):
        self.model_frame = tk.LabelFrame(self.top, text="Hyperparameters", padx=10, pady=10)
        self.model_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

    def clear_model_frame(self):
        for widget in self.model_frame.winfo_children():
            widget.destroy()

    def clear_form(self):
        for widget in self.model_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 1:  
                widget.grid_forget()


    def on_model_change(self, event):
        
        model = self.model_selected.get().lower() 
        self.clear_form()
        self.clear_model_frame()
        self.config_data.clear()

        if model not in self.config_data:
            self.config_data[model] = {}

        if model == "kmeans":
             self.show_kmeans_params()

        elif model == "agglomerative":
            self.show_agglomerative_params()

        elif model == "dbscan":
            self.show_dbscan_params()

        elif model == "optics":
            self.show_optics_params()     
    
    def on_grid_search_toggle(self):
        self.on_model_change(None)  

    def show_kmeans_params(self):
        self.create_param_field("kmeans", "Number of clusters", "n_clusters", row=0)
        self.create_param_field("kmeans", "Random state", "random_state", row=1)


    def show_agglomerative_params(self):
        self.create_param_field("agglomerative", "Number of clusters", "n_clusters", row=0)
        self.create_categorical_param_field("agglomerative", "Linkage", "linkages", row=1, options=self.linkages)

    def show_dbscan_params(self):
        self.create_param_field("dbscan", "Eps ", "eps", row=0)
        self.create_param_field("dbscan", "Min samples", "min_samples", row=1)

    def show_optics_params(self):
        self.create_param_field("optics", "Eps ", "eps", row=0)

    def update_int_grid_value(self, model, param_key, range_key, value):
        try:
            num_value = int(value) if value.isdigit() else float(value)
        except ValueError:
            num_value = None

        if range_key:
            self.config_data[model].setdefault(param_key, {})[range_key] = num_value
        else:
            self.config_data[model][param_key] = num_value

    def create_param_field(self, model, label_text, param_key, row):

        tk.Label(self.model_frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky="w")

        if not self.allow_grid or not self.grid_search.get() or param_key == "random_state":
            entry = tk.Entry(self.model_frame, width=10)
            entry.grid(row=row, column=1, padx=5, pady=5)
            entry.insert(0, "")  
            entry.bind("<KeyRelease>", lambda event: self.update_int_grid_value(model, param_key, None, entry.get()))
            
            self.config_data[model][param_key] = entry.get()  
            return
        
        if self.grid_search.get():

            param_key = f"{param_key}_range"

            self.config_data[model][param_key] = {"min": "", "max": "", "step": ""}

            tk.Label(self.model_frame, text="Min:").grid(row=row, column=1, padx=2, pady=5)
            entry_min = tk.Entry(self.model_frame, width=5)
            entry_min.grid(row=row, column=2, padx=2, pady=5)
            entry_min.insert(0, "")
            entry_min.bind("<KeyRelease>", lambda event: self.update_int_grid_value(model, param_key, "min", entry_min.get()))

            tk.Label(self.model_frame, text="Max:").grid(row=row, column=3, padx=2, pady=5)
            entry_max = tk.Entry(self.model_frame, width=5)
            entry_max.grid(row=row, column=4, padx=2, pady=5)
            entry_max.insert(0, "")
            entry_max.bind("<KeyRelease>", lambda event: self.update_int_grid_value(model, param_key, "max", entry_max.get()))

            tk.Label(self.model_frame, text="Step:").grid(row=row, column=5, padx=2, pady=5)
            entry_step = tk.Entry(self.model_frame, width=5)
            entry_step.grid(row=row, column=6, padx=2, pady=5)
            entry_step.insert(0, "")
            entry_step.bind("<KeyRelease>", lambda event: self.update_int_grid_value(model, param_key, "step", entry_step.get()))

    def update_selected_values(self, event, listbox, options, model, param_key):
        selected_indices = listbox.curselection()  
        selected_values = [options[i] for i in selected_indices]  
        self.config_data[model][param_key] = selected_values  

    def update_selected_values_listbox(self, event, listbox, options, model, param_key):
        selected_indices = listbox.curselection()
        self.config_data[model][param_key] = [options[i] for i in selected_indices]

    def update_selected_value_combobox(self, event, combobox, model, param_key):
        self.config_data[model][param_key] = combobox.get()
        
    def create_categorical_param_field(self, model, label_text, param_key, row, options):
        tk.Label(self.model_frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky="w")

        if self.allow_grid and self.grid_search.get():
            param_key = "linkages"
            self.config_data[model][param_key] = []
            
            listbox = tk.Listbox(self.model_frame, selectmode="multiple", height=len(options))
            for option in options:
                listbox.insert(tk.END, option)
            listbox.grid(row=row, column=1, columnspan=2, padx=5, pady=5)

            listbox.bind('<<ListboxSelect>>', lambda event: self.update_selected_values_listbox(event, listbox, options, model, param_key))
            return
        
        param_key = "linkage"
        self.config_data[model][param_key] = tk.StringVar(value=options[0])
        
        combobox = ttk.Combobox(self.model_frame, textvariable=self.config_data[model][param_key], values=options, state="readonly")
        combobox.grid(row=row, column=1, padx=5, pady=5)

        combobox.bind("<<ComboboxSelected>>", lambda event: self.update_selected_value_combobox(event, combobox, model, param_key))

    def clear_and_close_form(self):
        self.config_data = {}
        self.model_name = ""

        self.top.destroy()
    