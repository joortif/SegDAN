import tkinter as tk
from tkinter import ttk
import copy

from tktooltip import ToolTip
from src.utils.confighandler import ConfigHandler

class ClusteringModelForm():
    def __init__(self, parent, allow_grid=False, models=ConfigHandler.CONFIGURATION_VALUES["clustering_models"], config_data={}, edit=False):
            self.top = tk.Toplevel(parent)
            self.allow_grid = allow_grid

            self.top.title("Clustering model selection")

            self.top.geometry("450x275")
            self.top.resizable(False, False)

            self.model_selected = tk.StringVar(value="")
            self.grid_search = tk.BooleanVar(value=False)

            self.config_data = config_data
            self.temp_config_data = {}

            if edit and config_data:
                self.model_selected.set(list(config_data.keys())[0])
                self.model_name = self.model_selected.get()
            else:
                self.model_name = ""
            self.edit = edit
            self.models = models
            self.linkages = ConfigHandler.CONFIGURATION_VALUES["linkages"]
            self.default_hyperparameters = ConfigHandler.DEFAULT_CLUSTERING_HYPERPARAMETERS

            self.top.grid_columnconfigure(0, weight=1)
            self.top.grid_columnconfigure(1, weight=1)
            self.top.grid_columnconfigure(2, weight=1)
            self.top.grid_columnconfigure(3, weight=1)

            self.model_frame = ttk.LabelFrame(self.top, text="Hyperparameters", padding=(20, 10))
            self.model_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")
            
            self.create_controls()
            
            self.on_model_change(None)
            
            if edit and config_data:
                self.model_selected.set(list(config_data.keys())[0])
                self.model_name = self.model_selected.get()
                
                model_data = config_data.get(self.model_name, {})
                grid_params = any("_range" in key for key in model_data.keys())  

                if grid_params:
                    self.grid_search.set(True) 
                    self.on_grid_search_toggle()


            self.top.protocol("WM_DELETE_WINDOW", self.clear_and_close_form)
            

    def create_controls(self):
        for i in range(6):
            self.top.grid_columnconfigure(i, weight=1)

        center_frame = tk.Frame(self.top)
        center_frame.grid(row=1, column=0, columnspan=3, pady=10) 

        clustering_model_label = tk.Label(center_frame, text="Clustering model *")
        clustering_model_label.grid(row=1, column=0, padx=10, sticky="e")
        ToolTip(clustering_model_label, msg="Choose a clustering algorithm from the dropdown list.")

        model_dropdown = ttk.Combobox(center_frame, textvariable=self.model_selected, values=self.models, state="readonly", width=15)
        model_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w") 
        model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)

        if self.allow_grid:
            grid_search_label = tk.Label(center_frame, text="Use grid search")
            grid_search_label.grid(row=1, column=2, sticky="w", padx=10)
            ToolTip(grid_search_label, msg="Check this box to perform a grid search for hyperparameters.")

            grid_search_checkbox = tk.Checkbutton(center_frame, variable=self.grid_search, command=self.on_grid_search_toggle)
            grid_search_checkbox.grid(row=1, column=3, padx=10, pady=5, sticky="w")

        btn_frame = tk.Frame(self.top)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=20)  

        if self.edit:
            tk.Button(btn_frame, text="Delete model", command=self.delete_model).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Save", command=self.validate_and_close_form).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.clear_and_close_form).grid(row=0, column=2, padx=5)
        

    def create_model_frame(self):
        self.model_frame = ttk.LabelFrame(self.top, text="Hyperparameters", padx=10, pady=10, padding=(20, 10))
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
        
        if model not in self.config_data:
            self.temp_config_data = {}  
        else:
            self.temp_config_data = self.config_data.copy()

        if model != "best model":
            self.model_frame.grid()

        if model == "kmeans":
             self.show_kmeans_params()

        elif model == "agglomerative":
            self.show_agglomerative_params()

        elif model == "dbscan":
            self.show_dbscan_params()

        elif model == "optics":
            self.show_optics_params()     
        elif model == "best model":
            self.model_frame.grid_remove()
    
    def on_grid_search_toggle(self):
        model = self.model_selected.get().lower()

        if self.grid_search.get():
            if model not in self.temp_config_data:
                self.temp_config_data[model] = {}

                grid_defaults = {
                    "eps": {"min": 0.1, "max": 1.0, "step": 0.1},
                    "min_samples": {"min": 2, "max": 10, "step": 1},
                    "num_clusters": {"min": 2, "max": 10, "step": 1},
                }
                
                for param_key, default_value in grid_defaults.items():
                    if param_key not in self.temp_config_data[model]:
                        self.temp_config_data[model][f"{param_key}_range"] = default_value
        else:
            for param_key in ["n_clusters","eps","min_samples"]:
                self.temp_config_data[model].pop(f"{param_key}_range", None)

        self.on_model_change(None)  

    def show_kmeans_params(self):
        self.create_param_field("kmeans", "Number of clusters", "n_clusters", row=0)
        self.create_param_field("kmeans", "Random state", "random_state", row=1)

    def show_agglomerative_params(self):
        self.create_param_field("agglomerative", "Number of clusters", "n_clusters", row=0)
        self.create_categorical_param_field("agglomerative", "Linkage", "linkage", row=1, options=self.linkages)

    def show_dbscan_params(self):
        self.create_param_field("dbscan", "Eps ", "eps", row=0)
        self.create_param_field("dbscan", "Min samples", "min_samples", row=1)

    def show_optics_params(self):
        self.create_param_field("optics", "Min samples ", "min_samples", row=0)

    def update_int_grid_value(self, model, param_key, range_key, value):
        try:
            num_value = int(value) if value.isdigit() else float(value)
        except ValueError:
            num_value = None

        if model not in self.temp_config_data:
            self.temp_config_data[model] = {}

        if range_key:
            self.temp_config_data[model].setdefault(param_key, {})[range_key] = num_value
        else:
            self.temp_config_data[model][param_key] = num_value

    def update_range_value(self, model, param_key, range_key, value):
        try:
            num_value = int(value) if value else None
        except ValueError:
            num_value = float(value) if value else None

        if model not in self.temp_config_data:
            self.temp_config_data[model] = {}

        if range_key:
            self.temp_config_data[model].setdefault(f"{param_key}", {})[range_key] = num_value
        else:
            self.temp_config_data[model][param_key] = num_value

    def update_single_value(self, model, param_key, value):
        try:
            num_value = int(value) if value else None
        except ValueError:
            num_value = float(value) if value else None

        if model not in self.temp_config_data:
            self.temp_config_data[model] = {}

        self.temp_config_data[model][param_key] = num_value


    def create_param_field(self, model, label_text, param_key, row):

        if model not in self.temp_config_data:
            self.temp_config_data[model] = {}

        if any(key.endswith("_range") for key in self.temp_config_data[model]) and not self.grid_search.get():
            self.grid_search.set(True)


        param_label = tk.Label(self.model_frame, text=label_text)
        param_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        if param_key == "random_state":
            ToolTip(param_label, msg="Seed for random number generation, ensuring reproducibility.")
        elif param_key == "n_clusters":
            ToolTip(param_label, msg="Number of clusters to form; must be at least 2.")
        elif param_key == "min_samples":
            ToolTip(param_label, msg="Minimum points required to form a dense region; must be at least 2.")
        elif param_key == "eps":
            ToolTip(param_label, msg="Maximum distance between points to be considered neighbors.")


        if param_key=="random_state" or not self.grid_search.get():
            value = self.temp_config_data[model].get(param_key, self.default_hyperparameters.get(param_key, 5))

            self.temp_config_data[model][param_key] = value

            entry = tk.Entry(self.model_frame, width=10)
            entry.grid(row=row, column=1, padx=5, pady=5)
            entry.insert(0, str(value))
            entry.bind("<KeyRelease>", lambda event: self.update_single_value(model, param_key, entry.get()))
            return
            
        param_key = f"{param_key}_range"
        value_range = self.temp_config_data[model].get(param_key, copy.deepcopy(self.default_hyperparameters.get(param_key, {"min": 2, "max": 10, "step": 1})))

        self.temp_config_data[model].setdefault(param_key, value_range)

        labels_entries = [
        ("Min:", "min", 1, 2),
        ("Max:", "max", 3, 4),
        ("Step:", "step", 5, 6),
        ]

        for text, key, col_label, col_entry in labels_entries:
            tk.Label(self.model_frame, text=text).grid(row=row, column=col_label, padx=2, pady=5)
            entry = tk.Entry(self.model_frame, width=5)
            entry.grid(row=row, column=col_entry, padx=2, pady=5)
            entry.insert(0, str(value_range[key]))
            entry.bind("<KeyRelease>", lambda event, k=key, e=entry: self.update_range_value(model, param_key, k, e.get()))

    def update_selected_values(self, event, listbox, options, model, param_key):
        selected_indices = listbox.curselection()  
        selected_values = [options[i] for i in selected_indices]  
        self.temp_config_data[model][param_key] = selected_values  

    def update_selected_values_listbox(self, event, listbox, options, model, param_key):
        selected_indices = listbox.curselection()
        self.temp_config_data[model][param_key] = [options[i] for i in selected_indices]

    def update_selected_value_combobox(self, event, combobox, model, param_key):
        value = combobox.get()
        self.temp_config_data[model][param_key] = value
        
    def create_categorical_param_field(self, model, label_text, param_key, row, options):
        param_label = tk.Label(self.model_frame, text=label_text)
        param_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        ToolTip(param_label, msg="Select the linkage criterion for cluster merging.")

        if param_key not in self.temp_config_data[model]:
            self.temp_config_data[model][param_key] = options[0]

        value = self.temp_config_data[model][param_key]

        if self.allow_grid and self.grid_search.get():
            param_key = "linkages"
            
            self.temp_config_data[model].setdefault(param_key, [])
            
            listbox = tk.Listbox(self.model_frame, selectmode="multiple", height=len(options), exportselection=0)
            
            for option in options:
                listbox.insert(tk.END, option)
            
            listbox.grid(row=row, column=1, columnspan=2, padx=5, pady=5)

            selected_values = set(self.temp_config_data[model][param_key])  
        
            for i, option in enumerate(options):
                if option in selected_values:
                    listbox.selection_set(i)

            listbox.bind('<<ListboxSelect>>', lambda event: self.update_selected_values_listbox(event, listbox, options, model, param_key))
            return
        
        param_key = "linkage"
        
        combobox = ttk.Combobox(self.model_frame, textvariable=self.temp_config_data[model][param_key], values=options, state="readonly", width=10)
        combobox.grid(row=row, column=1, padx=5, pady=5)
        combobox.set(value)
        combobox.bind("<<ComboboxSelected>>", lambda event: self.update_selected_value_combobox(event, combobox, model, param_key))

    def validate_numeric(self, value):
        return value.isdigit() or value == "" 

    def validate_numeric_and_float(self, value):
        if value == "" or value.isdigit():
            return True
        if value.count('.') == 1 and value.replace('.', '').isdigit():
            return True
        return False
    
    def validate_form(self):

        model = self.model_selected.get().lower()

        if model == "":
            tk.messagebox.showerror("Error", "No model selected. Please select a clustering model.")
            return False

        if model in self.temp_config_data:
            params = self.temp_config_data[model]
            

            if model in ["kmeans", "agglomerative"]:
                if "n_clusters" in params:
                    n_clusters = params["n_clusters"]
                    if n_clusters == "":
                        tk.messagebox.showerror("Error", f"Number of clusters must be specified.")
                        return False
                    if n_clusters < 2:
                        tk.messagebox.showerror("Error", f"Number of clusters with {model} model must be at least 2.")
                        return False
                if "linkages" in params:
                    linkages = params["linkages"]
                    if len(linkages) == 0:
                        tk.messagebox.showerror("Error", "At least one linkage for agglomerative clustering must be selected.")
                        return False
                if "n_clusters_range" in params:
                    n_clusters_min = params["n_clusters_range"]["min"]
                    n_clusters_max = params["n_clusters_range"]["max"]
                    if n_clusters_min == "" or n_clusters_max == "" or params["n_clusters_range"]["step"] == "":
                        tk.messagebox.showerror("Error", "Cluster range values cannot be empty.")
                        return False
                    
                    if n_clusters_min < 2:
                        tk.messagebox.showerror("Error", f"Minimum number of clusters with {model} model must be at least 2.")
                        return False

                    if n_clusters_min > n_clusters_max:
                        tk.messagebox.showerror("Error", f"Min value of clusters range must be less than max value.")
                        return False
                
            if model == "dbscan":
                if "eps_range" in params:
                    min_eps_range = params["eps_range"]["min"]
                    max_eps_range = params["eps_range"]["max"]
                    
                    if min_eps_range == "" or max_eps_range == "" or params["eps_range"]["step"] == "":
                        tk.messagebox.showerror("Error", "Eps range values cannot be empty.")
                        return False

                    if min_eps_range > max_eps_range:
                        tk.messagebox.showerror("Error", f"Min value of eps range must be less than max value.")
                        return False

            if model in ["dbscan","optics"]:
                if "min_samples" in params:
                    min_samples = params["min_samples"]
                    if min_samples == "":
                        tk.messagebox.showerror("Error", "Min samples value cannot be empty.")
                        return False
            
                    if min_samples < 2:
                        tk.messagebox.showerror("Error", f"Min samples with {model} model must be at least 2.")
                        return False
                elif "min_samples_range" in params:
                    min_samples_min = params["min_samples_range"]["min"]
                    min_samples_max = params["min_samples_range"]["max"]
                    if min_samples_min == "" or min_samples_max == "" or params["min_samples_range"]["step"] == "":
                        tk.messagebox.showerror("Error", "Min samples range values cannot be empty.")
                        return False

                    if min_samples_min < 2:
                        tk.messagebox.showerror("Error", f"Minimum min samples with {model} model must be at least 2.")
                        return False
                    if min_samples_min > min_samples_max:
                        tk.messagebox.showerror("Error", f"Min value of min samples range must be less than max value.")
                        return False

        return True

    def clear_and_close_form(self):
        self.temp_config_data = {}
        self.grid_search.set(False)
        self.top.destroy()

    def update_params(self, model):
        grid_search_removals = {
            "kmeans": ["n_clusters"],
            "agglomerative": ["linkage", "n_clusters"],
            "dbscan": ["eps", "min_samples"],
            "optics": ["min_samples"]
        }

        normal_removals = {
            "kmeans": ["n_clusters_range"],
            "agglomerative": ["linkages", "n_clusters_range"],
            "dbscan": ["eps_range", "min_samples_range"],
            "optics": ["min_samples_range"]
        }

        if self.grid_search.get():
            if model in grid_search_removals:
                for key in grid_search_removals[model]:
                    self.temp_config_data[model].pop(key, None)
            return

        if model in normal_removals:
            for key in normal_removals[model]:
                self.temp_config_data[model].pop(key, None)


    def validate_and_close_form(self):
        if not self.validate_form():
            return

        model = self.model_selected.get().lower()
        
        self.update_params(model)

        self.config_data[model] = self.temp_config_data[model]

        self.top.destroy() 
           

    def delete_model(self):
        if self.model_name == "":
            tk.messagebox.showerror("Error", "You must select a model to delete.")
            return
        
        delete = tk.messagebox.askyesno("Delete model", "Are you sure you want to delete this model and all of its information?")

        if not delete:
            return
    
        deleted_model = self.model_selected.get().lower()
        self.config_data.pop(deleted_model, None)
        self.temp_config_data = {}

        self.top.destroy()
        tk.messagebox.showinfo("Delete model", f"Model {deleted_model} has been successfully deleted from the configuration.")

        