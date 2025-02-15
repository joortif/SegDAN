import tkinter as tk
from tkinter import ttk

from src.forms.formutils import FormUtils
from src.forms.clusteringmodelform import ClusteringModelForm
from src.utils.confighandler import ConfigHandler

class ClusteringConfigForm():

    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.clustering_models = ConfigHandler.CONFIGURATION_VALUES["clustering_models"]
        self.reduction_models = ConfigHandler.CONFIGURATION_VALUES["reduction_models"]
       
        self.top.title("Clustering analysis configuration")

        self.cluster_images = {"cluster_images": True}
        self.clustering_data = {
            "embedding_model": {
                "framework": tk.StringVar(value="opencv"),
                "name": tk.StringVar(value=""),
                "resize_height": tk.IntVar(value=224),
                "resize_width": tk.IntVar(value=224),
                "lbp_radius": tk.IntVar(value=16),
                "lbp_num_points": tk.IntVar(value=48),
                "lbp_method": tk.StringVar(value="uniform"),
                
            },

            "clustering_models": {},

            "clustering_metric": tk.StringVar(value="calinski"),
            "plot": tk.BooleanVar(value=False),
            "visualization_technique": tk.StringVar(value="pca"),

            "reduction": tk.BooleanVar(value=False),
            "reduction_percentage": tk.DoubleVar(value=0.0),
            "reduction_type": tk.StringVar(value="representative"),
            "diverse_percentage": tk.DoubleVar(value=0.0),
            "include_outliers": tk.BooleanVar(value=False),
            "reduction_model": {},
            "use_reduced": tk.BooleanVar(value=False)
            }

        self.create_widgets()

    def create_widgets(self):
        self.clustering_config_widgets()
        self.reduction_config_widgets()

        btn_frame = tk.Frame(self.top)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Cancel", command=self.clear_and_close_form).pack(side="right", padx=5)
        tk.Button(btn_frame, text="Save configuration", command=self.save_config).pack(side="right", padx=5)


    def clustering_config_widgets(self):
        frameworks = ConfigHandler.CONFIGURATION_VALUES['frameworks']
        lbp_methods = ConfigHandler.CONFIGURATION_VALUES['lbp_methods']
        clustering_metrics = ConfigHandler.CONFIGURATION_VALUES['clustering_metric']
        visualization_techniques = ConfigHandler.CONFIGURATION_VALUES['visualization_techniques']
        
        self.clustering_frame = tk.LabelFrame(self.top, text="Clustering configuration", padx=10, pady=10)
        self.clustering_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(self.clustering_frame, text="Framework").grid(row=0, column=0, padx=5, sticky="w")
        self.framework_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["framework"], values=frameworks, state="readonly", width=15)
        self.framework_combobox.grid(row=1, column=0, pady=5, padx=5, sticky="w")
        
        self.name_label = tk.Label(self.clustering_frame, text="Name")
        self.name_label.grid(row=0, column=1, padx=5, sticky="w")

        self.name_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["name"])
        self.name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.resize_height_label = tk.Label(self.clustering_frame, text="Resize height")
        self.resize_height_label.grid(row=0, column=1, padx=10, sticky="w")
        self.resize_height_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_height"], width=10)
        self.resize_height_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.resize_width_label = tk.Label(self.clustering_frame, text="Resize width")
        self.resize_width_label.grid(row=0, column=2, padx=10, sticky="w")
        self.resize_width_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_width"], width=10)
        self.resize_width_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        self.lbp_radius_label = tk.Label(self.clustering_frame, text="LBP Radius")
        self.lbp_radius_label.grid(row=0, column=3, padx=10, sticky="w")
        self.lbp_radius_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_radius"], width=10)
        self.lbp_radius_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")

        self.lbp_num_points_label = tk.Label(self.clustering_frame, text="LBP Num Points")
        self.lbp_num_points_label.grid(row=0, column=4, padx=10, sticky="w")
        self.lbp_num_points_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_num_points"], width=10)
        self.lbp_num_points_entry.grid(row=1, column=4, padx=10, pady=5, sticky="w")

        self.lbp_method_label = tk.Label(self.clustering_frame, text="LBP Method")
        self.lbp_method_label.grid(row=0, column=5, padx=10, sticky="w")
        self.lbp_method_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_method"], values=lbp_methods, state="readonly",  width=15)
        self.lbp_method_combobox.grid(row=1, column=5, padx=10, pady=5, sticky="w")

        tk.Label(self.clustering_frame, text="Clustering models").grid(row=2, column=0, sticky="w")
        self.clustering_models_text = tk.Text(self.clustering_frame, height=5, width=15, state="disabled")
        self.clustering_models_text.grid(row=3, column=0, sticky="w")
        tk.Label(self.clustering_frame, text="Clustering models to use with the embeddings.", fg="gray", font=("Arial", 8)).grid(row=4, column=0)

        tk.Button(self.clustering_frame, text="Add model", command=lambda: self.open_model_form(self.clustering_models, "clustering_models")).grid(row=3, column=1)

        tk.Label(self.clustering_frame, text="Clustering metric").grid(row=2, column=2, sticky="w")
        ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data["clustering_metric"], values=clustering_metrics, state="readonly", width=10).grid(row=3, column=2)        
        tk.Label(self.clustering_frame, text="Clustering models to use with the embeddings.", fg="gray", font=("Arial", 8), wraplength=200).grid(row=4, column=2)

        tk.Label(self.clustering_frame, text="Reduction").grid(row=2, column=3, sticky="ew", padx=5, pady=5)
        ttk.Checkbutton(self.clustering_frame, variable=self.clustering_data["reduction"], command= lambda: FormUtils.toggle_widget(self.reduction_frame, self.clustering_data['reduction'])).grid(row=3, column=3, padx=5, pady=5)
        tk.Label(self.clustering_frame, text="Enables dataset reduction using clustering information.", fg="gray", font=("Arial", 8), wraplength=200).grid(row=4, column=3, sticky="w", padx=5, pady=5)

        tk.Label(self.clustering_frame, text="Plot").grid(row=2, column=4, sticky="ew", padx=5, pady=5)
        ttk.Checkbutton(self.clustering_frame, variable=self.clustering_data["plot"], command=lambda: FormUtils.toggle_label_entry(self.clustering_data["plot"], self.visualization_label, 
                                                                                                                          self.visualization_technique_combobox, self.visualization_comment, 2, 5)).grid(row=3, column=4, padx=5, pady=5)
        tk.Label(self.clustering_frame, text="Enable plotting of the results.", fg="gray", font=("Arial", 8), wraplength=200).grid(row=4, column=4, padx=5, pady=5)

        self.visualization_label = tk.Label(self.clustering_frame, text="Visualization technique")
        self.visualization_label.grid(row=2, column=5, sticky="w")
        self.visualization_technique_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data["visualization_technique"], values=visualization_techniques, state="readonly", width=6)
        self.visualization_technique_combobox.grid(row=3, column=5, padx=5, pady=5, sticky="w")       
        self.visualization_comment = tk.Label(self.clustering_frame, text="Dimensionality reduction technique to be used for plotting.", fg="gray", font=("Arial", 8), wraplength=200)
        self.visualization_comment.grid(row=4, column=5, sticky="w", padx=5, pady=5)

        self.framework_combobox.bind("<<ComboboxSelected>>", self.update_framework_fields)

        self.name_label.grid_forget()
        self.name_entry.grid_forget()
        self.visualization_label.grid_forget()
        self.visualization_technique_combobox.grid_forget()
        self.visualization_comment.grid_forget()

    def reduction_config_widgets(self):
        reduction_types = ConfigHandler.CONFIGURATION_VALUES['reduction_type']
        self.reduction_frame = tk.LabelFrame(self.top, text="Reduction configuration", padx=10, pady=10)
        self.reduction_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.reduction_frame.grid_remove()

        tk.Label(self.reduction_frame, text="Reduction type").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Combobox(self.reduction_frame, textvariable=self.clustering_data["reduction_type"], values=reduction_types, state="readonly", width=15).grid(row=1, column=0, padx=5, sticky="w")        

        tk.Label(self.reduction_frame, text="Reduction percentage").grid(row=0, column=1, padx=5, sticky="w")
        tk.Entry(self.reduction_frame, textvariable=self.clustering_data['reduction_percentage'], width=10).grid(row=1, column=1, padx=7, sticky="w")

        tk.Label(self.reduction_frame, text="Diverse percentage").grid(row=0, column=2, padx=10, sticky="w")
        tk.Entry(self.reduction_frame, textvariable=self.clustering_data['diverse_percentage'], width=10).grid(row=1, column=2, padx=7, sticky="w")

        tk.Label(self.reduction_frame, text="Reduction model").grid(row=0, column=3, padx=10, sticky="w")
        self.reduction_model_entry = tk.Entry(self.reduction_frame, textvariable=self.clustering_data['reduction_model'], width=15, state="readonly")
        self.reduction_model_entry.grid(row=1, column=3, padx=7, sticky="w")

        tk.Button(self.reduction_frame, text="Add model", command=lambda: self.open_model_form(self.reduction_models, "reduction_model", False)).grid(row=1, column=4)

        tk.Label(self.reduction_frame, text="Use reduced dataset").grid(row=0, column=5, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.reduction_frame, variable=self.clustering_data["use_reduced"]).grid(row=1, column=5, padx=5, pady=5)

    def update_framework_fields(self, event):
        framework = self.clustering_data['embedding_model']["framework"].get()

        self.name_label.grid_remove()
        self.name_entry.grid_remove()
        self.resize_height_label.grid_remove()
        self.resize_height_entry.grid_remove()
        self.resize_width_label.grid_remove()
        self.resize_width_entry.grid_remove()
        self.lbp_radius_label.grid_remove()
        self.lbp_radius_entry.grid_remove()
        self.lbp_num_points_label.grid_remove()
        self.lbp_num_points_entry.grid_remove()
        self.lbp_method_label.grid_remove()
        self.lbp_method_combobox.grid_remove()

        if framework == "opencv":
            self.resize_height_label.grid(row=0, column=1, padx=10, sticky="w")
            self.resize_height_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
            self.resize_width_label.grid(row=0, column=2, padx=10, sticky="w")
            self.resize_width_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
            self.lbp_radius_label.grid(row=0, column=3, padx=10, sticky="w")
            self.lbp_radius_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
            self.lbp_num_points_label.grid(row=0, column=4, padx=10, sticky="w")
            self.lbp_num_points_entry.grid(row=1, column=4, padx=10, pady=5, sticky="w")
            self.lbp_method_label.grid(row=0, column=5, padx=10, sticky="w")
            self.lbp_method_combobox.grid(row=1, column=5, padx=10, pady=5, sticky="w")

        elif framework == "huggingface":
            self.name_label.grid(row=0, column=1, padx=10, sticky="w")
            self.name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        else:
            self.name_label.grid(row=0, column=1, padx=10, sticky="w")
            self.name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
            self.resize_height_label.grid(row=0, column=2, padx=10, sticky="w")
            self.resize_height_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
            self.resize_width_label.grid(row=0, column=3, padx=10, sticky="w")
            self.resize_width_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")

    def open_model_form(self, models, param_key, allow_grid=True):
        
        model_form = ClusteringModelForm(self.top, allow_grid=allow_grid, models=models)

        model_form.top.transient(self.top)
        model_form.top.grab_set()

        self.top.wait_window(model_form.top)

        if model_form.config_data:
            self.add_model(model_form.config_data, param_key)

    def add_model(self, model_params, param_key):
        model_name = list(model_params.keys())[0]
        
        if param_key == 'clustering_models':
            self.clustering_data[param_key][model_name] = model_params[model_name]
            
            model_names = list(self.clustering_data[param_key].keys())
            models_text = "\n".join(model_names)
            
            self.clustering_models_text.config(state="normal")  
            self.clustering_models_text.delete(1.0, tk.END)  
            self.clustering_models_text.insert(tk.END, models_text)
            self.clustering_models_text.config(state="disabled")

        elif param_key == 'reduction_model':
            if model_name == "best_model":
                self.clustering_data[param_key] = "best_model"
            else:
                self.clustering_data[param_key] = {model_name: model_params[model_name]}
            
            self.reduction_model_entry.config(state="normal")
            self.reduction_model_entry.delete(0, tk.END)
            self.reduction_model_entry.insert(tk.END, model_name)
            self.reduction_model_entry.config(state="readonly")        

    def save_config(self):
        config_data = {}

        for key, value in self.clustering_data.items():
            if isinstance(value, dict):
                config_data[key] = {sub_key: sub_value.get() if isinstance(sub_value, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)) else sub_value
                                    for sub_key, sub_value in value.items()}
            else:
                config_data[key] = value.get() if isinstance(value, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)) else value

        print(config_data)

    def clear_and_close_form(self):

        self.clustering_data = {}
        self.cluster_images = {"cluster_images": False}

        self.top.destroy()

