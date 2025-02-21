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
       
        self.top.geometry("1000x525")

        self.top.resizable(False, False)

        self.top.title("Clustering analysis configuration")

        self.cluster_images = {"cluster_images": True}
        self.clustering_data = {
            "embedding_model": {
                "framework": tk.StringVar(value="opencv"),
                "name": tk.StringVar(value=""),
                "resize_height": tk.StringVar(value="224"),
                "resize_width": tk.StringVar(value="224"),
                "lbp_radius": tk.StringVar(value="16"),
                "lbp_num_points": tk.StringVar(value="48"),
                "lbp_method": tk.StringVar(value="uniform"),
                
            },

            "clustering_models": {},

            "clustering_metric": tk.StringVar(value="calinski"),
            "plot": tk.BooleanVar(value=False),
            "visualization_technique": tk.StringVar(value="pca"),

            "reduction": tk.BooleanVar(value=False),
            "reduction_percentage": tk.StringVar(value="0.7"),
            "reduction_type": tk.StringVar(value="representative"),
            "diverse_percentage": tk.StringVar(value="0.0"),
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
        tk.Button(btn_frame, text="Save configuration", command=lambda: self.valid_and_save_form()).pack(side="right", padx=5)


    def clustering_config_widgets(self):
        frameworks = ConfigHandler.CONFIGURATION_VALUES['frameworks']
        lbp_methods = ConfigHandler.CONFIGURATION_VALUES['lbp_methods']
        clustering_metrics = ConfigHandler.CONFIGURATION_VALUES['clustering_metric']
        visualization_techniques = ConfigHandler.CONFIGURATION_VALUES['visualization_techniques']
        vcmd = (self.top.register(self.validate_numeric), "%P")


        self.clustering_frame = tk.LabelFrame(self.top, text="Clustering configuration", padx=10, pady=10)
        self.clustering_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(self.clustering_frame, text="Framework").grid(row=0, column=0, padx=5, sticky="w")
        self.framework_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["framework"], values=frameworks, state="readonly", width=15)
        self.framework_combobox.grid(row=1, column=0, pady=5, padx=5, sticky="w")
        
        self.name_label = tk.Label(self.clustering_frame, text="Name")
        self.name_label.grid(row=0, column=1, padx=5, sticky="w")

        self.name_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["name"])
        self.name_entry.grid(row=1, column=1, padx=5, pady=5)

        self.resize_height_label = tk.Label(self.clustering_frame, text="Resize height")
        self.resize_height_label.grid(row=0, column=1, padx=10)
        self.resize_height_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_height"], width=10, validate="key", validatecommand=vcmd)
        self.resize_height_entry.grid(row=1, column=1, padx=10, pady=5)

        self.resize_width_label = tk.Label(self.clustering_frame, text="Resize width")
        self.resize_width_label.grid(row=0, column=2, padx=10)
        self.resize_width_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_width"], width=10, validate="key", validatecommand=vcmd)
        self.resize_width_entry.grid(row=1, column=2, padx=10, pady=5)

        self.lbp_radius_label = tk.Label(self.clustering_frame, text="LBP Radius")
        self.lbp_radius_label.grid(row=0, column=3, padx=10)
        self.lbp_radius_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_radius"], width=10, validate="key", validatecommand=vcmd)
        self.lbp_radius_entry.grid(row=1, column=3, padx=10, pady=5)

        self.lbp_num_points_label = tk.Label(self.clustering_frame, text="LBP Num Points")
        self.lbp_num_points_label.grid(row=0, column=4, padx=10)
        self.lbp_num_points_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_num_points"], width=10, validate="key", validatecommand=vcmd)
        self.lbp_num_points_entry.grid(row=1, column=4, padx=10, pady=5)

        self.lbp_method_label = tk.Label(self.clustering_frame, text="LBP Method")
        self.lbp_method_label.grid(row=0, column=5, padx=10)
        self.lbp_method_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_method"], values=lbp_methods, state="readonly",  width=15)
        self.lbp_method_combobox.grid(row=1, column=5, padx=10, pady=5)

        tk.Label(self.clustering_frame, text="Clustering models").grid(row=2, column=0, sticky="ew")
        self.clustering_models_text = tk.Text(self.clustering_frame, height=5, width=15, state="disabled")
        self.clustering_models_text.grid(row=3, column=0, sticky="ew")
        tk.Label(self.clustering_frame, text="Clustering models to use with the embeddings.", fg="gray", font=("Arial", 8)).grid(row=4, column=0, columnspan=2)

        tk.Button(self.clustering_frame, text="Add model", command=lambda: self.open_model_form(self.clustering_models, "clustering_models")).grid(row=3, column=1)

        tk.Label(self.clustering_frame, text="Clustering metric").grid(row=2, column=2)
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
        self.visualization_label.grid(row=2, column=5)
        self.visualization_technique_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data["visualization_technique"], values=visualization_techniques, state="readonly", width=6)
        self.visualization_technique_combobox.grid(row=3, column=5, padx=5, pady=5)       
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

        tk.Label(self.reduction_frame, text="Reduction type").grid(row=0, column=0, padx=5, sticky="w")
        ttk.Combobox(self.reduction_frame, textvariable=self.clustering_data["reduction_type"], values=reduction_types, state="readonly", width=15).grid(row=1, column=0, padx=5, sticky="w")        

        tk.Label(self.reduction_frame, text="Reduction percentage").grid(row=0, column=1, padx=5, sticky="w")
        tk.Entry(self.reduction_frame, textvariable=self.clustering_data['reduction_percentage'], width=10).grid(row=1, column=1, padx=7)

        tk.Label(self.reduction_frame, text="Diverse percentage").grid(row=0, column=2, padx=10, sticky="w")
        tk.Entry(self.reduction_frame, textvariable=self.clustering_data['diverse_percentage'], width=10).grid(row=1, column=2, padx=7)

        reduction_model_name = self.get_reduction_model_name()
        tk.Label(self.reduction_frame, text="Reduction model").grid(row=0, column=3, padx=10, sticky="w")
        self.reduction_model_entry = tk.Entry(self.reduction_frame, textvariable=reduction_model_name, width=15, state="readonly")
        self.reduction_model_entry.grid(row=1, column=3, padx=7)

        tk.Button(self.reduction_frame, text="Add model", command=lambda: self.open_model_form(self.reduction_models, "reduction_model", False)).grid(row=1, column=4)

        tk.Label(self.reduction_frame, text="Use reduced dataset").grid(row=0, column=5, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.reduction_frame, variable=self.clustering_data["use_reduced"]).grid(row=1, column=5, padx=5, pady=5)

        tk.Label(self.reduction_frame, text="Include outliers").grid(row=0, column=6, padx=5, pady=5)
        ttk.Checkbutton(self.reduction_frame, variable=self.clustering_data["include_outliers"]).grid(row=1, column=6, padx=5, pady=5)
        tk.Label(self.reduction_frame, text="Include outlier images.\nThese images are marked with the label -1.", fg="gray", font=("Arial", 8), wraplength=200).grid(row=2, column=6, padx=5, pady=5)

    def get_reduction_model_name(self):
        if isinstance(self.clustering_data['reduction_model'], dict) and self.clustering_data['reduction_model']:
            return list(self.clustering_data['reduction_model'].keys())[0]
        return ''

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
            self.resize_height_label.grid(row=0, column=1, padx=10)
            self.resize_height_entry.grid(row=1, column=1, padx=10, pady=5)
            self.resize_width_label.grid(row=0, column=2, padx=10)
            self.resize_width_entry.grid(row=1, column=2, padx=10, pady=5)
            self.lbp_radius_label.grid(row=0, column=3, padx=10)
            self.lbp_radius_entry.grid(row=1, column=3, padx=10, pady=5)
            self.lbp_num_points_label.grid(row=0, column=4, padx=10)
            self.lbp_num_points_entry.grid(row=1, column=4, padx=10, pady=5)
            self.lbp_method_label.grid(row=0, column=5, padx=10)
            self.lbp_method_combobox.grid(row=1, column=5, padx=10, pady=5)

        elif framework == "huggingface":
            self.name_label.grid(row=0, column=1, padx=10)
            self.name_entry.grid(row=1, column=1, padx=10, pady=5)

        else:
            self.name_label.grid(row=0, column=1, padx=10)
            self.name_entry.grid(row=1, column=1, padx=10, pady=5)
            self.resize_height_label.grid(row=0, column=2, padx=10)
            self.resize_height_entry.grid(row=1, column=2, padx=10, pady=5)
            self.resize_width_label.grid(row=0, column=3, padx=10)
            self.resize_width_entry.grid(row=1, column=3, padx=10, pady=5)

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
            
            if model_name != "" or model_name is not None:
                if model_name == "best_model":
                    self.clustering_data[param_key] = "best_model"
                else:
                    self.clustering_data[param_key] = {model_name: model_params[model_name]}
                
            self.reduction_model_entry.config(state="normal")
            self.reduction_model_entry.delete(0, tk.END)
            self.reduction_model_entry.insert(tk.END, model_name)
            self.reduction_model_entry.config(state="readonly")     

    def validate_numeric(self, value):
        return value.isdigit() or value == ""   

    def validate_form(self):
        errors = []
        warnings = []
        default_model_values = {
            "resize_height": 224,
            "resize_width": 224,
            "lbp_radius": 16,
            "lbp_num_points": 48
        }
        default_reduction_values = {
            "reduction_percentage": 0.7,
            "diverse_percentage": 0.0
        }

        if self.name_entry.winfo_ismapped() and not self.clustering_data['embedding_model']["name"].get().strip():
            errors.append("The 'Name' field is required.")

        if not self.clustering_data['clustering_models']:
            errors.append("At least one clustering model must be defined.")

        if self.clustering_data['reduction'].get():
            if not self.clustering_data['reduction_model']:
                errors.append("At least one reduction model must be defined.")

            reduction_percentage = self.clustering_data['reduction_percentage'].get()
            diverse_percentage = self.clustering_data['diverse_percentage'].get()

            if reduction_percentage == "":
                warnings.append(f"The 'Reduction Percentage' field can't be empty. Default value ({default_reduction_values['reduction_percentage']} will be used.)")
                self.clustering_data['reduction_percentage'].set(default_reduction_values['reduction_percentage'])  

            if diverse_percentage == "":
                warnings.append(f"The 'Diverse Percentage' field can't be empty. Default value ({default_reduction_values['diverse_percentage']} will be used.)")
                self.clustering_data['diverse_percentage'].set(default_reduction_values['diverse_percentage'])

            try:
                reduction_percentage = float(reduction_percentage) if reduction_percentage != "" else default_reduction_values['reduction_percentage']
                diverse_percentage = float(diverse_percentage) if diverse_percentage != "" else default_reduction_values['diverse_percentage']

                if not (0 < reduction_percentage < 1):
                    errors.append("The 'Reduction Percentage' field must be greater than 0 and less than 1.")
                if not (0 <= diverse_percentage < 1):
                    errors.append("The 'Diverse Percentage' field can't be negative and must be less than 1.")
            except ValueError:
                errors.append("The 'Reduction Percentage' and 'Diverse Percentage' must be numeric values.")

        
        for key, default in default_model_values.items():
            field = self.clustering_data['embedding_model'][key]
            entry_widget = getattr(self, f"{key}_entry")  

            if entry_widget.winfo_ismapped():  
                field_value = field.get().strip()
                
                if field_value == "" or field_value is None:  
                    field.set(default)
                    warnings.append(f"'{key.replace('_', ' ').title()}' is empty. Default value ({default}) will be used.")
                    field_value = default
                else:
                    field_value = int(field_value)
                    field.set(field_value)

                if field_value < 0:
                    errors.append(f"'{key.replace('_', ' ').title()}' cannot be negative.")
            
        if errors:
            tk.messagebox.showerror("Error", "\n".join(errors))
            return False
        
        if warnings:
            tk.messagebox.showwarning("Warning", "\n".join(warnings))

        return True
        
    def valid_and_save_form(self):
        result_ok = self.validate_form()

        if not result_ok:
            return
        
        self.save_config()
        self.top.destroy()
        tk.messagebox.showinfo("Clustering configuration", "Configuration saved successfully.")
        

    def save_config(self):
        for category, category_data in list(self.clustering_data.items()):
            if isinstance(category_data, dict):
                for key, var in list(category_data.items()):
                    if isinstance(var, tk.StringVar):
                        value = var.get().strip()

                        if key in ["resize_height", "resize_width", "lbp_radius", "lbp_num_points"]:
                            self.clustering_data[category][key] = int(value) 
                        else:
                            self.clustering_data[category][key] = value or ""

                    elif isinstance(var, tk.BooleanVar):
                        self.clustering_data[category][key] = var.get()

                    else:
                        self.clustering_data[category][key] = var

            else:
                if category in ["reduction_percentage", "diverse_percentage"]:
                    value = category_data.get()
                    self.clustering_data[category] = float(value) 
                else:
                    if isinstance(category_data, tk.StringVar):
                        value = category_data.get().strip()
                        self.clustering_data[category] = value if value else ""

                    elif isinstance(category_data, tk.BooleanVar):
                        self.clustering_data[category] = category_data.get()
                    else:
                        self.clustering_data[category] = category_data

    def clear_and_close_form(self):

        self.cluster_images = {"cluster_images": False}
        self.top.destroy()

