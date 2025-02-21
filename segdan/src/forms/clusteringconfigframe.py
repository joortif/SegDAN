import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip

from src.utils.confighandler import ConfigHandler
from src.forms.formutils import FormUtils
from src.forms.clusteringmodelform import ClusteringModelForm

class ClusteringConfigFrame(ttk.Frame):
    def __init__(self, parent, controller, config_data):
            ttk.Frame.__init__(self, parent)
            self.top = parent

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
            self.config_data = config_data
            self.controller = controller
            self.row=0

            label_title = ttk.Label(self, text="Image embedding and clustering", font=("Arial", 18, "bold"))
            label_title.grid(row=self.row, column=0, columnspan=5, pady=(20,10), padx=10)
            
            self.clustering_frame = tk.Frame(self, padx=10, pady=10)
            self.clustering_frame.grid(row=self.row+1, column=0, padx=10, pady=10)

            self.clustering_config_widgets()

            button_frame = ttk.Frame(self.clustering_frame)
            button_frame.grid(row=7, column=0, columnspan=5, pady=10)  

            self.clustering_frame.grid_rowconfigure(0, weight=0)
            self.clustering_frame.grid_columnconfigure(0, weight=1)

            button_back = ttk.Button(button_frame, text="Back", command=lambda: controller.show_frame("ClusteringFrame"))
            button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

            button_next = ttk.Button(button_frame, text="Next", command=self.next)
            button_next.grid(row=0, column=1, padx=50, pady=5, sticky="e")

            button_frame.grid_columnconfigure(0, weight=0)
            button_frame.grid_columnconfigure(1, weight=0)

    def clustering_config_widgets(self):
        frameworks = ConfigHandler.CONFIGURATION_VALUES['frameworks']

        lbp_methods = ConfigHandler.CONFIGURATION_VALUES['lbp_methods']
        clustering_metrics = ConfigHandler.CONFIGURATION_VALUES['clustering_metric']
        visualization_techniques = ConfigHandler.CONFIGURATION_VALUES['visualization_techniques']

        vcmd = (self.top.register(self.validate_numeric), "%P")

        self.framework_label = tk.Label(self.clustering_frame, text="Framework")
        self.framework_label.grid(row=self.row, column=0, padx=5)
        ToolTip(self.framework_label, msg="Select the framework used for image embeddings.")
        
        self.name_label = tk.Label(self.clustering_frame, text="Name")
        self.name_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["name"], state="readonly", width=35)
        ToolTip(self.name_label, msg="Embedding model name.")
        self.framework_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["framework"], values=frameworks, state="readonly", width=15)

        self.resize_height_label = tk.Label(self.clustering_frame, text="Resize height")  
        self.resize_height_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_height"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.resize_height_label, msg="Image resizing height before applying the embedding model.")

        self.resize_width_label = tk.Label(self.clustering_frame, text="Resize width")        
        self.resize_width_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["resize_width"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.resize_width_label, msg="Image resizing width before applying the embedding model.")

        self.lbp_radius_label = tk.Label(self.clustering_frame, text="LBP Radius")        
        self.lbp_radius_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_radius"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.lbp_radius_label, msg="Defines the neighborhood radius for LBP feature extraction.\nHigher values capture texture at a larger scale.")

        self.lbp_num_points_label = tk.Label(self.clustering_frame, text="LBP Num Points")       
        self.lbp_num_points_entry = tk.Entry(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_num_points"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.lbp_num_points_label, msg="Number of sampling points around the defined radius.\nMore points capture finer texture details.")

        self.lbp_method_label = tk.Label(self.clustering_frame, text="LBP Method")
        self.lbp_method_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data['embedding_model']["lbp_method"], values=lbp_methods, state="readonly",  width=15)
        ToolTip(self.lbp_method_label, msg="LBP computation method.\nEach method affects how texture patterns are generated.")

        self.visualization_label = tk.Label(self.clustering_frame, text="Visualization technique")
        self.visualization_technique_combobox = ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data["visualization_technique"], values=visualization_techniques, state="readonly", width=6)        
        ToolTip(self.visualization_label, msg="Dimensionality reduction technique applied to the embeddings for plotting in 2D.")

        self.clustering_models_text = tk.Text(self.clustering_frame, height=5, width=15, state="disabled")

        self.name_label.grid(row=self.row, column=1, padx=5)       
        self.resize_height_label.grid(row=self.row, column=1, padx=10)
        self.resize_width_label.grid(row=self.row, column=2, padx=10)
        self.lbp_radius_label.grid(row=self.row, column=3, padx=10)
        self.lbp_num_points_label.grid(row=self.row, column=4, padx=10)
        self.lbp_method_label.grid(row=self.row, column=5, padx=10)

        self.row+=1

        self.framework_combobox.grid(row=self.row, column=0, pady=5, padx=5)
        self.name_combobox.grid(row=self.row, column=1, padx=5, pady=5)      
        self.resize_height_entry.grid(row=self.row, column=1, padx=10, pady=5)
        self.resize_width_entry.grid(row=self.row, column=2, padx=10, pady=5)
        self.lbp_radius_entry.grid(row=self.row, column=3, padx=10, pady=5)
        self.lbp_num_points_entry.grid(row=self.row, column=4, padx=10, pady=5)
        self.lbp_method_combobox.grid(row=self.row, column=5, padx=10, pady=5)

        self.row+=1

        self.clustering_models_label = tk.Label(self.clustering_frame, text="Clustering models")
        self.clustering_models_label.grid(row=self.row, column=0, padx=10, pady=5)
        ToolTip(self.clustering_models_label, msg="Clustering models list to use with the embeddings.")

        self.clustering_metric_label = tk.Label(self.clustering_frame, text="Clustering metric")
        self.clustering_metric_label.grid(row=self.row, column=3, padx=10, pady=5)

        ToolTip(self.clustering_metric_label, msg="Metric that defines how similarity between image clusters is measured.")
        
        self.row+=1

        self.clustering_models_text.grid(row=self.row, column=0, padx=10, pady=5)
        
        self.add_model_bt = tk.Button(self.clustering_frame, text="Add model", command=lambda: self.open_model_form())
        self.add_model_bt.grid(row=self.row, column=1)
        ToolTip(self.add_model_bt, msg="Add a new clustering model.")
        ttk.Combobox(self.clustering_frame, textvariable=self.clustering_data["clustering_metric"], values=clustering_metrics, state="readonly", width=10).grid(row=self.row, column=3)        
        self.visualization_label.grid(row=self.row, column=1)

        self.edit_model_bt = tk.Button(self.clustering_frame, text="Edit model", command=lambda: self.open_model_form(edit=True))
        self.edit_model_bt.grid(row=self.row, column=2)
        ToolTip(self.edit_model_bt, msg="Edit an existing clustering model.")

        self.edit_model_bt.grid_remove()

        self.row+=1

        self.clustering_plot_label = tk.Label(self.clustering_frame, text="Plot")
        self.clustering_plot_label.grid(row=self.row, column=0, padx=10, pady=5)
        ToolTip(self.clustering_plot_label, msg="Enables visualization of clustering results by reducing the dimensionality of the embeddings.\nPlots images in a 2 dimensional space.")

        self.row+=1

        ttk.Checkbutton(self.clustering_frame, variable=self.clustering_data["plot"], command=lambda: FormUtils.toggle_label_entry(self.clustering_data["plot"], self.visualization_label, 
                                                                                                                          self.visualization_technique_combobox, None, self.row-1, 1)).grid(row=self.row, column=0, padx=5, pady=5)
        self.visualization_technique_combobox.grid(row=self.row, column=1, padx=10, pady=5)

        self.framework_combobox.bind("<<ComboboxSelected>>", self.update_framework_fields)

        self.name_label.grid_forget()
        self.name_combobox.grid_forget()
        self.visualization_label.grid_forget()
        self.visualization_technique_combobox.grid_forget()

    def update_framework_fields(self, event):
        framework = self.clustering_data['embedding_model']["framework"].get()
        self.name_combobox.set('')
        hf_models = ConfigHandler.CONFIGURATION_VALUES['huggingface_models']
        py_models = ConfigHandler.CONFIGURATION_VALUES['pytorch_models']
        tf_models = ConfigHandler.CONFIGURATION_VALUES['tensorflow_models']
        model_list = []

        self.name_label.grid_remove()
        self.name_combobox.grid_remove()
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
        self.name_combobox.grid_remove()

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
            return

        elif framework == "huggingface":
            
            model_list = hf_models
            self.name_label.grid(row=0, column=1, padx=10)
            self.name_combobox.grid(row=1, column=1, padx=10, pady=5)

        else:

            if framework == "pytorch":
                model_list = py_models
            elif framework == "tensorflow":
                model_list = tf_models 

            self.name_label.grid(row=0, column=1, padx=10)
            self.name_combobox.grid(row=1, column=1, padx=10, pady=5)
            self.resize_height_label.grid(row=0, column=2, padx=10)
            self.resize_height_entry.grid(row=1, column=2, padx=10, pady=5)
            self.resize_width_label.grid(row=0, column=3, padx=10)
            self.resize_width_entry.grid(row=1, column=3, padx=10, pady=5)

        self.name_combobox["values"] = model_list
        return

    def validate_numeric(self, value):
         return value.isdigit() or value == ""
    
    def open_model_form(self, edit=False):

        added_models = self.clustering_data["clustering_models"].keys()

        if not edit:
            available_models = [model for model in ConfigHandler.CONFIGURATION_VALUES["clustering_models"] if model not in added_models]

            if not available_models:
                tk.messagebox.showinfo("Info", "All available models have already been added.")
                return
                
        else:
            available_models = [model for model in ConfigHandler.CONFIGURATION_VALUES["clustering_models"] if model in added_models]
            if not available_models:
                tk.messagebox.showinfo("Info", "No models have been added.")
                return


        model_form = ClusteringModelForm(self.top, allow_grid=True, models=available_models, config_data=self.clustering_data["clustering_models"], edit=edit)

        model_form.top.transient(self.top)
        model_form.top.grab_set()

        self.top.wait_window(model_form.top)

        if model_form.config_data:
            self.add_model(model_form.config_data, "clustering_models")
            return
        
        self.clustering_models_text.config(state="normal")  
        self.clustering_models_text.delete(1.0, tk.END) 
        self.clustering_models_text.config(state="disabled")

    def add_model(self, model_params, param_key):

        model_name = list(model_params.keys())[0]
        
        self.clustering_data[param_key][model_name] = model_params[model_name]
            
        model_names = list(self.clustering_data[param_key].keys())
        models_text = "\n".join(model_names)
            
        self.clustering_models_text.config(state="normal")  
        self.clustering_models_text.delete(1.0, tk.END)  
        self.clustering_models_text.insert(tk.END, models_text)
        self.clustering_models_text.config(state="disabled")

        if model_names:
            self.edit_model_bt.grid()
        else:
            self.edit_model_bt.grid_remove()

    def next(self):
        pass