import tkinter as tk
from tkinter import ttk, font
import huggingface_hub
from tktooltip import ToolTip
import webbrowser

from src.utils.confighandler import ConfigHandler
from src.forms.formutils import FormUtils
from src.forms.clusteringmodelform import ClusteringModelForm


import torchvision.models as models
import tensorflow as tf


class ClusteringConfigFrame(ttk.Frame):
    def __init__(self, parent, controller, config_data, final_dict):
            ttk.Frame.__init__(self, parent)
            self.top = parent
            self.config_data = config_data

            if "clustering_data" not in self.config_data:
                self.config_data["clustering_data"] = {
                    "embedding_model": {
                        "framework": self.config_data.get("framework", tk.StringVar(value="opencv")),
                        "name": self.config_data.get("name", tk.StringVar(value="")),
                        "name_other": self.config_data.get("name_other", tk.StringVar(value="")),
                        "resize_height": self.config_data.get("resize_height", tk.StringVar(value="224")),
                        "resize_width": self.config_data.get("resize_width", tk.StringVar(value="224")),
                        "embedding_batch_size": self.config_data.get("embedding_batch_size", tk.StringVar(value="8")), 
                        "lbp_radius": self.config_data.get("lbp_radius", tk.StringVar(value="16")),
                        "lbp_num_points": self.config_data.get("lbp_num_points", tk.StringVar(value="48")),
                        "lbp_method": self.config_data.get("lbp_method", tk.StringVar(value="uniform")),
                    },
                    "clustering_models": self.config_data.get("clustering_models", {}),
                    "clustering_metric": self.config_data.get("clustering_metric", tk.StringVar(value="calinski")),
                    "plot": self.config_data.get("plot", tk.BooleanVar(value=False)),
                    "visualization_technique": self.config_data.get("visualization_technique", tk.StringVar(value="pca")),
                }
            self.clustering_data = self.config_data["clustering_data"]
            self.controller = controller

            self.grid_rowconfigure(0, weight=0)
            self.grid_columnconfigure(0, weight=1)

            label_title = ttk.Label(self, text="Image embedding and clustering", font=("Arial", 18, "bold"))
            label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
            
            self.clustering_frame = tk.Frame(self, padx=10, pady=10)
            self.clustering_frame.grid(row=1, column=0, padx=10, pady=10)

            self.clustering_config_widgets()

            button_frame = ttk.Frame(self.clustering_frame)
            button_frame.grid(row=9, column=0, columnspan=5, pady=10, sticky="e")  

            self.clustering_frame.grid_rowconfigure(0, weight=0)
            self.clustering_frame.grid_columnconfigure(0, weight=0)
            self.clustering_frame.grid_columnconfigure(1, weight=0)


            button_back = ttk.Button(button_frame, text="Back", command=lambda: controller.show_frame("ClusteringFrame"))
            button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

            button_next = ttk.Button(button_frame, text="Next", command=self.next)
            button_next.grid(row=0, column=1, pady=5, sticky="e")

            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)

    def clustering_config_widgets(self):
        frameworks = ConfigHandler.CONFIGURATION_VALUES['frameworks']

        lbp_methods = ConfigHandler.CONFIGURATION_VALUES['lbp_methods']
        clustering_metrics = ConfigHandler.CONFIGURATION_VALUES['clustering_metric']
        visualization_techniques = ConfigHandler.CONFIGURATION_VALUES['visualization_techniques']

        vcmd = (self.top.register(self.validate_numeric), "%P")

        self.embedding_model_labelframe = ttk.LabelFrame(self.clustering_frame, text="Embedding model configuration", padding=(20, 10))

        self.framework_label = tk.Label(self.embedding_model_labelframe, text="Framework *")
        self.framework_label.grid(row=0, column=0, padx=5)
        ToolTip(self.framework_label, msg="Select the framework used for image embeddings.")
        
        self.name_label = tk.Label(self.embedding_model_labelframe, text="Name *")
        self.name_combobox = ttk.Combobox(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["name"], state="readonly")

        self.lbp_name_entry = tk.Entry(self.embedding_model_labelframe, textvariable=tk.StringVar(value="Local Binary Pattern (LBP)"), state="readonly", width=25)
        ToolTip(self.name_label, msg="Embedding model name.")
        self.framework_combobox = ttk.Combobox(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["framework"], values=frameworks, state="readonly", width=15)

        self.other_name_label = tk.Label(self.embedding_model_labelframe, text="Custom model name *")
        self.other_name_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["name_other"])
        ToolTip(self.other_name_label, msg="Custom embedding model name.")

        self.custom_model_info_label = tk.Label(self.embedding_model_labelframe)

        self.resize_height_label = tk.Label(self.embedding_model_labelframe, text="Resize height *")  
        self.resize_height_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["resize_height"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.resize_height_label, msg="Image resizing height before applying the embedding model.")

        self.resize_width_label = tk.Label(self.embedding_model_labelframe, text="Resize width *")        
        self.resize_width_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["resize_width"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.resize_width_label, msg="Image resizing width before applying the embedding model.")

        self.batch_size_label = tk.Label(self.embedding_model_labelframe, text="Batch size *")  
        self.batch_size_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["embedding_batch_size"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.batch_size_label, msg="Defines the number of samples per training step. Larger values need more memory.")

        self.lbp_radius_label = tk.Label(self.embedding_model_labelframe, text="Radius *")        
        self.lbp_radius_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["lbp_radius"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.lbp_radius_label, msg="Defines the neighborhood radius for LBP feature extraction.\nHigher values capture texture at a larger scale.")

        self.lbp_num_points_label = tk.Label(self.embedding_model_labelframe, text="Num Points *")       
        self.lbp_num_points_entry = tk.Entry(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["lbp_num_points"], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.lbp_num_points_label, msg="Number of sampling points around the defined radius.\nMore points capture finer texture details.")

        self.lbp_method_label = tk.Label(self.embedding_model_labelframe, text="Method *")
        self.lbp_method_combobox = ttk.Combobox(self.embedding_model_labelframe, textvariable=self.clustering_data['embedding_model']["lbp_method"], values=lbp_methods, state="readonly",  width=15)
        ToolTip(self.lbp_method_label, msg="LBP computation method.\nEach method affects how texture patterns are generated.")

        self.embedding_model_labelframe.grid(row=0, column=0, padx=5, pady=10, columnspan=4, sticky="ew")

        self.clustering_labelframe = ttk.LabelFrame(self.clustering_frame, text="Clustering configuration", padding=(20, 10))
        self.clustering_labelframe.grid(row=1, column=0, padx=5, pady=10, columnspan=4, sticky="ew")

        self.clustering_models_text = tk.Text(self.clustering_labelframe, height=5, width=15, state="disabled")


        self.clustering_models_label = tk.Label(self.clustering_labelframe, text="Clustering models *")
        self.clustering_models_label.grid(row=0, column=0, padx=10, pady=5)
        ToolTip(self.clustering_models_label, msg="Clustering models list to use with the embeddings.")

        self.clustering_metric_label = tk.Label(self.clustering_labelframe, text="Clustering metric *")
        self.clustering_metric_label.grid(row=0, column=2, padx=10, pady=5)

        ToolTip(self.clustering_metric_label, msg="Metric that defines how similarity between image clusters is measured.")
        
        self.clustering_models_text.grid(row=1, column=0, padx=10, pady=5, rowspan=2)
        
        self.add_model_bt = tk.Button(self.clustering_labelframe, text="Add model", command=lambda: self.open_model_form())
        self.add_model_bt.grid(row=1, column=1)
        ToolTip(self.add_model_bt, msg="Add a new clustering model.")
        ttk.Combobox(self.clustering_labelframe, textvariable=self.clustering_data["clustering_metric"], values=clustering_metrics, state="readonly", width=10).grid(row=1, column=2)        

        self.edit_model_bt = tk.Button(self.clustering_labelframe, text="Edit model", command=lambda: self.open_model_form(edit=True))
        self.edit_model_bt.grid(row=2, column=1)
        ToolTip(self.edit_model_bt, msg="Edit an existing clustering model.")

        self.edit_model_bt.grid_remove()

        self.visualization_labelframe = ttk.LabelFrame(self.clustering_frame, text="Visualization parameters", padding=(20, 10))
        self.visualization_labelframe.grid(row=2, column=0, padx=5, pady=10, columnspan=4, sticky="ew")

        self.visualization_label = tk.Label(self.visualization_labelframe, text="Visualization technique *")
        self.visualization_technique_combobox = ttk.Combobox(self.visualization_labelframe, textvariable=self.clustering_data["visualization_technique"], values=visualization_techniques, state="readonly", width=6)        
        ToolTip(self.visualization_label, msg="Dimensionality reduction technique applied to the embeddings for plotting in 2D.")

        self.visualization_label.grid(row=1, column=1)


        self.clustering_plot_label = tk.Label(self.visualization_labelframe, text="Plot")
        self.clustering_plot_label.grid(row=0, column=0, padx=10, pady=5)
        ToolTip(self.clustering_plot_label, msg="Enables visualization of clustering results by reducing the dimensionality of the embeddings.\nPlots images in a 2 dimensional space.")

        ttk.Checkbutton(self.visualization_labelframe, variable=self.clustering_data["plot"], command=lambda: FormUtils.toggle_label_entry(self.clustering_data["plot"], self.visualization_label, 
                                                                                                                          self.visualization_technique_combobox, None, 0, 1)).grid(row=1, column=0, padx=5, pady=5)
        self.visualization_technique_combobox.grid(row=0, column=1, padx=10, pady=5)

        

        self.framework_combobox.bind("<<ComboboxSelected>>", self.update_framework_fields)
        self.name_combobox.bind("<<ComboboxSelected>>", self.select_other_model)

        self.name_label.grid_forget()
        self.other_name_label.grid_forget()
        self.custom_model_info_label.grid_forget()
        self.other_name_entry.grid_forget()
        self.name_combobox.grid_forget()
        self.visualization_label.grid_forget()
        self.visualization_technique_combobox.grid_forget()

        self.framework_combobox.grid(row=1, column=0, padx=10, pady=5)
        self.name_label.grid(row=0, column=1, padx=10, pady=5)
        self.lbp_name_entry.grid(row=1, column=1, padx=10, pady=5)
        self.resize_height_label.grid(row=0, column=2, padx=10)
        self.resize_height_entry.grid(row=1, column=2, padx=10, pady=5)
        self.resize_width_label.grid(row=0, column=3, padx=10)
        self.resize_width_entry.grid(row=1, column=3, padx=10, pady=5)
        self.lbp_radius_label.grid(row=2, column=1, padx=10)
        self.lbp_radius_entry.grid(row=3, column=1, padx=10, pady=5)
        self.lbp_num_points_label.grid(row=2, column=2, padx=10)
        self.lbp_num_points_entry.grid(row=3, column=2, padx=10, pady=5)
        self.lbp_method_label.grid(row=2, column=3, padx=10)
        self.lbp_method_combobox.grid(row=3, column=3, padx=10, pady=5)
        self.batch_size_label.grid(row=2, column=0, padx=10, pady=5)
        self.batch_size_entry.grid(row=3, column=0, padx=10, pady=5)

    def select_other_model(self, event):

        selected_value = self.name_combobox.get()
        framework = self.framework_combobox.get()

        if selected_value.lower() == "other":
            self.other_name_entry.delete(0, tk.END)  
            f = font.Font(self.custom_model_info_label, self.custom_model_info_label.cget("font"))
            f.configure(underline=True)
            self.custom_model_info_label.configure(font=f)

            if framework == "huggingface":
                self.other_name_label.grid(row=0, column=3, pady=5, padx=5)
                self.other_name_entry.grid(row=1, column=3, pady=5,padx=5)

                self.custom_model_info_label.config(text = "HuggingFace models for image feature extraction", fg="blue", cursor="hand2", wraplength=200)
                self.custom_model_info_label.bind("<Button-1>", lambda e: webbrowser.open(ConfigHandler.FRAMEWORK_URLS["huggingface"]))
                self.custom_model_info_label.grid(row=2, column=3, pady=5, padx=5)

                return
            
            self.other_name_label.grid(row=0, column=5, pady=5, padx=5)
            self.other_name_entry.grid(row=1, column=5, pady=5,padx=5)

            if framework == "pytorch":
                self.custom_model_info_label.config(text = "Pytorch models for image feature extraction", fg="blue", cursor="hand2", wraplength=200)
                self.custom_model_info_label.bind("<Button-1>", lambda e: webbrowser.open(ConfigHandler.FRAMEWORK_URLS["pytorch"]))
                self.custom_model_info_label.grid(row=2, column=5, pady=5, padx=5)
                return

            self.custom_model_info_label.config(text = "TensorFlow models for image feature extraction", fg="blue", cursor="hand2", wraplength=200)
            self.custom_model_info_label.bind("<Button-1>", lambda e: webbrowser.open(ConfigHandler.FRAMEWORK_URLS["tensorflow"]))
            self.custom_model_info_label.grid(row=2, column=5, pady=5, padx=5)
            return
        
        self.other_name_entry.grid_forget()
        self.other_name_label.grid_forget()
        self.custom_model_info_label.grid_forget()

    def update_framework_fields(self, event):
        framework = self.clustering_data['embedding_model']["framework"].get()
        self.name_combobox.set('')
        hf_models = ConfigHandler.CONFIGURATION_VALUES['huggingface_models']
        py_models = ConfigHandler.CONFIGURATION_VALUES['pytorch_models']
        tf_models = ConfigHandler.CONFIGURATION_VALUES['tensorflow_models']
        model_list = []

        self.name_label.grid_remove()
        self.name_combobox.grid_remove()
        self.lbp_name_entry.grid_remove()
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
        self.other_name_entry.grid_remove()
        self.other_name_label.grid_remove()
        self.custom_model_info_label.grid_remove()

        if framework == "opencv":
            self.name_label.grid(row=0, column=1, padx=10, pady=5)
            self.lbp_name_entry.grid(row=1, column=1, padx=10, pady=5)
            self.resize_height_label.grid(row=0, column=2, padx=10)
            self.resize_height_entry.grid(row=1, column=2, padx=10, pady=5)
            self.resize_width_label.grid(row=0, column=3, padx=10)
            self.resize_width_entry.grid(row=1, column=3, padx=10, pady=5)
            self.lbp_radius_label.grid(row=2, column=1, padx=10)
            self.lbp_radius_entry.grid(row=3, column=1, padx=10, pady=5)
            self.lbp_num_points_label.grid(row=2, column=2, padx=10)
            self.lbp_num_points_entry.grid(row=3, column=2, padx=10, pady=5)
            self.lbp_method_label.grid(row=2, column=3, padx=10)
            self.lbp_method_combobox.grid(row=3, column=3, padx=10, pady=5)
            return

        elif framework == "huggingface":
            
            model_list = hf_models
            self.name_label.grid(row=0, column=1, padx=10, sticky="ew")
            self.name_combobox.grid(row=1, column=1, padx=10, pady=5, columnspan=2, sticky="ew")

        else:

            if framework == "pytorch":
                model_list = py_models
            elif framework == "tensorflow":
                model_list = tf_models 

            self.name_label.grid(row=0, column=1, padx=10, sticky="ew")
            self.name_combobox.grid(row=1, column=1, padx=10, pady=5, columnspan=2, sticky="ew")
            self.resize_height_label.grid(row=0, column=3, padx=10)
            self.resize_height_entry.grid(row=1, column=3, padx=10, pady=5)
            self.resize_width_label.grid(row=0, column=4, padx=10)
            self.resize_width_entry.grid(row=1, column=4, padx=10, pady=5)

        self.name_combobox["values"] = model_list

        max_length = max([len(model) for model in model_list])  
        self.name_combobox.config(width=max_length + 5)
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

        FormUtils.center_window(model_form, self.top)
        
        model_form.top.focus_set()

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

    def check_embedding_model(self, model_name, framework):
        if framework.lower() == 'huggingface':
            try:
                model_info = huggingface_hub.model_info(model_name)
                return True
            except Exception as e:
                return False
            
        if framework.lower() == 'pytorch':
            try:
                model = getattr(models, model_name)
                return True
            except AttributeError:
                return False
            
        if framework.lower() == 'tensorflow':
            try:
                model = getattr(tf.keras.applications, model_name)
                return True
            except AttributeError:
                return False
        

    def validate_form(self):

        embedding_model_info = self.clustering_data["embedding_model"]
        framework = embedding_model_info["framework"].get()

        if framework == "opencv":

            if embedding_model_info["resize_height"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", "Resize height must be specified when using an opencv feature extractor model.")
                return False
            
            if embedding_model_info["resize_width"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", "Resize width must be specified when using an opencv feature extractor model.")
                return False
            
            if embedding_model_info["lbp_radius"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", "LBP radius must be specified when using an opencv feature extractor model.")
                return False
            
            if embedding_model_info["lbp_num_points"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", "Num points for LBP must be specified when using an opencv feature extractor model.")
                return False
            
        elif framework == "huggingface":

            if embedding_model_info["name"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"The name of the {framework} embedding model to use must be specified.")
                return False

        elif framework == "pytorch":

            if embedding_model_info["name"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"The name of the {framework} embedding model to use must be specified.")
                return False
            
            if embedding_model_info["resize_height"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"Resize height must be specified when using a {framework} embedding model.")
                return False
            
            if embedding_model_info["resize_width"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"Resize width must be specified when using a {framework} embedding model.")
                return False
            
        elif framework == "tensorflow":

            if embedding_model_info["name"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"The name of the {framework} embedding model to use must be specified.")
                return False
            
            if embedding_model_info["resize_height"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"Resize height must be specified when using a {framework} embedding model.")
                return False
            
            if embedding_model_info["resize_width"].get().strip() == "":
                tk.messagebox.showerror("Clustering configuration error", f"Resize width must be specified when using a {framework} embedding model.")
                return False
            
        if embedding_model_info["name"].get().lower() == "other":

            custom_model_name = embedding_model_info["name_other"].get()
                
            if custom_model_name.strip() == "":
                tk.messagebox.showerror("Clustering configuration error", "Custom model name must be specified.")
                return False
            
            if not self.check_embedding_model(custom_model_name, framework):
                tk.messagebox.showerror("Clustering configuration error", f"Embedding model name {custom_model_name} does not exist for {framework} framework.")
                return False
         
            
        if not self.clustering_data["clustering_models"]:
            tk.messagebox.showerror("Clustering configuration error", "At least one clustering model must be defined.")
            return False
        
        if embedding_model_info["name"].get().lower() == "other":
            tk.messagebox.showwarning("Unverified model warning", f"The embedding model you selected from {framework} has not been verified with this application. There is a possibility of unexpected behavior or errors.\n\n"
                                      "If you experience issues, try selecting a model from the available list.")


        return True

    def next(self):
        if not self.validate_form():
            return
        
        self.config_data["clustering_data"].update(self.clustering_data)
        self.controller.show_frame("ReductionFrame")
        print(self.config_data["clustering_data"])
