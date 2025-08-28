import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip

from segdan.utils.constants import ClusteringModelName
from segdan.utils.confighandler import ConfigHandler

class ReductionConfigFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)
        self.top = parent
        self.config_data = config_data

        if "reduction_data" not in self.config_data:
            self.config_data["reduction_data"] = {
                "retention_percentage": self.config_data.get("retention_percentage", tk.StringVar(value="0.7")),
                "reduction_type": self.config_data.get("reduction_type", tk.StringVar(value="representative")),
                "diverse_percentage": self.config_data.get("diverse_percentage", tk.StringVar(value="0.2")),
                "include_outliers": self.config_data.get("include_outliers", tk.BooleanVar(value=False)),
                "reduction_model": self.config_data.get("reduction_model", tk.StringVar(value="")),
                "use_reduced": self.config_data.get("use_reduced", tk.BooleanVar(value=False))
            }

        self.reduction_data = self.config_data["reduction_data"]
        self.controller = controller
        self.row=0

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        label_title = ttk.Label(self, text="Dataset reduction", font=("Arial", 18, "bold"))
        label_title.grid(row=self.row, column=0, columnspan=5, pady=(20,10), padx=10)
            
        self.reduction_frame = tk.Frame(self, padx=10, pady=10)
        self.reduction_frame.grid(row=self.row+1, column=0, padx=10, pady=10)

        self.update_reduction_models()

        self.reduction_config_widgets()

        button_frame = ttk.Frame(self.reduction_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.reduction_frame.grid_rowconfigure(0, weight=0)
        self.reduction_frame.grid_columnconfigure(0, weight=0)
        self.reduction_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=lambda: controller.show_frame("ReductionFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

    def tkraise(self, aboveThis=None):
        self.update_reduction_models()
        self.reduction_model_combobox["values"] = self.reduction_models
        super().tkraise(aboveThis)

    def update_reduction_models(self):
        
        self.clustering_models = self.config_data.get("clustering_data", {}).get("clustering_models", {})

        self.reduction_models = list(self.clustering_models.keys())
        if len(self.reduction_models) > 1:
            self.reduction_models.append("best model")

        if self.reduction_data["reduction_model"].get() not in self.reduction_models:
            self.reduction_data["reduction_model"].set("")

    def reduction_config_widgets(self):

        reduction_types = ConfigHandler.CONFIGURATION_VALUES['reduction_type']
        
        vcmd = (self.top.register(self.validate_numeric), "%P")

        self.reduction_config_labelframe = ttk.LabelFrame(self.reduction_frame, text="Reduction configuration", padding=(20, 10))
        self.reduction_config_labelframe.grid(row=0, column=0, padx=5, pady=10, columnspan=2, sticky="ew")

        self.reduction_type_label = tk.Label(self.reduction_config_labelframe, text="Reduction type *")
        self.reduction_type_combobox = ttk.Combobox(self.reduction_config_labelframe, textvariable=self.reduction_data['reduction_type'], values=reduction_types, state="readonly", width=15)
        ToolTip(self.reduction_type_label, msg="Strategy for selecting the subset of images from each cluster.")

        self.retention_percentage_label = tk.Label(self.reduction_config_labelframe, text="Reduction percentage *")
        self.retention_percentage_entry = tk.Entry(self.reduction_config_labelframe, textvariable=self.reduction_data['retention_percentage'], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.retention_percentage_label, msg="Percentage of the dataset retained after reduction.")

        self.diverse_percentage_label = tk.Label(self.reduction_config_labelframe, text="Diverse percentage *")
        self.diverse_percentage_entry = tk.Entry(self.reduction_config_labelframe, textvariable=self.reduction_data['diverse_percentage'], width=10, validate="key", validatecommand=vcmd)
        ToolTip(self.diverse_percentage_label, msg="Specifies the percentage of diverse images to keep from each cluster, ensuring variety instead of only the most representative ones.")

        self.reduction_model_label = tk.Label(self.reduction_config_labelframe, text="Reduction model *")
        ToolTip(self.reduction_model_label, msg="Clustering model to use for reducing the dataset.")

        self.reduction_model_combobox = ttk.Combobox(self.reduction_config_labelframe, width=15, values=self.reduction_models, state="readonly")

        self.use_reduced_label = tk.Label(self.reduction_config_labelframe, text="Use reduced dataset *")
        self.use_reduced_checkbt = ttk.Checkbutton(self.reduction_config_labelframe, variable=self.reduction_data["use_reduced"])
        ToolTip(self.use_reduced_label, msg="Whether to use the reduced dataset in the training step.")

        self.include_outliers_label = tk.Label(self.reduction_config_labelframe, text="Include outliers *")
        self.include_outliers_checkbt = ttk.Checkbutton(self.reduction_config_labelframe, variable=self.reduction_data["include_outliers"])
        ToolTip(self.include_outliers_label, msg="Include outlier images.\nThese images are marked with the label -1.")

        self.reduction_type_label.grid(row=self.row, column=0, padx=10, pady=5)
        self.row+=1

        self.reduction_type_combobox.grid(row=self.row, column=0, padx=10, pady=5)

        self.row+=1
        self.retention_percentage_label.grid(row=self.row, column=0, padx=10, pady=5)
        self.diverse_percentage_label.grid(row=self.row, column=1, padx=10, pady=5)
            
        self.row+=1
        self.retention_percentage_entry.grid(row=self.row, column=0, padx=10, pady=5)
        self.diverse_percentage_entry.grid(row=self.row, column=1, padx=10, pady=5)

        self.row+=1
        self.reduction_model_label.grid(row=self.row, column=0, padx=10, pady=5)

        self.row+=1
        self.reduction_model_combobox.grid(row=self.row, column=0, padx=10, pady=5)

        self.row+=1
        self.use_reduced_label.grid(row=self.row, column=0, padx=10, pady=5)
        
        self.row+=1
        self.use_reduced_checkbt.grid(row=self.row, column=0, padx=10, pady=5)

        self.reduction_type_combobox.bind("<<ComboboxSelected>>", self.grid_diverse_percentage)
        self.reduction_model_combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)

    def grid_diverse_percentage(self, event):

        reduction_type = self.reduction_data["reduction_type"].get()
        
        if reduction_type == "representative":
            self.diverse_percentage_label.grid(row=2, column=1, padx=10, pady=5)
            self.diverse_percentage_entry.grid(row=3, column=1, padx=10, pady=5)
            return
        
        self.diverse_percentage_label.grid_remove()
        self.diverse_percentage_entry.grid_remove()

    def on_combobox_select(self, event):

        selected_model = self.reduction_model_combobox.get()

        if selected_model == "best model":
            self.reduction_data["reduction_model"] = tk.StringVar(value="best_model")
        else:
            self.reduction_data["reduction_model"] = tk.StringVar(value=selected_model)

        self.show_include_outliers(event)        

    def show_include_outliers(self, event):

        reduction_model = self.reduction_data["reduction_model"].get()

        if reduction_model == ClusteringModelName.DBSCAN.value or reduction_model == ClusteringModelName.OPTICS.value or (reduction_model == "best_model" and ("dbscan" in self.reduction_models or "optics" in self.reduction_models)):
            self.include_outliers_label.grid(row=6, column=1, padx=10, pady=5)
            self.include_outliers_checkbt.grid(row=7, column=1, padx=10, pady=5)
            return

        self.reduction_data["include_outliers"].set(False)
        self.include_outliers_label.grid_remove()
        self.include_outliers_checkbt.grid_remove()

    def validate_numeric(self, value):
        if value == "" or value.isdigit():
            return True
        if value.count('.') == 1 and value.replace('.', '').isdigit():
            return True
        return False 
    
    def validate_form(self):

        reduc_percentage = self.reduction_data["retention_percentage"].get()


        if reduc_percentage.strip() == "":
            tk.messagebox.showerror("Reduction configuration error", "Reduction percentage must be specified.")
            return False
        
        reduc_value = float(reduc_percentage)
        if reduc_value >= 1.0:
            tk.messagebox.showerror("Reduction configuration error", "Reduction percentage can't be equal or greater than 1.")
            return False

        if reduc_value == 0.0:
            tk.messagebox.showerror("Reduction configuration error", "Reduction percentage can't be 0.")
            return False
        
        if self.reduction_data["reduction_type"].get() == "representative":
            
            diverse_percentage = self.reduction_data["diverse_percentage"].get()
        
            if diverse_percentage.strip() == "":
                tk.messagebox.showerror("Reduction configuration error", "Diverse percentage must be specified.")
                return False
            
            diverse_value = float(diverse_percentage)

            if diverse_value >= 1.0:
                tk.messagebox.showerror("Reduction configuration error", "Diverse percentage can't be equal or greater than 1.")
                return False
            
            if reduc_value < diverse_value:
                tk.messagebox.showerror("Reduction configuration error", "Diverse percentage can't be greater than the reduction percentage.")
                return False

            
        if self.reduction_data["reduction_model"].get().strip() == "":
            tk.messagebox.showerror("Reduction configuration error", "A reduction model must be specified.")
            return False
        
        return True
        
    
    def next(self):

        if not self.validate_form():
            return 

        self.config_data["reduction_data"].update(self.reduction_data)
        self.controller.show_frame("DatasetSplitFrame")
