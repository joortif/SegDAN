import tkinter as tk
from tkinter import ttk

from src.forms.formutils import FormUtils
from src.utils.confighandler import ConfigHandler

class TrainingConfigForm():

    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
       
        self.top.title("Training configuration")

        self.top.geometry("1100x300")

        self.top.resizable(False,False)

        self.models_list = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]

        self.train_images = {"cluster_images": True}
        self.training_data = {
            "split_percentages": {
                "train": tk.DoubleVar(value=0.7),
                "valid": tk.DoubleVar(value=0.2),
                "test": tk.DoubleVar(value=0.1)
            },
            "stratification": tk.BooleanVar(value=False),
            "stratification_type": tk.StringVar(value="pixels"),
            "cross_validation": tk.BooleanVar(value=False),  
            "num_folds": tk.IntVar(value=5),

            "segmentation": tk.StringVar(value="semantic"),
            "models": tk.StringVar(value="unet"),
            "batch_size": tk.IntVar(value=8),
            "segmentation_metric": tk.StringVar(value="dice_score"),
            "epochs": tk.IntVar(value=100)
            }

        self.create_widgets()

    def create_widgets(self):
        self.train_config_widgets()

        btn_frame = tk.Frame(self.top)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Cancel", command=self.clear_and_close_form).pack(side="right", padx=5)
        tk.Button(btn_frame, text="Save configuration", command=lambda: FormUtils.save_config(self.training_data)).pack(side="right", padx=5)

    def train_config_widgets(self):
        stratification_types = ConfigHandler.CONFIGURATION_VALUES["stratification_types"]
        segmentation_types = ConfigHandler.CONFIGURATION_VALUES["segmentation"]
        
        segmentation_metrics = ConfigHandler.CONFIGURATION_VALUES["segmentation_metric"]

        self.training_frame = tk.LabelFrame(self.top, text="Training configuration", padx=10, pady=10)
        self.training_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=8)

        tk.Label(self.training_frame, text="Train percentage").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.training_data['split_percentages']['train'], width=10).grid(row=1, column=0, padx=5, sticky="w")
        tk.Label(self.training_frame, text="Validation percentage").grid(row=0, column=1, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.training_data['split_percentages']['valid'], width=10).grid(row=1, column=1, padx=5, sticky="w")
        tk.Label(self.training_frame, text="Test percentage").grid(row=0, column=2, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.training_data['split_percentages']['test'], width=10).grid(row=1, column=2, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Epochs").grid(row=0, column=3, padx=10, columnspan=3, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.training_data['epochs'], width=10).grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Stratification").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.training_frame, variable=self.training_data["stratification"], command=lambda: FormUtils.toggle_label_entry(self.training_data["stratification"], self.stratification_type_label, 
                                                                                                                          self.stratification_type_combobox, self.stratification_type_comment, 0, 6)).grid(row=1, column=4, padx=5, pady=5)

        tk.Label(self.training_frame, text="Cross validation").grid(row=0, column=5, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.training_frame, variable=self.training_data["cross_validation"], command=lambda: FormUtils.toggle_label_entry(self.training_data["cross_validation"], self.num_folds_label, 
                                                                                                                          self.num_folds_entry, self.num_folds_comment, 0, 7)).grid(row=1, column=5, padx=5, pady=5)
        
        self.stratification_type_label = tk.Label(self.training_frame, text="Stratification type")
        self.stratification_type_label.grid(row=0, column=6, sticky="w")
        self.stratification_type_combobox = ttk.Combobox(self.training_frame, textvariable=self.training_data["stratification_type"], values=stratification_types, state="readonly", )
        self.stratification_type_combobox.grid(row=1, column=6, padx=5, pady=5, sticky="w")       
        self.stratification_type_comment = tk.Label(self.training_frame, text="Strategy used to group the data", fg="gray", font=("Arial", 8), wraplength=200)
        self.stratification_type_comment.grid(row=2, column=6, padx=5, pady=5, sticky="w")

        self.num_folds_label = tk.Label(self.training_frame, text="Number of folds")
        self.num_folds_label.grid(row=0, column=7, sticky="w")
        self.num_folds_entry = ttk.Entry(self.training_frame, textvariable=self.training_data["num_folds"], width=10)
        self.num_folds_entry.grid(row=1, column=7, padx=5, pady=5, sticky="w")       
        self.num_folds_comment = tk.Label(self.training_frame, text="Number of groups the data is divided into.", fg="gray", font=("Arial", 8), wraplength=200)
        self.num_folds_comment.grid(row=2, column=7, padx=5, pady=5, sticky="w")

        tk.Label(self.training_frame, text="Batch size").grid(row=8, column=0, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.training_data['batch_size'], width=10).grid(row=9, column=0, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Segmentation type").grid(row=8, column=1, sticky="w")
        self.segmentation_type_combobox = ttk.Combobox(self.training_frame, textvariable=self.training_data["segmentation"], values=segmentation_types, state="readonly", width=10)
        self.segmentation_type_combobox.grid(row=9, column=1, sticky="w")
        
        tk.Label(self.training_frame, text="Segmentation metric").grid(row=8, column=2, sticky="w")
        self.segmentation_metric_combobox = ttk.Combobox(self.training_frame, textvariable=self.training_data["segmentation_metric"], values=segmentation_metrics, state="readonly", width=10)
        self.segmentation_metric_combobox.grid(row=9, column=2, sticky="w")

        self.segmentation_type_combobox.bind("<<ComboboxSelected>>", self.update_models_list)

        tk.Label(self.training_frame, text="Segmentation models").grid(row=8, column=3, sticky="w")
        self.listbox = tk.Listbox(self.training_frame, selectmode="multiple", height=len(self.models_list))

        self.update_listbox()

        self.listbox.bind('<<ListboxSelect>>', lambda event, listbox=self.listbox, options=self.models_list: 
                    self.update_models_config(event, listbox, options))

        self.stratification_type_label.grid_forget()
        self.stratification_type_combobox.grid_forget()
        self.stratification_type_comment.grid_forget()

        self.num_folds_label.grid_forget()
        self.num_folds_entry.grid_forget()
        self.num_folds_comment.grid_forget()

    def update_models_list(self, event):
        segmentation_type = self.training_data["segmentation"].get()
        semantic_models = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]
        instance_models = ConfigHandler.CONFIGURATION_VALUES["instance_segmentation_models"]

        if segmentation_type == "semantic":
            self.models_list = semantic_models
        elif segmentation_type == "instance":
            self.models_list = instance_models

        self.update_listbox()

    def update_listbox(self):
        self.listbox.delete(0, tk.END)  
        for option in self.models_list:
            self.listbox.insert(tk.END, option)
        self.listbox.grid(row=9, column=3, columnspan=2, padx=5, pady=5)

    def update_models_config(self, event, listbox, options):
        selected_indices = listbox.curselection()  
        selected_values = [options[i] for i in selected_indices]  
        self.training_data["models"] = selected_values

    def clear_and_close_form(self):

        self.training_data = {}
        self.train_images = {"cluster_images": False}

        self.top.destroy()
