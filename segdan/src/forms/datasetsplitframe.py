import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip
import numpy as np
import json
import yaml

from src.utils.confighandler import ConfigHandler
from src.forms.formutils import FormUtils

class DatasetSplitFrame(ttk.Frame):
        
    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)
        self.top = parent
        self.config_data = config_data
        self.controller = controller
        self.final_dict = final_dict

        self.models_list = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]

        if "split_data" not in self.config_data:
            self.config_data["split_data"] = {
                "split_method": self.config_data.get("split_method", tk.BooleanVar(value=True)), 
                "hold_out": {
                    "train": self.config_data.get("train", tk.StringVar(value="0.7")),
                    "valid": self.config_data.get("valid", tk.StringVar(value="0.2")),
                    "test": self.config_data.get("test", tk.StringVar(value="0.1"))
                },
                "cross_val": {
                    "num_folds": self.config_data.get("num_folds", tk.StringVar(value="5")),
                },
                "stratification": self.config_data.get("stratification", tk.BooleanVar(value=False)),
                "stratification_type": self.config_data.get("stratification_type", tk.StringVar(value="pixels")),
                "stratification_random_seed": self.config_data.get("stratification_random_seed", tk.StringVar(value="123")),
                }
            
        self.split_data = self.config_data["split_data"]
        self.split_method = self.split_data["split_method"]

        self.row=0

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        label_title = ttk.Label(self, text="Dataset split", font=("Arial", 18, "bold"))
        label_title.grid(row=self.row, column=0, columnspan=5, pady=(20,10), padx=10)
            
        self.training_frame = tk.Frame(self, padx=10, pady=10)
        self.training_frame.grid(row=self.row+1, column=0, padx=10, pady=10)

        self.training_config_widgets()

        button_frame = ttk.Frame(self.training_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.training_frame.grid_rowconfigure(0, weight=0)
        self.training_frame.grid_columnconfigure(0, weight=0)
        self.training_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=self.back)
        button_back.grid(row=0, column=0, padx=50, pady=10, sticky="w")

        button_save = ttk.Button(button_frame, text="Next", command=self.next)
        button_save.grid(row=0, column=1, pady=10, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

    def training_config_widgets(self):
        stratification_types = ConfigHandler.CONFIGURATION_VALUES["stratification_types"]

        val_float = (self.top.register(self.validate_float), "%P")
        val_int = (self.top.register(self.validate_int), "%P")

        self.dataset_config_labelframe = ttk.LabelFrame(self.training_frame, text="Dataset configuration", padding=(20,10))

        self.hold_out_radiobt_yes = tk.Radiobutton(self.dataset_config_labelframe, text="Hold out", variable=self.split_method, value=True)
        self.hold_out_radiobt_no = tk.Radiobutton(self.dataset_config_labelframe, text="Cross validation", variable=self.split_method, value=False)
        ToolTip(self.hold_out_radiobt_yes, msg="Splits the dataset into training, validation, and test sets in a single step.")
        ToolTip(self.hold_out_radiobt_no, msg="Repeatedly splits the dataset into training and validation sets to improve reliability; may significantly increase training time.")

        self.train_percentage_label = tk.Label(self.dataset_config_labelframe, text="Train percentage *")
        self.train_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.split_data['hold_out']['train'], width=10, validate="key", validatecommand=val_float)
        self.valid_percentage_label = tk.Label(self.dataset_config_labelframe, text="Validation percentage *")
        self.valid_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.split_data['hold_out']['valid'], width=10, validate="key", validatecommand=val_float)
        self.test_percentage_label = tk.Label(self.dataset_config_labelframe, text="Test percentage *")
        self.test_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.split_data['hold_out']['test'], width=10, validate="key", validatecommand=val_float)
        ToolTip(self.train_percentage_label, msg="Percentage of data allocated for training.")
        ToolTip(self.valid_percentage_label, msg="Percentage of data used for model validation.")
        ToolTip(self.test_percentage_label, msg="Percentage of data reserved for testing the model's performance.")

        self.stratification_label = tk.Label(self.dataset_config_labelframe, text="Stratification")
        self.stratification_checkbt = ttk.Checkbutton(self.dataset_config_labelframe, variable=self.split_data["stratification"], command=self.on_stratification_select)   
        ToolTip(self.stratification_label, msg="Whether to maintain class distribution when splitting the dataset.")

        self.stratification_type_label = tk.Label(self.dataset_config_labelframe, text="Stratification type *")
        self.stratification_type_combobox = ttk.Combobox(self.dataset_config_labelframe, textvariable=self.split_data["stratification_type"], values=stratification_types, state="readonly")
        ToolTip(self.stratification_type_label, msg="Method used for stratifying the dataset (e.g., by pixel distribution, number of objects from each class...)")

        self.stratification_random_seed_label = tk.Label(self.dataset_config_labelframe, text="Random seed")
        self.stratification_random_seed_entry = ttk.Entry(self.dataset_config_labelframe, textvariable=self.split_data["stratification_random_seed"], validate="key", width=10, validatecommand=val_int)
        ToolTip(self.stratification_random_seed_label, msg="Specify the random seed for stratification. A random seed ensures reproducibility by initializing the random number generator to the same state.")

        self.num_folds_label = tk.Label(self.dataset_config_labelframe, text="Number of folds *")
        self.num_folds_entry = ttk.Entry(self.dataset_config_labelframe, textvariable=self.split_data["cross_val"]["num_folds"], width=10, validate="key", validatecommand=val_int)
        ToolTip(self.num_folds_label, msg="Number of groups into which the dataset is split for cross-validation.")

        self.dataset_config_labelframe.grid(row=0, column=0, padx=5, pady=10, columnspan=4, sticky="ew")
        
        self.hold_out_radiobt_yes.bind("<ButtonRelease-1>", self.update_training_fields)
        self.hold_out_radiobt_no.bind("<ButtonRelease-1>", self.update_training_fields)

        self.hold_out_radiobt_yes.grid(row=self.row, column=0, padx=10, pady=10)
        self.hold_out_radiobt_no.grid(row=self.row, column=1, padx=10, pady=10)

        self.row+=1
        self.train_percentage_label.grid(row=self.row, column=0, padx=10, pady=10)
        self.valid_percentage_label.grid(row=self.row, column=1, padx=10, pady=10)
        self.test_percentage_label.grid(row=self.row, column=2, padx=10, pady=10)
        
        self.row+=1

        self.train_percentage_entry.grid(row=self.row, column=0, padx=10, pady=10)
        self.valid_percentage_entry.grid(row=self.row, column=1, padx=10, pady=10)
        self.test_percentage_entry.grid(row=self.row, column=2, padx=10, pady=10)

        self.row += 1

        self.stratification_label.grid(row=self.row, column=0, padx=10, pady=10)

        self.row +=1

        self.stratification_checkbt.grid(row=self.row, column=0, padx=10, pady=10)

    def on_stratification_select(self):

        if self.split_data["stratification"].get():

            self.stratification_type_label.grid(row=3, column=1, padx=10, pady=10)
            self.stratification_type_combobox.grid(row=4, column=1, padx=10, pady=10)

            self.stratification_random_seed_label.grid(row=3, column=2, padx=10, pady=10)
            self.stratification_random_seed_entry.grid(row=4, column=2, padx=10, pady=10)
            return
        
        self.stratification_type_label.grid_forget()
        self.stratification_type_combobox.grid_forget()
        self.stratification_random_seed_label.grid_forget()
        self.stratification_random_seed_entry.grid_forget()

    def update_training_fields(self, event=None):
        self.train_percentage_label.grid_forget()
        self.train_percentage_entry.grid_forget()
        self.valid_percentage_label.grid_forget()
        self.valid_percentage_entry.grid_forget()
        self.test_percentage_label.grid_forget()
        self.test_percentage_entry.grid_forget()
        self.num_folds_label.grid_forget()
        self.num_folds_entry.grid_forget()
        
        if not self.split_method.get():  
            self.train_percentage_label.grid(row=1, column=0, padx=10, pady=10)
            self.valid_percentage_label.grid(row=1, column=1, padx=10, pady=10)
            self.test_percentage_label.grid(row=1, column=2, padx=10, pady=10)
            
            self.train_percentage_entry.grid(row=2, column=0, padx=10, pady=10)
            self.valid_percentage_entry.grid(row=2, column=1, padx=10, pady=10)
            self.test_percentage_entry.grid(row=2, column=2, padx=10, pady=10)
            return

        self.num_folds_label.grid(row=1, column=0, padx=10, pady=10)
        self.num_folds_entry.grid(row=2, column=0, padx=10, pady=10)

    def validate_float(self, value):
        if value == "" or value.isdigit():
            return True
        if value.count('.') == 1 and value.replace('.', '').isdigit():
            return True
        return False
    
    def validate_int(self, value):
        return value.isdigit() or value == "" 


    def validate_form(self):

        if self.split_method.get(): 
            hold_out = self.config_data["split_data"].get("hold_out")
            
            train_percentage = hold_out.get("train")
            val_percentage = hold_out.get("valid")
            test_percentage = hold_out.get("test")

            if train_percentage.get().strip() == "":
                tk.messagebox.showerror("Dataset split error", "The training percentage for hold out can't be empty.")
                return False
            
            if val_percentage.get().strip() == "":
                tk.messagebox.showerror("Dataset split error", "The validation percentage for hold out can't be empty.")
                return False
            
            if test_percentage.get().strip() == "":
                tk.messagebox.showerror("Dataset split error", "The test percentage for hold out can't be empty.")
                return False
            
            train_value = float(train_percentage.get())
            val_value = float(val_percentage.get())
            test_value = float(test_percentage.get())

            if train_value == 0.0:
                tk.messagebox.showerror("Dataset split error", "Hold out train percentage can't be 0.")
                return False
            
            if test_value == 0.0:
                tk.messagebox.showerror("Dataset split error", "Hold out test percentage can't be 0.")
                return False

            if train_value > 1:
                tk.messagebox.showerror("Dataset split error", "Hold out train percentage can't be greater than 1.")
                return False
            
            if val_value > 1:
                tk.messagebox.showerror("Dataset split error", "Hold out validation percentage can't be greater than 1.")
                return False
            
            if test_value > 1:
                tk.messagebox.showerror("Dataset split error", "Hold out test percentage can't be greater than 1.")
                return False
        
            if not np.isclose(train_value + val_value + test_value,  1.0):
                tk.messagebox.showerror("Dataset split error", "The sum of train, validation, and test percentages must equal 1.")
                return False
        else:
            num_folds = self.config_data["split_data"].get("cross_val").get("num_folds").get()

            if num_folds.strip() == "":
                tk.messagebox.showerror("Dataset split error", "The number of folds for cross validation can't be empty.")
                return False
            
            num_fols_value = int(num_folds)

            if num_fols_value == 0:
                tk.messagebox.showerror("Dataset split error", "The number of folds for cross validation can't be 0.")
                return False
            
        if self.split_data["stratification"].get():

            random_seed = self.split_data["stratification_random_seed"].get().strip()

            if not random_seed:
                self.split_data["stratification_random_seed"].set("123")
        
        return True
    
    def update_config(self, event, listbox, options, param_key):
        selected_indices = listbox.curselection()  
        selected_values = [options[i] for i in selected_indices]  
        self.split_data[param_key] = selected_values

    def back(self):

            if not self.config_data["cluster_images"].get():
                self.controller.show_frame("ClusteringFrame")
                return

            if self.config_data["reduce_images"].get():
                self.controller.show_frame("ReductionConfigFrame")
            else:
                self.controller.show_frame("ReductionFrame")

    def next(self):
        if not self.validate_form():
            return
        
        self.config_data["split_data"].update(self.split_data)
        self.controller.show_frame("ModelConfigFrame")


