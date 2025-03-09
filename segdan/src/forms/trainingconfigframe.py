import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip
import numpy as np
import json
import yaml

from src.utils.confighandler import ConfigHandler
from src.forms.formutils import FormUtils

class TrainingConfigFrame(ttk.Frame):
        
    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)
        self.top = parent
        self.config_data = config_data
        self.controller = controller
        self.final_dict = final_dict

        self.models_list = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]


        if "training_data" not in self.config_data:
            self.config_data["training_data"] = {
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

                "segmentation": self.config_data.get("segmentation", tk.StringVar(value="semantic")),
                "models": self.config_data.get("models", []),
                "batch_size": self.config_data.get("batch_size", tk.StringVar(value="8")),
                "evaluation_metrics": self.config_data.get("evaluation_metrics", []),
                "selection_metric": self.config_data.get("selection_metric", tk.StringVar(value="dice_score")),
                "epochs": self.config_data.get("epochs", tk.StringVar(value="100"))
                }
            
        
        self.training_data = self.config_data["training_data"]
        self.split_method = self.training_data["split_method"]

        self.row=0

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        label_title = ttk.Label(self, text="Training configuration", font=("Arial", 18, "bold"))
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

        button_save = ttk.Button(button_frame, text="Save configuration", command=self.save)
        button_save.grid(row=0, column=1, pady=10, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

    def training_config_widgets(self):
        stratification_types = ConfigHandler.CONFIGURATION_VALUES["stratification_types"]
        segmentation_types = ConfigHandler.CONFIGURATION_VALUES["segmentation"]
        metrics = ConfigHandler.CONFIGURATION_VALUES["segmentation_metric"]

        val_float = (self.top.register(self.validate_float), "%P")
        val_int = (self.top.register(self.validate_int), "%P")

        self.dataset_config_labelframe = ttk.LabelFrame(self.training_frame, text="Dataset configuration", padding=(20,10))

        self.hold_out_radiobt_yes = tk.Radiobutton(self.dataset_config_labelframe, text="Hold out", variable=self.split_method, value=True)
        self.hold_out_radiobt_no = tk.Radiobutton(self.dataset_config_labelframe, text="Cross validation", variable=self.split_method, value=False)
        ToolTip(self.hold_out_radiobt_yes, msg="Splits the dataset into training, validation, and test sets in a single step.")
        ToolTip(self.hold_out_radiobt_no, msg="Repeatedly splits the dataset into training and validation sets to improve reliability; may significantly increase training time.")

        self.train_percentage_label = tk.Label(self.dataset_config_labelframe, text="Train percentage *")
        self.train_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.training_data['hold_out']['train'], width=10, validate="key", validatecommand=val_float)
        self.valid_percentage_label = tk.Label(self.dataset_config_labelframe, text="Validation percentage *")
        self.valid_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.training_data['hold_out']['valid'], width=10, validate="key", validatecommand=val_float)
        self.test_percentage_label = tk.Label(self.dataset_config_labelframe, text="Test percentage *")
        self.test_percentage_entry = tk.Entry(self.dataset_config_labelframe, textvariable=self.training_data['hold_out']['test'], width=10, validate="key", validatecommand=val_float)
        ToolTip(self.train_percentage_label, msg="Percentage of data allocated for training.")
        ToolTip(self.valid_percentage_label, msg="Percentage of data used for model validation.")
        ToolTip(self.test_percentage_label, msg="Percentage of data reserved for testing the model's performance.")

        self.stratification_label = tk.Label(self.dataset_config_labelframe, text="Stratification")
        self.stratification_checkbt = ttk.Checkbutton(self.dataset_config_labelframe, variable=self.training_data["stratification"], command=lambda: FormUtils.toggle_label_entry(self.training_data["stratification"], self.stratification_type_label, 
                                                                                                                          self.stratification_type_combobox, None, 3, 1))   
        ToolTip(self.stratification_label, msg="Whether to maintain class distribution when splitting the dataset.")

        self.stratification_type_label = tk.Label(self.dataset_config_labelframe, text="Stratification type")
        self.stratification_type_combobox = ttk.Combobox(self.dataset_config_labelframe, textvariable=self.training_data["stratification_type"], values=stratification_types, state="readonly")
        ToolTip(self.stratification_type_label, msg="Method used for stratifying the dataset (e.g., by pixel distribution, number of objects from each class...)")

        self.num_folds_label = tk.Label(self.dataset_config_labelframe, text="Number of folds *")
        self.num_folds_entry = ttk.Entry(self.dataset_config_labelframe, textvariable=self.training_data["cross_val"]["num_folds"], width=10, validate="key", validatecommand=val_int)
        ToolTip(self.num_folds_label, msg="Number of groups into which the dataset is split for cross-validation.")

        self.model_config_labelframe = ttk.LabelFrame(self.training_frame, text="Segmentation model configuration")

        self.segmentation_type_label = tk.Label(self.model_config_labelframe, text="Segmentation type")
        self.segmentation_type_combobox = ttk.Combobox(self.model_config_labelframe, textvariable=self.training_data["segmentation"], values=segmentation_types, state="readonly", width=10)
        ToolTip(self.segmentation_type_label, msg="Method used for segmenting images in the dataset.")

        self.evaluation_metrics_label = tk.Label(self.model_config_labelframe, text="Evaluation metrics *")
        self.evaluation_metrics_listbox = tk.Listbox(self.model_config_labelframe, selectmode="multiple", height=len(metrics), exportselection=0)
        ToolTip(self.evaluation_metrics_label, msg="Metrics used to assess model performance.")

        self.segmentation_selection_metric_label = tk.Label(self.model_config_labelframe, text="Segmentation model selection metric", wraplength=150) 
        self.segmentation_selection_metric_combobox = ttk.Combobox(self.model_config_labelframe, textvariable=self.training_data["selection_metric"], values=metrics, state="readonly", width=10)
        ToolTip(self.segmentation_selection_metric_label, msg="Metric used to choose the best segmentation model.")

        self.epochs_label = tk.Label(self.model_config_labelframe, text="Epochs *")
        self.epochs_entry = tk.Entry(self.model_config_labelframe, textvariable=self.training_data['epochs'], width=10, validate="key", validatecommand=val_int)
        ToolTip(self.epochs_label, msg="Number of times the entire dataset is passed through the model during training.")

        self.batch_size_label = tk.Label(self.model_config_labelframe, text="Batch size *")
        self.batch_size_entry = tk.Entry(self.model_config_labelframe, textvariable=self.training_data['batch_size'], width=10, validate="key", validatecommand=val_int)
        ToolTip(self.batch_size_label, msg="Number of samples processed together before updating the model.\nTypically a multiple of 2 (e.g., 8, 16, 32, 64).")

        self.segmentation_models_label = tk.Label(self.model_config_labelframe, text="Segmentation models *")
        self.segmentation_models_listbox = tk.Listbox(self.model_config_labelframe, selectmode="multiple", height=len(self.models_list), exportselection=0)
        ToolTip(self.segmentation_models_label, msg="List of segmentation models available for training and evaluation.")

        self.dataset_config_labelframe.grid(row=0, column=0, padx=5, pady=10, columnspan=4, sticky="ew")
        self.model_config_labelframe.grid(row=1, column=0, padx=5, pady=10, columnspan=3, sticky="ew")

        self.update_listbox(self.segmentation_models_listbox, self.models_list)
        self.update_listbox(self.evaluation_metrics_listbox, metrics)

        self.segmentation_type_combobox.bind("<<ComboboxSelected>>", self.update_models_list)

        self.evaluation_metrics_listbox.bind('<<ListboxSelect>>', lambda event, listbox=self.evaluation_metrics_listbox, options=metrics: 
                    self.update_evaluation_metrics(event, listbox, options, "evaluation_metrics"))

        self.segmentation_models_listbox.bind('<<ListboxSelect>>', lambda event, listbox=self.segmentation_models_listbox, options=self.models_list: 
                    self.update_segmentation_models(event, listbox, options, "models"))
        
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

        self.row +=1
        self.segmentation_type_label.grid(row=self.row, column=0, padx=10, pady=10)
        self.evaluation_metrics_label.grid(row=self.row, column=1, padx=10, pady=10)
        self.segmentation_selection_metric_label.grid(row=self.row, column=2, padx=10, pady=10)
        self.segmentation_models_label.grid(row=self.row, column=3, padx=5, pady=10)
        
        self.row +=1
        self.segmentation_type_combobox.grid(row=self.row, column=0, padx=10, pady=10)
        self.evaluation_metrics_listbox.grid(row=self.row, column=1, padx=10, pady=10)
        self.segmentation_selection_metric_combobox.grid(row=self.row, column=2, padx=10, pady=10)
        self.segmentation_models_listbox.grid(row=self.row, column=3, padx=5, pady=10)

        self.row +=1

        self.batch_size_label.grid(row=self.row, column=0, padx=10, pady=10)
        self.epochs_label.grid(row=self.row, column=1, padx=10, pady=10)

        self.row +=1

        self.batch_size_entry.grid(row=self.row, column=0, padx=10, pady=10)
        self.epochs_entry.grid(row=self.row, column=1, padx=10, pady=10)

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
    
    def update_listbox(self, listbox, option_list):
        listbox.delete(0, tk.END)  
        for option in option_list:
            listbox.insert(tk.END, option)
    
    def update_models_list(self, event):
        segmentation_type = self.training_data["segmentation"].get()
        semantic_models = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]
        instance_models = ConfigHandler.CONFIGURATION_VALUES["instance_segmentation_models"]

        if segmentation_type == "semantic":
            self.models_list = semantic_models
        elif segmentation_type == "instance":
            self.models_list = instance_models

        self.update_listbox(self.segmentation_models_listbox, self.models_list)

    def validate_form(self):

        if self.split_method.get(): 
            hold_out = self.config_data["training_data"].get("hold_out")
            
            train_percentage = hold_out.get("train")
            val_percentage = hold_out.get("valid")
            test_percentage = hold_out.get("test")

            if train_percentage.get().strip() == "":
                tk.messagebox.showerror("Training configuration error", "The training percentage for hold out can't be empty.")
                return False
            
            if val_percentage.get().strip() == "":
                tk.messagebox.showerror("Training configuration error", "The validation percentage for hold out can't be empty.")
                return False
            
            if test_percentage.get().strip() == "":
                tk.messagebox.showerror("Training configuration error", "The test percentage for hold out can't be empty.")
                return False
            
            train_value = float(train_percentage.get())
            val_value = float(val_percentage.get())
            test_value = float(test_percentage.get())

            if train_value == 0.0:
                tk.messagebox.showerror("Training configuration error", "Hold out train percentage can't be 0.")
                return False
            
            if test_value == 0.0:
                tk.messagebox.showerror("Training configuration error", "Hold out test percentage can't be 0.")
                return False

            if train_value > 1:
                tk.messagebox.showerror("Training configuration error", "Hold out train percentage can't be greater than 1.")
                return False
            
            if val_value > 1:
                tk.messagebox.showerror("Training configuration error", "Hold out validation percentage can't be greater than 1.")
                return False
            
            if test_value > 1:
                tk.messagebox.showerror("Training configuration error", "Hold out test percentage can't be greater than 1.")
                return False
        
            if not np.isclose(train_value + val_value + test_value,  1.0):
                print((train_value + val_value + test_value))
                tk.messagebox.showerror("Training configuration error", "The sum of train, validation, and test percentages must equal 1.")
                return False
        else:
            num_folds = self.config_data["training_data"].get("cross_val").get("num_folds").get()

            if num_folds.strip() == "":
                tk.messagebox.showerror("Training configuration error", "The number of folds for cross validation can't be empty.")
                return False
            
            num_fols_value = int(num_folds)

            if num_fols_value == 0:
                tk.messagebox.showerror("Training configuration error", "The number of folds for cross validation can't be 0.")
                return False

        if len(self.training_data["evaluation_metrics"]) == 0:
            tk.messagebox.showerror("Training configuration error", "You must select at least one evaluation metric.")
            return False
        
        if len(self.training_data["models"]) == 0:
            tk.messagebox.showerror("Training configuration error", "You must select at least one model for training.")
            return False

        if self.training_data["batch_size"].get().strip() == "":
            tk.messagebox.showerror("Training configuration error", "The batch size of the models can't be empty.")
            return False
        
        batch_size_value = int(self.training_data["batch_size"].get())
        if batch_size_value == 0:
            tk.messagebox.showerror("Training configuration error", "The batch size of the models can't be 0.")
            return False
        
       
        if self.training_data["epochs"].get().strip() == "":
            tk.messagebox.showerror("Training configuration error", "The number of epochs for training the models can't be empty.")
            return False
        
        epochs_value = int(self.training_data["epochs"].get())
        if epochs_value == 0:
            tk.messagebox.showerror("Training configuration error", "The number of epochs for training the models can't be 0.")
            return False
        
        return True
    
    def update_config(self, event, listbox, options, param_key):
        selected_indices = listbox.curselection()  
        selected_values = [options[i] for i in selected_indices]  
        self.training_data[param_key] = selected_values

    def update_evaluation_metrics(self, event, listbox, options, param_key):
        selected_indices = listbox.curselection()
        selected_values = [options[i] for i in selected_indices]
        self.training_data[param_key] = selected_values

    def update_segmentation_models(self, event, listbox, options, param_key):
        selected_indices = listbox.curselection()
        selected_values = [options[i] for i in selected_indices]
        self.training_data[param_key] = selected_values

    def back(self):

            if not self.config_data["cluster_images"].get():
                self.controller.show_frame("ClusteringFrame")
                return

            if self.config_data["reduce_images"].get():
                self.controller.show_frame("ReductionConfigFrame")
            else:
                self.controller.show_frame("ReductionFrame")

    def save(self):
        
        if self.validate_form():
            self.config_data["training_data"].update(self.training_data)
            config_dict = FormUtils.save_config(self.config_data)
            self.final_dict.update(config_dict)
            result = tk.messagebox.askquestion(message="Do you wish to save the full configuration in a JSON or YAML file?", title="Configuration saving")
            if result == 'yes':
                self.save_dict_local()
            self.top.quit()
            self.top.destroy()            


    def save_dict_local(self):
        file_path = tk.filedialog.asksaveasfilename(
        defaultextension=".yaml", 
        filetypes=[("YAML files", "*.yaml;*.yml"), ("JSON files", "*.json")],
        title="Save Configuration File"
        )

        if not file_path:
            return  
        
        try:
            if file_path.endswith(".json"):
                with open(file_path, "w", encoding="utf-8") as json_file:
                    json.dump(self.final_dict, json_file, indent=4)
            elif file_path.endswith((".yaml", ".yml")):
                with open(file_path, "w", encoding="utf-8") as yaml_file:
                    yaml.dump(self.final_dict, yaml_file, default_flow_style=False, allow_unicode=True)
            
            tk.messagebox.showinfo("Success", f"Configuration saved successfully!")

            return

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save the file.\n\nError: {str(e)}")
            return

