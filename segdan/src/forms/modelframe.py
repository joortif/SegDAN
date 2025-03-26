import tkinter as tk
from tkinter import BOTH, RIGHT, Scrollbar, ttk
from tktooltip import ToolTip
import json
import yaml

from src.utils.confighandler import ConfigHandler
from src.forms.formutils import FormUtils

class ModelConfigFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)
        self.top = parent
        self.config_data = config_data
        self.controller = controller
        self.final_dict = final_dict

        self.semantic_models = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]
        self.instance_models = ConfigHandler.CONFIGURATION_VALUES["instance_segmentation_models"]
        self.model_sizes = list(ConfigHandler.CONFIGURATION_VALUES["model_sizes"].keys())
        self.semantic_metrics = ConfigHandler.CONFIGURATION_VALUES["semantic_metrics"]
        self.instance_metrics = ConfigHandler.CONFIGURATION_VALUES["instance_metrics"]


        if "model_data" not in self.config_data:
            self.config_data["model_data"] = {
                "segmentation": self.config_data.get("segmentation", tk.StringVar(value="semantic")),
                "models": self.config_data.get("models", []),
                "evaluation_metrics": self.config_data.get("evaluation_metrics", []),
                "selection_metric": self.config_data.get("selection_metric", tk.StringVar(value="dice_score")),
                "epochs": self.config_data.get("epochs", tk.StringVar(value="100"))
                }
            
        self.model_data = self.config_data["model_data"]
        
        
        self.row=0

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        label_title = ttk.Label(self, text="Model configuration", font=("Arial", 18, "bold"))
        label_title.grid(row=self.row, column=0, columnspan=5, pady=(20,10), padx=10)
            
        self.model_frame = tk.Frame(self, padx=10, pady=10)
        self.model_frame.grid(row=self.row+1, column=0, padx=10, pady=10)

        self.model_config_widgets()

        button_frame = ttk.Frame(self.model_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.model_frame.grid_rowconfigure(0, weight=0)
        self.model_frame.grid_columnconfigure(0, weight=0)
        self.model_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=self.back)
        button_back.grid(row=0, column=0, padx=50, pady=10, sticky="w")

        button_save = ttk.Button(button_frame, text="Save configuration", command=self.save)
        button_save.grid(row=0, column=1, pady=10, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

    def model_config_widgets(self):
        self.models_list = self.semantic_models
        segmentation_types = ConfigHandler.CONFIGURATION_VALUES["segmentation"]

        val_int = (self.top.register(self.validate_int), "%P")

        self.model_config_labelframe = ttk.LabelFrame(self.model_frame, text="Segmentation model configuration")
        self.model_config_labelframe.grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="ew")

        self.segmentation_type_label = tk.Label(self.model_config_labelframe, text="Segmentation type")
        self.segmentation_type_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.segmentation_type_combobox = ttk.Combobox(
            self.model_config_labelframe, textvariable=self.model_data["segmentation"],
            values=segmentation_types, state="readonly", width=15
        )
        self.segmentation_type_combobox.grid(row=1, column=0, padx=10, pady=10)
        ToolTip(self.segmentation_type_label, msg="Method used for segmenting images in the dataset.")

        self.segmentation_models_label = tk.Label(self.model_config_labelframe, text="Main segmentation model *")
        self.segmentation_models_label.grid(row=0, column=1, sticky="w", padx=10, pady=10)

        self.segmentation_models_listbox = tk.Listbox(
            self.model_config_labelframe, selectmode="single", height=10, exportselection=0, width=5
        )
        
        scrollbar_models = tk.Scrollbar(self.segmentation_models_listbox, orient=tk.VERTICAL, command=self.segmentation_models_listbox.yview, width=0)
        scrollbar_models.grid(row=0, column=1, sticky="nsew")

        self.segmentation_models_listbox.config(yscrollcommand=scrollbar_models.set)

        self.segmentation_models_listbox.grid(row=1, column=1, padx=10, pady=10, sticky="ew", rowspan=2)
        ToolTip(self.segmentation_models_label, msg="Select the primary segmentation model for training.")

        self.model_size_label = tk.Label(self.model_config_labelframe, text="Model size *")
        self.model_size_label.grid(row=0, column=2, sticky="w", padx=10, pady=10)
        ToolTip(self.model_size_label, msg="Select the size of the model. Smaller models are faster but less accurate while large models are more accurate but slower, using more resources.")


        self.model_size_combobox = ttk.Combobox(
            self.model_config_labelframe, values=self.model_sizes, state="readonly", width=15
        )
        self.model_size_combobox.grid(row=1, column=2, padx=10, pady=10)
        
        self.evaluation_metrics_label = tk.Label(self.model_config_labelframe, text="Evaluation metrics *")
        self.evaluation_metrics_label.grid(row=3, column=0, sticky="w", padx=10, pady=10)

        self.evaluation_metrics_listbox = tk.Listbox(
            self.model_config_labelframe, selectmode="multiple", height=5, exportselection=0, width=50
        )

        scrollbar_metrics = tk.Scrollbar(self.evaluation_metrics_listbox, orient=tk.VERTICAL, command=self.evaluation_metrics_listbox.yview, width=0)
        scrollbar_metrics.grid(row=0, column=1, sticky="nsew")

        self.evaluation_metrics_listbox.config(yscrollcommand=scrollbar_metrics.set)

        self.evaluation_metrics_listbox.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        ToolTip(self.evaluation_metrics_label, msg="Metrics used to assess model performance.")

        self.epochs_label = tk.Label(self.model_config_labelframe, text="Epochs *")
        self.epochs_label.grid(row=5, column=0, sticky="w", padx=10, pady=10)

        self.epochs_entry = tk.Entry(self.model_config_labelframe, textvariable=self.model_data['epochs'], width=10, validate="key", validatecommand=val_int)
        self.epochs_entry.grid(row=6, column=0, sticky="w", padx=10, pady=10)
        ToolTip(self.epochs_label, msg="Number of times the entire dataset is passed through the model during training.")

        self.additional_models_label = tk.Label(self.model_config_labelframe, text="Select additional models for comparison", wraplength=200)
        self.additional_models_label.grid(row=5, column=1, padx=10, pady=10)

        self.compare_models_var = tk.BooleanVar(value=False)
        self.compare_models_checkbox = ttk.Checkbutton(
            self.model_config_labelframe, variable=self.compare_models_var,
            command=self.toggle_comparison_section
        )
        self.compare_models_checkbox.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        self.comparison_frame = ttk.LabelFrame(self.model_frame, text="Comparison Models")
        self.comparison_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.comparison_frame.grid_remove()  

        self.added_models_label = tk.Label(self.comparison_frame, text="Models")
        self.added_models_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.added_models_listbox = tk.Listbox(self.comparison_frame, height=5, exportselection=0)
        self.added_models_listbox.grid(row=1, column=0, padx=10, pady=10, sticky="ew", rowspan=2)

        self.add_button = ttk.Button(self.comparison_frame, text="Add", command=self.open_add_model_form)
        self.add_button.grid(row=1, column=1, padx=10, pady=10)

        self.edit_button = ttk.Button(self.comparison_frame, text="Edit", command=self.open_edit_model_form)

        self.segmentation_selection_metric_label = tk.Label(self.comparison_frame, text="Model selection metric")
        self.segmentation_selection_metric_label.grid(row=0, column=2, sticky="w", padx=10, pady=10)

        self.segmentation_selection_metric_combobox = ttk.Combobox(
            self.comparison_frame, textvariable=self.model_data["selection_metric"],
            state="readonly", width=15
        )
        self.segmentation_selection_metric_combobox.grid(row=1, column=2, padx=10, pady=10)
        ToolTip(self.segmentation_selection_metric_label, msg="Metric used to choose the best segmentation model.")

        self.update_listbox(self.evaluation_metrics_listbox, self.semantic_metrics)
        self.update_listbox(self.segmentation_models_listbox, self.semantic_models)

        self.segmentation_type_combobox.bind("<<ComboboxSelected>>", self.update_segmentation)

        self.added_models_listbox.bind("<<ListboxSelect>>", self.toggle_edit_button)

    def update_listbox(self, listbox, items):
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)

    def update_segmentation(self, event):
        selected_type = self.model_data["segmentation"].get()

        if selected_type == "semantic":
            self.models_list = self.semantic_models

            self.update_listbox(self.evaluation_metrics_listbox, self.semantic_metrics)
            self.update_listbox(self.segmentation_models_listbox, self.semantic_models)

        elif selected_type == "instance":
            self.models_list = self.instance_models

            self.update_listbox(self.evaluation_metrics_listbox, self.instance_metrics)
            self.update_listbox(self.segmentation_models_listbox, self.instance_models)
        
        self.model_data["evaluation_metrics"] = []
        self.model_data["models"] = []

    def validate_int(self, value):
        return value.isdigit() or value == ""

    def toggle_comparison_section(self):
        if self.compare_models_var.get():
            self.comparison_frame.grid()
        else:
            self.comparison_frame.grid_remove()

    def toggle_edit_button(self, event):
        if self.added_models_listbox.curselection():
            self.edit_button.config(state="normal")
        else:
            self.edit_button.config(state="disabled")

    def open_add_model_form(self):
        self.open_model_form(mode="add")

    def open_edit_model_form(self):
        selected_index = self.added_models_listbox.curselection()
        if not selected_index:
            tk.messagebox.showerror("Edit model", f"You need to select a model from the list.")
            return
        self.open_model_form(mode="edit", selected_index=selected_index[0])

    def open_model_form(self, mode="add", selected_index=None):
        self.model_form = tk.Toplevel(self)
        self.model_form.title("Add Model" if mode == "add" else "Edit Model")

        self.model_form.grab_set()

        window_width = 400
        window_height = 200

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        self.model_form.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.model_form.resizable(False, False)

        self.model_form.grid_columnconfigure(0, weight=1)
        self.model_form.grid_columnconfigure(1, weight=1)


        tk.Label(self.model_form, text="Model").grid(row=0, column=0, padx=10, pady=20, sticky="e")
        tk.Label(self.model_form, text="Size").grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.model_var = tk.StringVar(value=self.models_list[0])
        self.size_var = tk.StringVar(value=self.model_sizes[0])

        self.model_combobox = ttk.Combobox(self.model_form, textvariable=self.model_var, values=self.models_list, width=15, state="readonly")
        self.model_combobox.grid(row=0, column=1, padx=10, pady=20, sticky="w")

        self.size_combobox = ttk.Combobox(self.model_form, textvariable=self.size_var, values=self.model_sizes, width=15, state="readonly")
        self.size_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        if mode == "edit":
            selected_model = self.added_models_listbox.get(selected_index)
            model_name, model_size = selected_model.split(" - ")
            self.model_var.set(model_name)
            self.size_var.set(model_size)

        button_text = "Add" if mode == "add" else "Save"
        self.save_button = ttk.Button(self.model_form, text=button_text, command=lambda: self.save_model(mode, selected_index))

        if mode == "edit":
            self.save_button.grid(row=2, column=0, padx=10, pady=10, sticky="e")
            self.delete_button = ttk.Button(self.model_form, text="Delete", command=lambda: self.delete_model(selected_index))
            self.delete_button.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        else:
            self.save_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def save_model(self, mode, selected_index):
        model_name = self.model_var.get()
        model_size = self.size_var.get()
        model_entry = f"{model_name} - {model_size}"
        model_info = {"model_name": model_name, "model_size": model_size}

        for existing_model in self.model_data["models"]:
            if existing_model["model_name"] == model_name and existing_model["model_size"] == model_size:
                tk.messagebox.showerror("Add model error", f"A {model_size.lower()} {model_name} has already been added.")
                return

        if mode == "add":
            self.added_models_listbox.insert(tk.END, model_entry)

            self.model_data["models"].append(model_info)
        elif mode == "edit":
            self.added_models_listbox.delete(selected_index)
            self.added_models_listbox.insert(selected_index, model_entry)

            self.model_data["models"][selected_index] = model_info

        if not self.edit_button.winfo_ismapped():
            self.edit_button.grid(row=2, column=1, padx=10, pady=10)

        self.model_form.destroy()

    def delete_model(self, selected_index):
        confirm = tk.messagebox.askyesno("Confirmation", "Are you sure you want to delete this model from the list?")
        if confirm:
            self.added_models_listbox.delete(selected_index)

            self.model_data["models"].pop(selected_index)

            if len(self.model_data["models"]) == 0:
                self.edit_button.grid_remove()

            self.model_form.destroy()


    def validate_form(self):

        if len(self.model_data["evaluation_metrics"]) == 0:
            tk.messagebox.showerror("Dataset split error", "You must select at least one evaluation metric.")
            return False
        
        if len(self.model_data["models"]) == 0:
            tk.messagebox.showerror("Dataset split error", "You must select at least one model for training.")
            return False
    
        if self.model_data["epochs"].get().strip() == "":
            tk.messagebox.showerror("Dataset split error", "The number of epochs for training the models can't be empty.")
            return False
        
        epochs_value = int(self.model_data["epochs"].get())
        if epochs_value == 0:
            tk.messagebox.showerror("Dataset split error", "The number of epochs for training the models can't be 0.")
            return False

        return True

    def save(self):
        
        if self.validate_form():
            self.config_data["model_data"].update(self.model_data)
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
        
    def back(self):
        self.controller.show_frame("DatasetSplitFrame")