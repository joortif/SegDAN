import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tktooltip import ToolTip

import os
import yaml
import json

class ConfigFileLoaderFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)

        self.top = parent
        self.reduction_var = tk.BooleanVar(value=False)
        self.config_data = config_data
        self.controller = controller
        self.config_path = tk.StringVar(value="")
        self.final_dict = final_dict

        label_title = ttk.Label(self, text="Load configuration settings", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.config_loader_frame = tk.Frame(self, padx=10, pady=10)
        self.config_loader_frame.grid(row=1, column=0, padx=10, pady=10, columnspan=3)

        self.config_loader_frame.grid_rowconfigure(0, weight=0)
        self.config_loader_frame.grid_columnconfigure(0, weight=0)
        self.config_loader_frame.grid_columnconfigure(1, weight=0)

        self.config_file_label = ttk.Label(self.config_loader_frame, text="Configuration file")
        self.config_file_label.grid(row=0, column=0, padx=5, pady=10)

        self.config_file_entry = ttk.Entry(self.config_loader_frame, textvariable=self.config_path, width=50, state="readonly")
        self.config_file_button = tk.Button(self.config_loader_frame, text="Select file 📄", command=lambda: self.select_file())
        ToolTip(self.config_file_button, msg="Select the configuration file (JSON or YAML).")

        self.config_file_load_button = tk.Button(self.config_loader_frame, text="Load file 📄", command=lambda: self.load_file())
        ToolTip(self.config_file_load_button, msg="Load the selected configuration file.")

        self.config_file_entry.grid(row=2, column=0, padx=5, pady=10)
        self.config_file_button.grid(row=2, column=1, padx=5, pady=10)
        self.config_file_load_button.grid(row=2, column=2, padx=5, pady=10)

        self.config_file_load_button.grid_remove()

        button_frame = ttk.Frame(self.config_loader_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="e")  

        button_back = ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("IntroductionFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_save = ttk.Button(button_frame, text="Save configuration", command=self.save_configuration)
        button_save.grid(row=0, column=1, pady=5, sticky="w")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def select_file(self):
        filetypes = [("YAML Files", "*.yaml;*.yml"),("JSON Files", "*.json")]
        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        
        if file_path:
            self.config_path.set(file_path)
            self.toggle_load_button()

    def toggle_load_button(self):
        if self.config_path.get().strip() != "":
            self.config_file_load_button.grid()
        else:
            self.config_file_load_button.grid_remove()

    def load_file(self):
        file_path = self.config_path.get().strip()
        
        if not file_path:
            messagebox.showwarning("No file selected", "Please select a configuration file first.")
            return
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.json':
                with open(file_path, 'r') as json_file:
                    loaded_data = json.load(json_file)
            elif file_extension == '.yaml' or file_extension == '.yml':
                with open(file_path, 'r') as yaml_file:
                    loaded_data = yaml.safe_load(yaml_file)
            else:
                messagebox.showerror("Invalid File", "Selected file is neither a JSON nor a YAML file.")
                return
            
            messagebox.showinfo("File Loaded", "Configuration file has been successfully loaded.")

            if 'class_mapping' in loaded_data:
                loaded_data['class_mapping'] = {int(k): v for k, v in loaded_data['class_mapping'].items()}
            
            if 'color_dict' in loaded_data:
                loaded_data['color_dict'] = {int(k): v for k, v in loaded_data['color_dict'].items()}

            self.final_dict.update(loaded_data)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the configuration file. Error: {str(e)}")

    def save_configuration(self):
        self.top.quit()

