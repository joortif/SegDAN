import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tktooltip import ToolTip
from src.utils.confighandler import ConfigHandler

import os

class GeneralConfigFrame(ttk.Frame):
    def __init__(self, parent, controller, config_data):
        ttk.Frame.__init__(self, parent)
        
        self.config_data = config_data
        self.controller = controller
        self.general_config = {
            "class_map_file" : tk.StringVar(value=""),
            "label_format": tk.StringVar(value=""),
            "class_mapping": {},
            "image_path": tk.StringVar(value=""),
            "label_path": tk.StringVar(value=""),
            "output_path": tk.StringVar(value=""),
            "verbose": tk.BooleanVar(value=False),
            "binary": tk.BooleanVar(value=False),
            "background": tk.IntVar(value=0)}

        label_title = ttk.Label(self, text="General configuration", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
        self.general_config_widgets()
        self.create_widgets()
        
        button_frame = ttk.Frame(self.general_frame)
        button_frame.grid(row=7, column=0, columnspan=5, pady=10)  

        self.general_frame.grid_rowconfigure(7, weight=0)
        self.general_frame.grid_columnconfigure(0, weight=1)

        button_back = ttk.Button(button_frame, text="Back", command=lambda: controller.show_frame("IntroductionFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, padx=50, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

        self.on_format_change()
        self.toggle_load_button()

    def create_widgets(self):
        self.row+=1
        self.class_map_label = tk.Label(self.general_frame, text="Class mapping")
        self.class_map_label.grid(row=self.row, column=0, padx=5)
        self.class_mapping = ttk.Entry(self.general_frame, textvariable=self.general_config["class_map_file"], width=50, state="readonly")
        self.class_mapping.grid(row=self.row, column=1, columnspan=2, padx=5, pady=5)
        self.class_map_bt = ttk.Button(self.general_frame, text="Select file 📄", command=lambda: self.select_file("class_map_file", "txt"))
        self.class_map_bt.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.class_map_bt, msg="Select a file containing the class mapping.\nThis file should contain the class ids and its class names separated by \":\".")
        self.load_classmap_bt = ttk.Button(self.general_frame, text="Load file 📄", command=lambda: self.load_file())
        self.load_classmap_bt.grid(row=self.row, column=4, padx=5, pady=5)
        ToolTip(self.load_classmap_bt, msg="Load the file containing the class mapping file")
        self.load_classmap_bt.grid_remove()  


        self.row+=1
        self.image_path_label = tk.Label(self.general_frame, text="Image path")
        self.image_path_label.grid(row=self.row, column=0, padx=5)
        self.entry_images = ttk.Entry(self.general_frame, textvariable=self.general_config["image_path"], width=50, state="readonly")
        self.entry_images.grid(row=self.row, column=1, columnspan=2, padx=5, pady=5)
        self.img_path_bt = ttk.Button(self.general_frame, text="Select folder 📂", command=lambda: self.select_folder("image_path"))
        self.img_path_bt.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.img_path_bt, msg="Select the folder containing the images.")

        self.row+=1
        self.label_path_label = tk.Label(self.general_frame, text="Label path")
        self.label_path_label.grid(row=self.row, column=0, padx=5)
        self.label_entry = ttk.Entry(self.general_frame, textvariable=self.general_config["label_path"], width=50, state="readonly")
        self.label_entry.grid(row=self.row, column=1, columnspan=2, padx=5, pady=5)
        
        self.label_path_bt = ttk.Button(self.general_frame, text="Select folder 📂", command=lambda: self.select_folder("label_path"))
        self.label_path_bt.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.label_path_bt, msg="Select the folder containing the labels.")
        
        self.label_file_bt = ttk.Button(self.general_frame, text="Select file 📄", command=lambda: self.select_file("label_path", "json"))
        self.label_file_bt.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.label_file_bt, msg="Select the annotation file.")

        self.row+=1
        self.output_path_label = tk.Label(self.general_frame, text="Output path")
        self.output_path_label.grid(row=self.row, column=0, padx=5)
        self.output_entry = ttk.Entry(self.general_frame, textvariable=self.general_config["output_path"], width=50, state="readonly")
        self.output_entry.grid(row=self.row, column=1, columnspan=2, padx=5, pady=5)
        self.output_path_bt = ttk.Button(self.general_frame, text="Select folder 📂", command=lambda: self.select_folder("output_path"))
        self.output_path_bt.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.output_path_bt, msg="Select the folder where the results will be saved.")

        self.row+=1
        self.verbose_label = tk.Label(self.general_frame, text="Verbose")
        self.verbose_label.grid(row=self.row, column=1, padx=5, pady=5)
        
        self.binary_label = tk.Label(self.general_frame, text="Binary")
        self.binary_label.grid(row=self.row, column=2, padx=5, pady=5)
        
        self.background_label = tk.Label(self.general_frame, text="Background")
        self.background_label.grid(row=self.row, column=3, padx=5, pady=5)

        self.row+=1
        self.verbose_checkbutton = ttk.Checkbutton(self.general_frame, variable=self.general_config["verbose"])
        self.verbose_checkbutton.grid(row=self.row, column=1, padx=5, pady=5)
        ToolTip(self.verbose_label, msg="Enable detailed logging during execution.")
        
        self.binary_checkbutton = ttk.Checkbutton(self.general_frame, variable=self.general_config["binary"])
        self.binary_checkbutton.grid(row=self.row, column=2, padx=5, pady=5)
        ToolTip(self.binary_label, msg="Set to True for binary segmentation (foreground/background).")

        self.background_entry = ttk.Entry(self.general_frame, textvariable=self.general_config["background"], width=5)
        self.background_entry.grid(row=self.row, column=3, padx=5, pady=5)
        ToolTip(self.background_label, msg="Class label assigned to background pixels.")

    def general_config_widgets(self):
        self.general_frame = tk.Frame(self, padx=10, pady=10)
        self.general_frame.grid(row=1, column=0, padx=10, pady=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.row=0
        self.label_format_label = tk.Label(self.general_frame, text="Label format")
        self.label_format_label.grid(row=self.row, column=0, padx=10, sticky="e")
        self.label_format_dropdown = ttk.Combobox(self.general_frame, textvariable=self.general_config["label_format"], values=ConfigHandler.CONFIGURATION_VALUES["label_formats"], state="readonly", width=15)
        self.label_format_dropdown.grid(row=self.row, column=1, padx=10, pady=5, sticky="w") 
        ToolTip(self.label_format_dropdown, msg="Select the label format.\nTXT for YOLO, JSON for COCO and multilabel for multiple labels per image.")

        self.general_config["label_format"].trace("w", lambda *args: self.on_format_change())
        self.general_config["class_map_file"].trace_add("write", lambda *args: self.toggle_load_button())
        


    def select_file(self, file_type, extension):
        filetypes = [("All Files", "*.*"), ("Text Files", "*.txt"), ("JSON Files", "*.json")]
        if extension == "txt":
            filetypes = [filetypes[1]]
        elif extension == "json":
            filetypes = [filetypes[2]]
        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        
        if file_path:
            self.general_config[file_type].set(file_path)
            self.toggle_load_button()

    def load_file(self):
        file_path = self.general_config["class_map_file"].get()
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = f.readlines()
                    class_mapping = {}
                    for line in data:
                        parts = line.strip().split(":")
                        if len(parts) == 2:
                            class_id, class_name = parts
                            class_name = class_name.strip()
                            if class_name == "":
                                messagebox.showerror("Error", f"ID {class_id} does not have any class name assigned.")
                                return
                            class_mapping[class_name.strip()] = int(class_id.strip())
                        else:
                            messagebox.showerror("Error", "Invalid class mapping format. Each line should be in 'id:class_name' format.")
                            return
                    
                    self.general_config["class_mapping"] = class_mapping
                    messagebox.showinfo("Success", "Class mapping file loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def select_folder(self, key):
        folder = filedialog.askdirectory(title=f"Select {key.replace('_', ' ')}")
        if folder:
            self.general_config[key].set(folder)
            

    def on_format_change(self):
        label_format = self.general_config["label_format"].get()
        
        show_image_related = bool(label_format)  
        show_class_mapping = label_format != "" and label_format != "json"  
        show_label_file = label_format == "json"  

        for widget in [self.image_path_label, self.entry_images, self.img_path_bt,
                    self.label_path_label, self.label_entry, self.label_path_bt, self.label_file_bt,
                    self.output_path_label, self.output_entry, self.output_path_bt,
                    self.verbose_label, self.verbose_checkbutton,
                    self.binary_label, self.binary_checkbutton,
                    self.background_label, self.background_entry]:
            widget.grid_remove()
        
        if show_image_related:
            for widget in [self.image_path_label, self.entry_images, self.img_path_bt,
                        self.output_path_label, self.output_entry, self.output_path_bt,
                        self.verbose_label, self.verbose_checkbutton,
                        self.binary_label, self.binary_checkbutton,
                        self.background_label, self.background_entry]:
                widget.grid()
        
        if show_class_mapping:
            for widget in [self.class_map_label, self.class_mapping, self.class_map_bt]:
                widget.grid()
        else:
            for widget in [self.class_map_label, self.class_mapping, self.class_map_bt]:
                widget.grid_remove()
        
        if label_format:
            if show_label_file:
                self.label_path_bt.grid_remove()
                self.label_file_bt.grid()
            else:
                self.label_file_bt.grid_remove()
                self.label_path_bt.grid()
        
        if show_image_related:
            for widget in [self.label_path_label, self.label_entry]:
                widget.grid()

    def validate_form(self):
        
        class_map_path = self.general_config["class_map_file"].get()
        if class_map_path != "" and not self.general_config["class_mapping"]:  
            result_warning = messagebox.askokcancel("Warning", "Class map file path defined but not loaded. If you continue, the map file will not be used.")

            if not result_warning:
                return False
            
            self.general_config["class_map_file"] = tk.StringVar(value="")

        label_format = self.general_config["label_format"].get()

        if label_format == "":
            tk.messagebox.showerror("Error", "You must choose a label format before proceeding.")
            return False
        image_path = self.general_config["image_path"].get()
        
        if image_path == "":
            tk.messagebox.showerror("Error", "You must select an image path before proceeding.")
            return False

        files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        if len(files) == 0:
            tk.messagebox.showerror("Error", "The selected image folder does not contain images.")
            return False

        non_images = [f for f in files if os.path.isfile(os.path.join(image_path, f)) and not f.lower().endswith(tuple(ConfigHandler.VALID_IMAGE_EXTENSIONS))]

        if non_images:
            if len(non_images) <10:   
                messagebox.showerror("Error", f"The folder contains non-images files:\n{', '.join(non_images)}")
            else:
                messagebox.showerror("Error", f"The folder contains non-images files (showing first 10 invalid files):\n{', '.join(non_images[:10])}")

            return False
        
        label_path = self.general_config["label_path"].get()

        if label_path == "":
            tk.messagebox.showerror("Error", "You must select a label path before proceeding.")
            return False
        
        label_ext = f".{label_format}"
        if label_format == "multilabel":
            label_ext = ConfigHandler.VALID_IMAGE_EXTENSIONS

        label_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
        if len(label_files) == 0:
            tk.messagebox.showerror("Error", "The selected label folder does not contain images.")
            return False

        non_valid_labels = [f for f in label_files if os.path.isfile(os.path.join(label_path, f)) and not f.lower().endswith(tuple(label_ext))]

        if non_valid_labels:
            if len(non_valid_labels) <10:   
                messagebox.showerror("Error", f"The folder contains labels with invalid extensions:\n{', '.join(non_valid_labels)}")
            else:
                messagebox.showerror("Error", f"The folder contains labels with invalid extensions (showing first 10 invalid files):\n{', '.join(non_valid_labels[:10])}")

            return False

        output_path = self.general_config["output_path"].get()

        if output_path == "":
            tk.messagebox.showerror("Error", "You must select an output path before proceeding.")
            return False
        
        return True

    def toggle_load_button(self):
        if self.general_config["class_map_file"].get() != "":
            self.load_classmap_bt.grid()
        else:
            self.load_classmap_bt.grid_remove()

    def next(self):
        
        validation_ok = self.validate_form()
        if validation_ok:
            self.config_data.update(self.general_config)
            print(self.config_data)
            self.controller.show_frame("AnalysisConfigFrame")



    
