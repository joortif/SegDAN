import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tktooltip import ToolTip

from segdan.utils.constants import LabelFormat
from segdan.forms.colorconfigform import ColorConfigForm
from segdan.forms.formutils import FormUtils
from segdan.utils.confighandler import ConfigHandler

import os

class GeneralConfigFrame(ttk.Frame):
    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)
        
        self.config_data = config_data
        self.controller = controller
        self.top = parent

        if "general_data" not in self.config_data:
            self.config_data["general_data"] = {
                "class_map_file" : self.config_data.get("class_map_file", tk.StringVar(value="")),
                "label_format": self.config_data.get("label_format", tk.StringVar(value="txt")),
                "color_dict": self.config_data.get("color_dict", {}),
                "class_mapping": self.config_data.get("class_mapping", {}),
                "image_path": self.config_data.get("image_path", tk.StringVar(value="")),
                "label_path": self.config_data.get("label_path", tk.StringVar(value="")),
                "output_path": self.config_data.get("output_path", tk.StringVar(value="")),
                "verbose": self.config_data.get("verbose", tk.BooleanVar(value=False)),
                "binary": self.config_data.get("binary", tk.BooleanVar(value=False)),
                "background": self.config_data.get("background", tk.StringVar(value="0")),
                "threshold": self.config_data.get("threshold", tk.StringVar(value="255"))
            }
            

        label_title = ttk.Label(self, text="General configuration", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)

        self.general_data = self.config_data["general_data"]

        self.general_config_widgets()
        self.create_widgets()
        
        button_frame = ttk.Frame(self.general_frame)
        button_frame.grid(row=15, column=0, columnspan=5, pady=10, sticky="e")  

        self.general_frame.grid_rowconfigure(7, weight=0)
        self.general_frame.grid_columnconfigure(0, weight=1)

        button_back = ttk.Button(button_frame, text="Back", command=lambda: controller.show_frame("IntroductionFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)

        self.on_format_change()
        self.toggle_load_button()

    def validate_numeric(self, value):
        if value == "" or value.isdigit():
            return True
        return False

    def validate_threshold(self, value):
        if value == "" or (value.isdigit() and 0 <= int(value) <= 255):
            return True
        return False

    def create_widgets(self):

        vcmd = (self.top.register(self.validate_numeric), "%P")
        val_threshold = (self.top.register(self.validate_threshold), "%P")

        self.color_map_label = tk.Label(self.general_labelframe, text="Color mapping")
        self.transformation_color_dict_text = tk.Text(self.general_labelframe, height=5, width=20, state="disabled")
        ToolTip(self.color_map_label, msg="Mapping of class IDs to colors.")

        self.transformation_color_dict_text.insert("1.0", str(self.general_data['color_dict']))  
        self.transformation_color_dict_text.config(state="disabled")  

        self.add_color_bt = tk.Button(self.general_labelframe, text="Add dictionary", command=lambda: self.open_color_form())
        ToolTip(self.add_color_bt, msg="Add colors to the dictionary.")

        self.class_map_label = tk.Label(self.general_labelframe, text="Class mapping")
        self.class_mapping = ttk.Entry(self.general_labelframe, textvariable=self.general_data["class_map_file"], width=50, state="readonly")
        self.class_map_bt = tk.Button(self.general_labelframe, text="Select file 📄", command= lambda: self.select_file("class_map_file", "txt"))
        ToolTip(self.class_map_label, msg="Mapping of class IDs to names (format: ID:Name)")
        ToolTip(self.class_map_bt, msg="Select a file containing the class mapping.")

        self.clear_classmap_bt = tk.Button(self.general_labelframe, text="Delete file 📄", command= lambda: self.delete_file())
        ToolTip(self.clear_classmap_bt, msg="Clears class mapping configuration.")
        
        self.image_path_label = tk.Label(self.general_labelframe, text="Image path *")
        self.entry_images = ttk.Entry(self.general_labelframe, textvariable=self.general_data["image_path"], width=50, state="readonly")
        self.img_path_bt = tk.Button(self.general_labelframe, text="Select folder 📂", command=lambda: self.select_folder("image_path"))
        ToolTip(self.image_path_label, msg="Path to the folder containing the images of the dataset.")
        ToolTip(self.img_path_bt, msg="Select the folder containing the images.")
        
        self.label_path_label = tk.Label(self.general_labelframe, text="Label path *")
        self.label_entry = ttk.Entry(self.general_labelframe, textvariable=self.general_data["label_path"], width=50, state="readonly")
        self.label_path_bt = tk.Button(self.general_labelframe, text="Select folder 📂", command=lambda: self.select_folder("label_path"))
        ToolTip(self.label_path_label, msg="Path to the folder containing the labels of the dataset.")
        ToolTip(self.label_path_bt, msg="Select the folder containing the labels.")
        
        self.label_file_bt = tk.Button(self.general_labelframe, text="Select file 📄", command=lambda: self.select_file("label_path", "json"))
        ToolTip(self.label_file_bt, msg="Select the annotation file.")
       
        self.output_path_label = tk.Label(self.general_labelframe, text="Output path *")
        self.output_entry = ttk.Entry(self.general_labelframe, textvariable=self.general_data["output_path"], width=50, state="readonly")
        self.output_path_bt = tk.Button(self.general_labelframe, text="Select folder 📂", command=lambda: self.select_folder("output_path"))
        ToolTip(self.label_path_label, msg="Path to the folder for the results created by the program.")
        ToolTip(self.output_path_bt, msg="Select the folder where the results will be saved.")
        
        self.verbose_label = tk.Label(self.general_labelframe, text="Verbose")
        self.binary_label = tk.Label(self.general_labelframe, text="Binary")
        self.background_label = tk.Label(self.general_labelframe, text="Background")
        self.binary_threshold_label = tk.Label(self.general_labelframe, text="Binary threshold")
        
        self.verbose_checkbutton = ttk.Checkbutton(self.general_labelframe, variable=self.general_data["verbose"])
        ToolTip(self.verbose_label, msg="Enable detailed logging during execution.")
        
        self.binary_checkbutton = ttk.Checkbutton(self.general_labelframe, variable=self.general_data["binary"], command=lambda: FormUtils.toggle_label_entry(self.general_data["binary"], self.binary_threshold_label, 
                                                                                                                          self.threshold_entry, None, 12, 3))
        ToolTip(self.binary_label, msg="Set to True for binary segmentation (foreground/background).")
        
        self.background_entry = ttk.Entry(self.general_labelframe, textvariable=self.general_data["background"], width=5, validate="key", validatecommand=vcmd)
        ToolTip(self.background_label, msg="Class label assigned to background pixels.")

        self.threshold_entry = ttk.Entry(self.general_labelframe, textvariable=self.general_data["threshold"], width=5, validate="key", validatecommand=val_threshold)
        ToolTip(self.binary_threshold_label, msg="Binary label threshold value for two-class segmentation.\nIt must be between 0 and 255.")

        self.row+=1
        self.class_map_label.grid(row=self.row, column=0, padx=5, sticky="w")

        self.row+=1
        self.class_mapping.grid(row=self.row, column=0, columnspan=3, padx=5, pady=5)
        self.class_map_bt.grid(row=self.row, column=3, padx=5, pady=5)
        self.clear_classmap_bt.grid(row=self.row, column=6, padx=5, pady=5)
        self.clear_classmap_bt.grid_remove()

        self.row+=1
        self.color_map_label.grid(row=self.row, column=0, padx=5, pady=5, sticky="w")

        self.row+=1
        self.transformation_color_dict_text.grid(row=self.row, column=0, columnspan=3, pady=5, padx=5, sticky="we")
        self.add_color_bt.grid(row=self.row, column=3, padx=5, pady=5)        

        self.row+=1
        self.image_path_label.grid(row=self.row, column=0, padx=5, sticky="w")
        
        self.row+=1
        self.entry_images.grid(row=self.row, column=0, columnspan=3, padx=5, pady=5)
        self.img_path_bt.grid(row=self.row, column=3, padx=5, pady=5)

        self.row+=1
        self.label_path_label.grid(row=self.row, column=0, padx=5, sticky="w")

        self.row+=1
        self.label_entry.grid(row=self.row, column=0, columnspan=3, padx=5, pady=5)
        self.label_path_bt.grid(row=self.row, column=3, padx=5, pady=5)
        self.label_file_bt.grid(row=self.row, column=3, padx=5, pady=5)

        self.row+=1
        self.output_path_label.grid(row=self.row, column=0, padx=5, sticky="w")

        self.row+=1
        self.output_entry.grid(row=self.row, column=0, columnspan=3, padx=5, pady=5)
        self.output_path_bt.grid(row=self.row, column=3, padx=5, pady=5)

        self.row+=1
        self.verbose_label.grid(row=self.row, column=0, padx=5, pady=5)
        self.binary_label.grid(row=self.row, column=1, padx=5, pady=5)
        self.background_label.grid(row=self.row, column=2, padx=5, pady=5)
        self.binary_threshold_label.grid(row=self.row, column=3, padx=5, pady=5)

        self.row+=1
        self.verbose_checkbutton.grid(row=self.row, column=0, padx=5, pady=5)
        self.binary_checkbutton.grid(row=self.row, column=1, padx=5, pady=5)
        self.background_entry.grid(row=self.row, column=2, padx=5, pady=5)
        self.threshold_entry.grid(row=self.row, column=3, padx=5, pady=5)

        self.binary_threshold_label.grid_remove()
        self.threshold_entry.grid_remove()


    def open_color_form(self):
        color_form = ColorConfigForm(self.top, self.general_data["color_dict"])

        color_form.top.transient(self.top)
        color_form.top.grab_set()

        FormUtils.center_window(color_form, self.top)
        
        color_form.top.focus_set()

        self.top.wait_window(color_form.top)

        if color_form.colors:
            self.add_colors(color_form.colors)
            return
        
        self.transformation_color_dict_text.config(state="normal")  
        self.transformation_color_dict_text.delete(1.0, tk.END) 
        self.transformation_color_dict_text.config(state="disabled")

    def add_colors(self, colors):
        self.general_data["color_dict"] = colors

        colors_text = "\n".join([f"{color}: {class_id}" for color, class_id in colors.items()])

        self.transformation_color_dict_text.config(state="normal")  
        self.transformation_color_dict_text.delete(1.0, tk.END)  
        self.transformation_color_dict_text.insert(tk.END, colors_text)
        self.transformation_color_dict_text.config(state="disabled")

    def general_config_widgets(self):
        self.general_frame = tk.Frame(self, padx=10, pady=10)
        self.general_frame.grid(row=1, column=0, padx=10, pady=10)

        self.general_labelframe = ttk.LabelFrame(self.general_frame, text="General configuration", padding=(20,10))
        self.general_labelframe.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.row=0
        self.label_format_label = tk.Label(self.general_labelframe, text="Label format")
        self.label_format_label.grid(row=self.row, column=1, padx=10)

        self.row+=1
        self.label_format_dropdown = ttk.Combobox(self.general_labelframe, textvariable=self.general_data["label_format"], values=ConfigHandler.CONFIGURATION_VALUES["label_formats"], state="readonly", width=15)
        self.label_format_dropdown.grid(row=self.row, column=1, padx=10, pady=5) 
        ToolTip(self.label_format_label, msg="Select the label format.\nTXT for YOLO, JSON for COCO and mask for multiple labels per image.")

        self.general_data["label_format"].trace("w", lambda *args: self.on_format_change())
        self.general_data["class_map_file"].trace_add("write", lambda *args: self.toggle_load_button())

    def select_file(self, file_type, extension):
        filetypes = [("All Files", "*.*"), ("Text Files", "*.txt"), ("JSON Files", "*.json")]
        if extension == LabelFormat.TXT.value:
            filetypes = [filetypes[1]]
        elif extension == LabelFormat.JSON.value:
            filetypes = [filetypes[2]]
        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        
        if file_path:
            self.general_data[file_type].set(file_path) 
        
            if file_type == "class_map_file":
                self.load_class_map(file_path)
            
            self.toggle_load_button()

    def update_color_map(self):

        class_mapping = self.general_data["class_mapping"]
        color_map = self.general_data["color_dict"]

        for class_name, class_id in class_mapping.items():
            if class_id not in color_map:
                color_map[class_id] = ""

        self.general_data["color_dict"] = color_map
        self.add_colors(color_map)


    def load_class_map(self, file_path):
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
                            messagebox.showerror("Class file configuration error", f"ID {class_id} does not have any class name assigned.")
                            return
                        class_mapping[int(class_id.strip())] = class_name.strip()
                    else:
                        messagebox.showerror("Class file configuration error", "Invalid class mapping format. Each line should be in 'id:class_name' format.")
                        return
                    
                self.general_data["class_mapping"] = class_mapping
                    
                if self.general_data["label_format"].get() == LabelFormat.MASK.value:
                    self.update_color_map()
                messagebox.showinfo("Success", "Class mapping file loaded successfully.")
        except Exception as e:
                messagebox.showerror("Class file configuration error", f"Failed to load file: {e}")

    def delete_file(self):
        self.general_data["class_map_file"].set("")
        self.general_data["class_mapping"] = {}

    def select_folder(self, key):
        folder = filedialog.askdirectory(title=f"Select {key.replace('_', ' ')}")
        if folder:
            self.general_data[key].set(folder)
            

    def on_format_change(self):
        label_format = self.general_data["label_format"].get()
        
        show_image_related = bool(label_format)  
        show_class_mapping = label_format != "" and label_format != "json"  
        show_label_file = label_format == LabelFormat.JSON.value  
        show_color_dict = label_format == LabelFormat.MASK.value

        for widget in [self.color_map_label, self.add_color_bt, self.transformation_color_dict_text, 
                    self.image_path_label, self.entry_images, self.img_path_bt,
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
            if show_color_dict:
                self.color_map_label.grid()
                self.add_color_bt.grid()
                self.transformation_color_dict_text.grid()
            else:
                self.color_map_label.grid_remove()
                self.add_color_bt.grid_remove()
                self.transformation_color_dict_text.grid_remove()

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
        
        class_map_path = self.general_data["class_map_file"].get().strip()
        if class_map_path != "" and not self.general_data["class_mapping"]:  
            result_warning = messagebox.askokcancel("Warning", "Class map file path defined but not loaded. If you continue, the map file will not be used.")

            if not result_warning:
                return False
            
            self.general_data["class_map_file"] = tk.StringVar(value="")

        label_format = self.general_data["label_format"].get().strip()

        if label_format == "":
            tk.messagebox.showerror("General configuration error", "You must choose a label format before proceeding.")
            return False
        image_path = self.general_data["image_path"].get().strip()
        
        if image_path == "":
            tk.messagebox.showerror("General configuration error", "You must select an image path before proceeding.")
            return False

        images = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        if len(images) == 0:
            tk.messagebox.showerror("General configuration error", "The selected image folder does not contain images.")
            return False

        non_images = [f for f in images if os.path.isfile(os.path.join(image_path, f)) and not f.lower().endswith(tuple(list(ConfigHandler.VALID_IMAGE_EXTENSIONS)))]

        if non_images:
            if len(non_images) <10:   
                messagebox.showerror("General configuration error", f"The folder contains non-images files:\n{', '.join(non_images)}")
            else:
                messagebox.showerror("General configuration error", f"The folder contains non-images files (showing first 10 invalid files):\n{', '.join(non_images[:10])}")

            return False
        
        label_path = self.general_data["label_path"].get().strip()

        if label_path == "":
            tk.messagebox.showerror("General configuration error", "You must select a label path before proceeding.")
            return False
        
        label_ext = f".{label_format}"
        if label_format == LabelFormat.MASK.value:
            label_ext = ConfigHandler.VALID_IMAGE_EXTENSIONS

        if label_format != "json":

            label_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            if len(label_files) == 0:
                tk.messagebox.showerror("General configuration error", "The selected label folder does not contain images.")
                return False

            non_valid_labels = [f for f in label_files if os.path.isfile(os.path.join(label_path, f)) and not f.lower().endswith(tuple(label_ext))]

            if non_valid_labels:
                if len(non_valid_labels) <10:   
                    messagebox.showerror("General configuration error", f"The folder contains labels with invalid extensions:\n{', '.join(non_valid_labels)}")
                else:
                    messagebox.showerror("General configuration error", f"The folder contains labels with invalid extensions (showing first 10 invalid files):\n{', '.join(non_valid_labels[:10])}...")

                return False
            
            if len(label_files) != len(images):
                img_names = {os.path.splitext(image)[0] for image in images}
                label_names = {os.path.splitext(label)[0] for label in label_files}

                images_not_in_labels = img_names - label_names

                images_unmatched = [f for f in images if os.path.splitext(f)[0] in images_not_in_labels]

                move_images = messagebox.askyesno("Missing label files", f"The following images don't have label files in the label directory:\n{', '.join(images_unmatched)} \n"
                                    "All images need to have their respective labels for proper training.\nDo you wish to move these images automatically?")

                if not move_images:
                    return False
                
                save_path = os.path.join(os.path.dirname(image_path), "ignored_images")

                os.makedirs(save_path,exist_ok=True)

                for img in images_unmatched:
                    src = os.path.join(image_path, img)
                    dst = os.path.join(save_path, img)
                    shutil.move(src, dst)
                
                messagebox.showinfo("Missing label files", f"Images without labels have been moved to {save_path}")

        output_path = self.general_data["output_path"].get().strip()

        if output_path == "":
            tk.messagebox.showerror("General configuration error", "You must select an output path before proceeding.")
            return False
        
        if self.general_data["binary"].get() and self.general_data["threshold"].get().strip() == "":
            tk.messagebox.showerror("General configuration error", "You must select a threshold when the labels are binary.")
            return False
        
        colors = self.general_data["color_dict"]
        
        if self.general_data["color_dict"] and not all(color == "" for color in colors.values()):
        
            for class_id, color in colors.items():
                if color == "":
                    tk.messagebox.showerror("General configuration error", f"Missing color for {class_id}. All classes must have a color.")
                    return False
        
        return True

    def toggle_load_button(self):
        if self.general_data["class_map_file"].get().strip() != "":
            self.clear_classmap_bt.grid()
        else:
            self.clear_classmap_bt.grid_remove()

    def next(self):
        
        validation_ok = self.validate_form()
        if validation_ok:
            self.config_data["general_data"].update(self.general_data)
            self.controller.show_frame("AnalysisConfigFrame")



    
