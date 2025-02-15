import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from src.forms.clusteringconfigform import ClusteringConfigForm
from src.forms.colorform import ColorConfigForm
from src.forms.clusteringmodelform import ClusteringModelForm
from src.forms.formutils import FormUtils
from src.utils.confighandler import ConfigHandler

class ConfigForm:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Configuración YAML")
        self.root.resizable(False, False)

        self.config_data = {
            "dataset_path": tk.StringVar(value=""),
            "output_path": tk.StringVar(value=""),
            "verbose": tk.BooleanVar(value=False),
            "binary": tk.BooleanVar(value=False),
            "background": tk.IntVar(value=0),

            "analyze": tk.BooleanVar(value=False),

            "threshold": tk.IntVar(value=255),
            "color_dict": None,

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
        self.models_list = ConfigHandler.CONFIGURATION_VALUES["semantic_segmentation_models"]

        self.create_widgets()

    def select_folder(self, key):
        folder = filedialog.askdirectory(title=f"Seleccionar {key.replace('_', ' ')}")
        if folder:
            self.config_data[key].set(folder)

    def create_widgets(self):
        self.general_config_widgets()
        self.analysis_config_widgets()
        self.transformation_config_widgets()
        self.train_config_widgets()

        tk.Button(self.root, text="Save configuration", command=self.save_config).grid(row=14, column=0, columnspan=2, pady=10)

    def general_config_widgets(self):
        general_frame = tk.LabelFrame(self.root, text="General configuration", padx=10, pady=10)
        general_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        general_frame.grid_columnconfigure(0, weight=1)
        general_frame.grid_columnconfigure(1, weight=0)
        general_frame.grid_columnconfigure(2, weight=2)

        tk.Label(general_frame, text="Dataset path").grid(row=0, column=0, sticky="w", padx=5)
        entry_dataset = ttk.Entry(general_frame, textvariable=self.config_data["dataset_path"], width=100, state="readonly")
        entry_dataset.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(general_frame, text="Select folder 📂", command=lambda: self.select_folder("dataset_path")).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(general_frame, text="Output path").grid(row=2, column=0, sticky="w", padx=5)
        entry_output = ttk.Entry(general_frame, textvariable=self.config_data["output_path"], width=100, state="readonly")
        entry_output.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(general_frame, text="Select folder 📂", command=lambda: self.select_folder("output_path")).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(general_frame, text="Verbose").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(general_frame, variable=self.config_data["verbose"]).grid(row=4, column=1, padx=5, pady=5)
        tk.Label(general_frame, text="Enable detailed logging during execution.", fg="gray", font=("Arial", 8)).grid(row=4, column=2, sticky="w", padx=5, pady=5)

        tk.Label(general_frame, text="Binary").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(general_frame, variable=self.config_data["binary"], command=lambda: FormUtils.toggle_label_entry(self.config_data["binary"], self.bin_threshold_label, self.bin_threshold_entry, None, 0, 2)).grid(row=5, column=1, padx=5, pady=5)
        tk.Label(general_frame, text="Set to True for binary segmentation (foreground/background).", fg="gray", font=("Arial", 8)).grid(row=5, column=2, sticky="w", padx=5, pady=5)

        tk.Label(general_frame, text="Background").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(general_frame, textvariable=self.config_data["background"], width=5).grid(row=6, column=1, padx=5, pady=5)
        tk.Label(general_frame, text="Class label assigned to background pixels.", fg="gray", font=("Arial", 8)).grid(row=6, column=2, sticky="w", padx=5, pady=5)

    def analysis_config_widgets(self):
        analyze_frame = tk.LabelFrame(self.root, text="Analysis configuration", padx=10, pady=10)
        analyze_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        analyze_frame.grid_columnconfigure(0, weight=1) 
        analyze_frame.grid_columnconfigure(2, weight=1)

        tk.Label(analyze_frame, text="Analysis").grid(row=6, column=0, sticky="s", padx=5)
        tk.Label(analyze_frame, text="Perform full scan and analysis of the dataset. Results will be saved in output path.", fg="gray", font=("Arial", 8)).grid(row=8, column=0)
        tk.Checkbutton(analyze_frame, variable=self.config_data["analyze"]).grid(row=7, column=0, padx=5, pady=5)

        tk.Label(analyze_frame, text="Cluster images").grid(row=6, column=1, sticky="s", padx=5)
        tk.Label(analyze_frame, text="Enable image similarity study using image embeddings.", fg="gray", font=("Arial", 8)).grid(row=8, column=1)
        tk.Button(analyze_frame, text="Add clustering configuration", command=self.open_clustering_config_form).grid(row=7, column=1)
        
    def transformation_config_widgets(self):
        self.training_frame = tk.LabelFrame(self.root, text="Transformation configuration", padx=10, pady=10)
        self.training_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(self.training_frame, text="Color dictionary").grid(row=0, column=0, padx=10, sticky="w")
        tk.Button(self.training_frame, text="Add color dictionary", command=self.open_color_form).grid(row=1, column=0)
        self.transformation_color_dict_text = tk.Text(self.training_frame, textvariable=self.config_data['color_dict'], height=5, width=20, state="disabled")
        self.transformation_color_dict_text.grid(row=1, column=1, padx=7, sticky="w")

        self.bin_threshold_label = tk.Label(self.training_frame, text="Threshold")
        self.bin_threshold_label.grid(row=0, column=2, padx=5, sticky="w")
        self.bin_threshold_entry = tk.Entry(self.training_frame, textvariable=self.config_data['threshold'], width=10)
        self.bin_threshold_entry.grid(row=1, column=2, padx=7, sticky="w")

        self.bin_threshold_label.grid_forget()
        self.bin_threshold_entry.grid_forget()

    def train_config_widgets(self):
        stratification_types = ConfigHandler.CONFIGURATION_VALUES["stratification_types"]
        segmentation_types = ConfigHandler.CONFIGURATION_VALUES["segmentation"]
        
        segmentation_metrics = ConfigHandler.CONFIGURATION_VALUES["segmentation_metric"]

        self.training_frame = tk.LabelFrame(self.root, text="Training configuration", padx=10, pady=10)
        self.training_frame.grid(row=10, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(self.training_frame, text="Train percentage").grid(row=0, column=0, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.config_data['split_percentages']['train'], width=10).grid(row=1, column=0, padx=5, sticky="w")
        tk.Label(self.training_frame, text="Validation percentage").grid(row=0, column=1, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.config_data['split_percentages']['valid'], width=10).grid(row=1, column=1, padx=5, sticky="w")
        tk.Label(self.training_frame, text="Test percentage").grid(row=0, column=2, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.config_data['split_percentages']['test'], width=10).grid(row=1, column=2, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Epochs").grid(row=0, column=3, padx=10, columnspan=3, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.config_data['epochs'], width=10).grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Stratification").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.training_frame, variable=self.config_data["stratification"], command=lambda: FormUtils.toggle_label_entry(self.config_data["stratification"], self.stratification_type_label, 
                                                                                                                          self.stratification_type_combobox, self.stratification_type_comment, 0, 6)).grid(row=1, column=4, padx=5, pady=5)

        tk.Label(self.training_frame, text="Cross validation").grid(row=0, column=5, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self.training_frame, variable=self.config_data["cross_validation"], command=lambda: FormUtils.toggle_label_entry(self.config_data["cross_validation"], self.num_folds_label, 
                                                                                                                          self.num_folds_entry, self.num_folds_comment, 0, 7)).grid(row=1, column=5, padx=5, pady=5)
        
        self.stratification_type_label = tk.Label(self.training_frame, text="Stratification type")
        self.stratification_type_label.grid(row=0, column=6, sticky="w")
        self.stratification_type_combobox = ttk.Combobox(self.training_frame, textvariable=self.config_data["stratification_type"], values=stratification_types, state="readonly", )
        self.stratification_type_combobox.grid(row=1, column=6, padx=5, pady=5, sticky="w")       
        self.stratification_type_comment = tk.Label(self.training_frame, text="Strategy used to group the data", fg="gray", font=("Arial", 8), wraplength=200)
        self.stratification_type_comment.grid(row=2, column=6, padx=5, pady=5, sticky="w")

        self.num_folds_label = tk.Label(self.training_frame, text="Number of folds")
        self.num_folds_label.grid(row=0, column=7, sticky="w")
        self.num_folds_entry = ttk.Entry(self.training_frame, textvariable=self.config_data["num_folds"], width=10)
        self.num_folds_entry.grid(row=1, column=7, padx=5, pady=5, sticky="w")       
        self.num_folds_comment = tk.Label(self.training_frame, text="Number of groups the data is divided into.", fg="gray", font=("Arial", 8), wraplength=200)
        self.num_folds_comment.grid(row=2, column=7, padx=5, pady=5, sticky="w")

        tk.Label(self.training_frame, text="Batch size").grid(row=8, column=0, padx=10, sticky="w")
        tk.Entry(self.training_frame, textvariable=self.config_data['batch_size'], width=10).grid(row=9, column=0, padx=5, sticky="w")

        tk.Label(self.training_frame, text="Segmentation type").grid(row=8, column=1, sticky="w")
        self.segmentation_type_combobox = ttk.Combobox(self.training_frame, textvariable=self.config_data["segmentation"], values=segmentation_types, state="readonly", width=10)
        self.segmentation_type_combobox.grid(row=9, column=1, sticky="w")
        
        tk.Label(self.training_frame, text="Segmentation metric").grid(row=8, column=2, sticky="w")
        self.segmentation_metric_combobox = ttk.Combobox(self.training_frame, textvariable=self.config_data["segmentation_metric"], values=segmentation_metrics, state="readonly", width=10)
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
        segmentation_type = self.config_data["segmentation"].get()
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
        self.config_data["models"] = selected_values  

    def open_color_form(self):
        color_form = ColorConfigForm(self.root)

        color_form.top.transient(self.root)
        color_form.top.grab_set()

        self.root.wait_window(color_form.top)
        if color_form.colors:
            self.add_colors(color_form.colors)
    
    def open_clustering_config_form(self):
        clustering_form = ClusteringConfigForm(self.root)

        clustering_form.top.transient(self.root)
        clustering_form.top.grab_set()

        self.root.wait_window(clustering_form.top)

        self.config_data.update(clustering_form.cluster_images)
        if clustering_form.cluster_images:
            self.config_data.update(clustering_form.clustering_data)

   
    def add_colors(self, colors):
        self.config_data["color_dict"] = colors

        colors_text = "\n".join([f"{color}: {class_id}" for color, class_id in colors.items()])

        self.transformation_color_dict_text.config(state="normal")  
        self.transformation_color_dict_text.delete(1.0, tk.END)  
        self.transformation_color_dict_text.insert(tk.END, colors_text)
        self.transformation_color_dict_text.config(state="disabled")

    def save_config(self):
        # Crear un nuevo top-level modal para mostrar el contenido de self.config_data
        modal = tk.Toplevel(self.root)
        modal.title("Config Data")
        modal.geometry("600x400")  # Ajusta el tamaño según sea necesario

        # Función para convertir las variables Tkinter en sus valores reales
        def get_var_value(var):
            if isinstance(var, tk.BooleanVar):
                return var.get()
            elif isinstance(var, tk.StringVar):
                return var.get()
            elif isinstance(var, tk.IntVar):
                return var.get()
            elif isinstance(var, tk.DoubleVar):
                return var.get()
            else:
                return str(var)

        # Función para convertir dict a un string formateado con sus valores
        def format_config_data(config_data):
            formatted_str = ""
            for key, value in config_data.items():
                if isinstance(value, dict):
                    formatted_str += f"{key}:\n" + format_config_data(value) + "\n"
                else:
                    formatted_str += f"{key}: {get_var_value(value)}\n"
            return formatted_str

        # Convertir self.config_data en una cadena formateada
        config_str = format_config_data(self.config_data)

        # Crear un widget Text en la ventana modal para mostrar el contenido
        text_box = tk.Text(modal, wrap=tk.WORD, width=70, height=15)
        text_box.insert(tk.END, config_str)
        text_box.config(state=tk.DISABLED)  # Hacerlo solo lectura
        text_box.pack(padx=10, pady=10)

        # Botón para cerrar el modal
        close_button = tk.Button(modal, text="Close", command=modal.destroy)
        close_button.pack(pady=10)

        