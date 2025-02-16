import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from src.forms.trainingconfigform import TrainingConfigForm
from src.forms.clusteringconfigform import ClusteringConfigForm
from src.forms.colorform import ColorConfigForm
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
            "color_dict": None
        }

        self.create_widgets()

    def select_folder(self, key):
        folder = filedialog.askdirectory(title=f"Seleccionar {key.replace('_', ' ')}")
        if folder:
            self.config_data[key].set(folder)

    def create_widgets(self):
        self.general_config_widgets()
        self.additional_config_widgets()
        self.transformation_config_widgets()

        tk.Button(self.root, text="Save configuration", command=self.save_config).grid(row=14, column=0, columnspan=2, pady=10)

    def general_config_widgets(self):
        general_frame = tk.LabelFrame(self.root, text="General configuration", padx=10, pady=10)
        general_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        general_frame.grid_columnconfigure(0, weight=1)
        general_frame.grid_columnconfigure(1, weight=0)
        general_frame.grid_columnconfigure(2, weight=2)

        tk.Label(general_frame, text="Dataset path").grid(row=0, column=0, sticky="w", padx=5)
        entry_dataset = ttk.Entry(general_frame, textvariable=self.config_data["dataset_path"], width=100, state="readonly")
        entry_dataset.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        ttk.Button(general_frame, text="Select folder 📂", command=lambda: self.select_folder("dataset_path")).grid(row=1, column=2, padx=5, pady=5)

        tk.Label(general_frame, text="Output path").grid(row=2, column=0, sticky="w", padx=5)
        entry_output = ttk.Entry(general_frame, textvariable=self.config_data["output_path"], width=100, state="readonly")
        entry_output.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        ttk.Button(general_frame, text="Select folder 📂", command=lambda: self.select_folder("output_path")).grid(row=3, column=2, padx=5, pady=5)

        tk.Label(general_frame, text="Verbose").grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        ttk.Checkbutton(general_frame, variable=self.config_data["verbose"]).grid(row=5, column=0, padx=5, pady=5)
        tk.Label(general_frame, text="Enable detailed logging during execution.", fg="gray", font=("Arial", 8)).grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        tk.Label(general_frame, text="Binary").grid(row=4, column=1, sticky="ew", padx=5, pady=5)
        ttk.Checkbutton(general_frame, variable=self.config_data["binary"], command=lambda: FormUtils.toggle_label_entry(self.config_data["binary"], self.bin_threshold_label, self.bin_threshold_entry, None, 0, 2)).grid(row=5, column=1, padx=5, pady=5)
        tk.Label(general_frame, text="Set to True for binary segmentation (foreground/background).", fg="gray", font=("Arial", 8)).grid(row=6, column=1, sticky="ew", padx=5, pady=5)

        tk.Label(general_frame, text="Background").grid(row=4, column=2, sticky="ew", padx=5, pady=5)
        ttk.Entry(general_frame, textvariable=self.config_data["background"], width=5).grid(row=5, column=2, padx=5, pady=5)
        tk.Label(general_frame, text="Class label assigned to background pixels.", fg="gray", font=("Arial", 8)).grid(row=6, column=2, sticky="ew", padx=5, pady=5)

    def additional_config_widgets(self):
        analyze_frame = tk.LabelFrame(self.root, text="Additional configuration", padx=10, pady=10)
        analyze_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        analyze_frame.grid_columnconfigure(0, weight=1) 
        analyze_frame.grid_columnconfigure(2, weight=1)

        tk.Label(analyze_frame, text="Analysis").grid(row=6, column=0, sticky="s", padx=5)
        tk.Label(analyze_frame, text="Perform full scan and analysis of the dataset.\nResults will be saved in output path.", fg="gray", font=("Arial", 8)).grid(row=8, column=0)
        tk.Checkbutton(analyze_frame, variable=self.config_data["analyze"]).grid(row=7, column=0, padx=5, pady=5)

        tk.Label(analyze_frame, text="Clustering").grid(row=6, column=1, sticky="s", padx=5)
        tk.Label(analyze_frame, text="Enable image similarity study using image embeddings.", fg="gray", font=("Arial", 8)).grid(row=8, column=1)
        tk.Button(analyze_frame, text="Add clustering configuration", command=lambda: self.open_config_form(ClusteringConfigForm, 'cluster_images', self.update_clustering_data)).grid(row=7, column=1)

        tk.Label(analyze_frame, text="Training").grid(row=6, column=2, sticky="s", padx=5)
        tk.Label(analyze_frame, text="Find the best segmentation model for the dataset.", fg="gray", font=("Arial", 8)).grid(row=8, column=2)
        tk.Button(analyze_frame, text="Add training configuration", command=lambda: self.open_config_form(TrainingConfigForm, 'train_images', self.update_training_data)).grid(row=7, column=2)
        
    def transformation_config_widgets(self):
        self.training_frame = tk.LabelFrame(self.root, text="Transformation configuration", padx=10, pady=10)
        self.training_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(self.training_frame, text="Color dictionary").grid(row=0, column=0, padx=10, sticky="w")
        self.transformation_color_dict_text = tk.Text(self.training_frame, textvariable=self.config_data['color_dict'], height=5, width=20, state="disabled")
        self.transformation_color_dict_text.grid(row=1, column=0, padx=7, sticky="w")
        tk.Button(self.training_frame, text="Add color dictionary", command=lambda: self.open_config_form(ColorConfigForm, 'colors', self.add_colors)).grid(row=1, column=1)

        self.bin_threshold_label = tk.Label(self.training_frame, text="Threshold")
        self.bin_threshold_label.grid(row=0, column=2, padx=5, sticky="w")
        self.bin_threshold_entry = tk.Entry(self.training_frame, textvariable=self.config_data['threshold'], width=10)
        self.bin_threshold_entry.grid(row=1, column=2, padx=7, sticky="w")

        self.bin_threshold_label.grid_forget()
        self.bin_threshold_entry.grid_forget()

    def open_config_form(self, form_class, config_data_attr, config_update_func=None):
        form = form_class(self.root)
        form.top.transient(self.root)
        form.top.grab_set()

        self.root.wait_window(form.top)
        if hasattr(form, config_data_attr):
            data = getattr(form, config_data_attr)
            if data:
                if config_update_func:
                    config_update_func(data)
                else:
                    self.config_data.update(data)

    def update_config_data(self, form_data, data_key, additional_data_key=None):
        self.config_data[data_key] = form_data
        
        if additional_data_key and hasattr(self, additional_data_key):
            additional_data = getattr(self, additional_data_key)
            if additional_data:
                self.config_data[additional_data_key] = additional_data

    def open_color_form(self):
        color_form = ColorConfigForm(self.root)

        color_form.top.transient(self.root)
        color_form.top.grab_set()

        self.root.wait_window(color_form.top)
        if color_form.colors:
            self.add_colors(color_form.colors)
   
    def add_colors(self, colors):
        self.config_data["color_dict"] = colors

        colors_text = "\n".join([f"{color}: {class_id}" for color, class_id in colors.items()])

        self.transformation_color_dict_text.config(state="normal")  
        self.transformation_color_dict_text.delete(1.0, tk.END)  
        self.transformation_color_dict_text.insert(tk.END, colors_text)
        self.transformation_color_dict_text.config(state="disabled")

    def update_clustering_data(self, cluster_images):
        self.update_config_data(cluster_images, 'cluster_images', 'clustering_data')

    def update_training_data(self, train_images):
        self.update_config_data(train_images, 'train_images', 'training_data')

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

        