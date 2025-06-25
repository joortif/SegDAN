
from tkinter import ttk

class IntroductionFrame(ttk.Frame):

    def __init__(self, parent, controller, final_dict):
        ttk.Frame.__init__(self, parent)
        
        label_title = ttk.Label(self, text="Welcome to SegDAN configuration wizard.", font=("Arial", 18, "bold"))
        label_title.pack(pady=(20, 10), padx=10)
        label_text = ttk.Label(self, text="This wizard will guide you through the necessary configuration steps to configure the application.\n" + 
                               "SegDAN includes several options that allow the user to: ", font=("Arial", 10))
        label_text.pack(pady=10, padx=10)

        list_frame = ttk.Frame(self)
        list_frame.pack(pady=10) 

        options = [
            "Perform data analysis, creating graphs and reports to explain the content of the dataset such as number of objects for each class, bounding boxes and object areas metrics...",
            "Use image embedding models to study the similarity of images. Then the user can apply different clustering models to find patterns in the images and visualize the most similar ones.",
            "Reduce the dataset. Once the clusters are defined, they can be used to reduce the dataset by selecting a subset from each one. By selecting a representative subset, the user can apply the active learning technique.",
            "Train and save the best segmentation model. The images and labels can be used to train various segmentation models, and by selecting a performance metric, the best model will be chosen from the trained models."        ]

        for option in options:
            ttk.Label(list_frame, text=f"• {option}", font=("Arial", 10),wraplength=500).pack(pady=10)

        ttk.Label(self, text="If you already have a configuration file (YAML or JSON) it can be directly loaded into the application.").pack(pady=10)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        configuration_load_bt = ttk.Button(button_frame, text="Load configuration file", command=lambda: controller.show_frame("ConfigFileLoaderFrame"))
        configuration_load_bt.pack(side="left", padx=10)

        button = ttk.Button(button_frame, text="Next", command=lambda: controller.show_frame("GeneralConfigFrame"))
        button.pack(side="left", padx=10)


        

    