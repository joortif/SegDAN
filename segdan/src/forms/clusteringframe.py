import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ClusteringFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data):
        ttk.Frame.__init__(self, parent)

        self.clustering = {"cluster_images": False}
        self.clustering_var = tk.BooleanVar(value=False)
        self.config_data = config_data
        self.controller = controller

        label_title = ttk.Label(self, text="Image similarity", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
        
        self.clustering_frame = tk.Frame(self, padx=10, pady=10)
        self.clustering_frame.grid(row=1, column=0, padx=10, pady=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.row=0
        self.clustering_question_label = tk.Label(self.clustering_frame, text="Using image embedding models, we can analyze the similarity between images in a dataset in two steps.\r\n"+
                                                "First, an image embedding model (chosen by the user) converts images into numerical representations, capturing their most important features " +
                                                "while reducing unnecessary details. This allows us to compare images more effectively.\r\n" +
                                                "Next, clustering algorithms group similar images together based on these numerical representations. You can choose from different clustering " +
                                                "methods, such as KMeans, Agglomerative Clustering, DBSCAN, and OPTICS, either with preset settings or optimized through a search process.\r\n" +
                                                "If this step is not relevant to your needs, you can skip it to speed up the application process.", wraplength=500)
        self.clustering_question_label.grid(row=self.row, column=0, padx=10, sticky="e") 

        self.row+=1

        self.clustering_question_label = tk.Label(self.clustering_frame, text="Do you wish to apply clustering model?", wraplength=500, font=("Arial", 14))
        self.clustering_question_label.grid(row=self.row, column=0, padx=10, pady=(15,10)) 

        self.row+=1
        clustering_button_frame = ttk.Frame(self.clustering_frame)
        clustering_button_frame.grid(row=self.row, column=0, columnspan=5, pady=10)  

        button_clustering_no = tk.Radiobutton(clustering_button_frame, text="No", variable=self.clustering_var, value=False)
        button_clustering_no.grid(row=0, column=0, padx=5, pady=5)

        button_clustering_yes = tk.Radiobutton(clustering_button_frame, text="Yes", variable=self.clustering_var, value=True)
        button_clustering_yes.grid(row=0, column=1, padx=5, pady=5)

        clustering_button_frame.grid_columnconfigure(0, weight=0)
        clustering_button_frame.grid_columnconfigure(1, weight=0)

        self.row+=1
        button_frame = ttk.Frame(self.clustering_frame)
        button_frame.grid(row=self.row, column=0, columnspan=5, pady=10)  

        button_back = ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("AnalysisConfigFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, padx=50, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def next(self):
       
        self.clustering["cluster_images"] = self.clustering_var.get()
        self.config_data.update(self.clustering)
        if self.clustering["cluster_images"]: 
            self.controller.show_frame("ClusteringConfigFrame")
            return
        
        self.controller.show_frame("ReductionFrame")