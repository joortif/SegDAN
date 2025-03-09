import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ClusteringFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)

        self.config_data = config_data
        self.controller = controller

        label_title = ttk.Label(self, text="Image similarity", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
        
        self.clustering_frame = tk.Frame(self, padx=10, pady=10)
        self.clustering_frame.grid(row=1, column=0, padx=10, pady=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.row=0
        self.clustering_question_label = tk.Label(self.clustering_frame, text="Image embedding models allow us to analyze image similarity by converting images into numerical representations and using clustering algorithms like KMeans, " +
                                                  "Agglomerative Clustering, DBSCAN, and OPTICS to group similar images. You can use preset or optimized settings for these methods. If this step isn’t necessary, you can skip it to speed up the process.", wraplength=500)
        self.clustering_question_label.grid(row=self.row, column=0, padx=10, sticky="e") 

        self.row+=1

        self.more_info_button = tk.Button(self.clustering_frame, text="More information", command=self.show_more_info)
        self.more_info_button.grid(row=self.row, column=0, pady=10)

        self.row+=1

        self.clustering_question_label = tk.Label(self.clustering_frame, text="Do you wish to apply clustering model?", wraplength=500, font=("Arial", 14))
        self.clustering_question_label.grid(row=self.row, column=0, padx=10, pady=(15,10)) 

        self.row+=1
        clustering_button_frame = ttk.Frame(self.clustering_frame)
        clustering_button_frame.grid(row=self.row, column=0, columnspan=5, pady=10)  

        button_clustering_no = tk.Radiobutton(clustering_button_frame, text="No", variable=self.config_data["cluster_images"], value=False)
        button_clustering_no.grid(row=0, column=0, padx=5, pady=5)

        button_clustering_yes = tk.Radiobutton(clustering_button_frame, text="Yes", variable=self.config_data["cluster_images"], value=True)
        button_clustering_yes.grid(row=0, column=1, padx=5, pady=5)

        clustering_button_frame.grid_columnconfigure(0, weight=0)
        clustering_button_frame.grid_columnconfigure(1, weight=0)

        self.row+=1
        button_frame = ttk.Frame(self.clustering_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.clustering_frame.grid_rowconfigure(0, weight=0)
        self.clustering_frame.grid_columnconfigure(0, weight=0)
        self.clustering_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("AnalysisConfigFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def show_more_info(self):

        tk.messagebox.showinfo("Image similarity information", "Using image embedding models, we can analyze the similarity between images in a dataset in two steps.\n\n"+
                                                "First, an image embedding model (chosen by the user) converts images into numerical representations, capturing their most important features " +
                                                "while reducing unnecessary details. This allows us to compare images more effectively.\n\n" +
                                                "Next, clustering algorithms group similar images together based on these numerical representations. You can choose from different clustering " +
                                                "methods, such as KMeans, Agglomerative Clustering, DBSCAN, and OPTICS, either with preset settings or optimized through a search process.\n\n" +
                                                "If this step is not relevant to your needs, you can skip it to speed up the application process.")

    def next(self):

        if self.config_data["cluster_images"].get(): 
            self.controller.show_frame("ClusteringConfigFrame")
            return
        
        self.controller.show_frame("TrainingConfigFrame")