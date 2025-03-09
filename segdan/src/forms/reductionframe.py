import tkinter as tk
from tkinter import ttk


class ReductionFrame(ttk.Frame):
    
    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)

        self.reduction_var = tk.BooleanVar(value=False)
        self.config_data = config_data
        self.controller = controller

        label_title = ttk.Label(self, text="Dataset reduction", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
        
        self.reduction_frame = tk.Frame(self, padx=10, pady=10)
        self.reduction_frame.grid(row=1, column=0, padx=10, pady=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)


        self.row=0
        self.reduction_question_label = tk.Label(self.reduction_frame, text="After grouping images into clusters, we can simplify the dataset by focusing on representative images closest to the cluster center. This reduces redundancy and helps manage large datasets, lowering computational costs and speeding up model training." +
                                                 "It can also guide image selection in Active Learning. In some cases, selecting images farther from the center may be useful to capture outliers or anomalies. Alternatively, randomly selecting images from each cluster combines both strategies for a balanced approach."
                                                                    , wraplength=500)
        self.reduction_question_label.grid(row=self.row, column=0, padx=10, sticky="e") 

        self.row+=1

        self.more_info_reduction = tk.Button(self.reduction_frame, text="More information", command=self.show_more_info)
        self.more_info_reduction.grid(row=self.row, column=0, pady=10)

        self.row+=1

        self.reduction_question_label = tk.Label(self.reduction_frame, text="Do you wish to apply a reduction process to the dataset?", wraplength=500, font=("Arial", 14))
        self.reduction_question_label.grid(row=self.row, column=0, padx=10, pady=(15,10)) 

        self.row+=1
        reduction_button_frame = ttk.Frame(self.reduction_frame)
        reduction_button_frame.grid(row=self.row, column=0, columnspan=5, pady=10)  

        button_reduction_no = tk.Radiobutton(reduction_button_frame, text="No", variable=self.config_data["reduce_images"], value=False)
        button_reduction_no.grid(row=0, column=0, padx=5, pady=5)

        button_reduction_yes = tk.Radiobutton(reduction_button_frame, text="Yes", variable=self.config_data["reduce_images"], value=True)
        button_reduction_yes.grid(row=0, column=1, padx=5, pady=5)

        reduction_button_frame.grid_columnconfigure(0, weight=0)
        reduction_button_frame.grid_columnconfigure(1, weight=0)

        self.row+=1
        button_frame = ttk.Frame(self.reduction_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.reduction_frame.grid_rowconfigure(0, weight=0)
        self.reduction_frame.grid_columnconfigure(0, weight=0)
        self.reduction_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=self.back)
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def show_more_info(self):
        tk.messagebox.showinfo("Dataset reduction information", "Once the images have been grouped into different clusters based on their similarity, we can apply a reduction process. This process helps to simplify the dataset, making it easier to work with, especially when dealing with large amounts of data.\n\n" +
                                                                    "One common approach is to focus on the images that are closest to the center of each cluster. These images are typically the most representative of their group, so by selecting just this subset, we can reduce redundant information. " +
                                                                    "This is particularly useful for large datasets that contain many similar images, as it helps to reduce the overall size of the dataset. As a result, this strategy can significantly lower the computational costs when training the segmentation models, " + 
                                                                    "making the process faster and more efficient. Additionally, it can also help in deciding which images to label first in an Active Learning scenario, where labeling is done strategically.\n\n" +
                                                                    "In some cases, however, the images that are farther away from the center of the clusters may be valuable. For example, selecting the most diverse images in the dataset might be helpful for identifying anomalies or outliers." + 
                                                                    "This approach ensures that the dataset includes a variety of different examples, which can be important for detecting unusual patterns.\n\n" +
                                                                    "Finally, another option is to randomly select images from each cluster, combining both of the previous strategies. This can offer a balanced approach, depending on the specific needs of your analysis.")

    def back(self):
        if not self.config_data["cluster_images"]:
            self.controller.show_frame("ClusteringFrame")
            return

        self.controller.show_frame("ClusteringConfigFrame")

    def next(self):
       
        if self.config_data["reduce_images"].get():
            self.controller.show_frame("ReductionConfigFrame")
            return
    
        self.controller.show_frame("TrainingConfigFrame")