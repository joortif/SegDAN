import tkinter as tk
from tkinter import ttk

class AnalysisConfigFrame(ttk.Frame):

    def __init__(self, parent, controller, config_data, final_dict):
        ttk.Frame.__init__(self, parent)

        self.analyze = {"analyze": False}
        self.analyze_var = tk.BooleanVar(value=False)
        self.config_data = config_data
        self.controller = controller

        label_title = ttk.Label(self, text="Dataset analysis", font=("Arial", 18, "bold"))
        label_title.grid(row=0, column=0, columnspan=5, pady=(20,10), padx=10)
        
        self.analysis_frame = tk.Frame(self, padx=10, pady=10)
        self.analysis_frame.grid(row=1, column=0, padx=10, pady=10)

        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)


        self.row=0
        self.analysis_question_label = tk.Label(self.analysis_frame, text="Performing a dataset analysis helps to understand the patterns and identifying the information underlying in the images and its labels.\r\n"+
                                                "This step can be useful if you don't have in depth information about the nature of the dataset or if you just want to know more about the dataset.\r\n"+
                                                "It is also a good practice as it can serve as a debugging phase in order to identify that the images are correctly annotated and there are not corrupt files.\r\n" +
                                                "In addition, the analysis creates useful graphs that present the information of objects and classes in the dataset. In particular, the graphs created "+
                                                "indicate how many objects from each class are in the images of the dataset as well as calculate some metrics about its sizes. This last calculation is also made " +
                                                "for their bounding boxes and ellipses.\r\n"+
                                                "If you already know this information or don't find it useful for your use case, you can skip this process and hasten the process of the application.\r\n", wraplength=500)
        self.analysis_question_label.grid(row=self.row, column=0, padx=10, sticky="e") 

        self.row+=1

        self.analysis_question_label = tk.Label(self.analysis_frame, text="Do you wish to perform a full analysis of the image dataset and its labels?", wraplength=500, font=("Arial", 14))
        self.analysis_question_label.grid(row=self.row, column=0, padx=10, pady=(15,10)) 

        self.row+=1
        analysis_button_frame = ttk.Frame(self.analysis_frame)
        analysis_button_frame.grid(row=self.row, column=0, columnspan=5, pady=10)  

        button_analysis_no = tk.Radiobutton(analysis_button_frame, text="No", variable=self.analyze_var, value=False)
        button_analysis_no.grid(row=0, column=0, padx=5, pady=5)

        button_analysis_yes = tk.Radiobutton(analysis_button_frame, text="Yes", variable=self.analyze_var, value=True)
        button_analysis_yes.grid(row=0, column=1, padx=5, pady=5)

        analysis_button_frame.grid_columnconfigure(0, weight=0)
        analysis_button_frame.grid_columnconfigure(1, weight=0)

        self.row+=1
        button_frame = ttk.Frame(self.analysis_frame)
        button_frame.grid(row=11, column=0, columnspan=5, pady=10)  

        button_back = ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("GeneralConfigFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, padx=50, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def next(self):
       
        self.analyze["analyze"] = self.analyze_var.get()
        self.config_data.update(self.analyze)
        self.controller.show_frame("ClusteringFrame")
        
