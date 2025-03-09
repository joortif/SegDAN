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
        self.analysis_question_label = tk.Label(self.analysis_frame, text="Dataset analysis helps understand patterns, verify annotations, and detect errors. It also generates graphs on class distribution, object sizes, and bounding boxes. You can skip this step if not needed.", wraplength=500)
        self.analysis_question_label.grid(row=self.row, column=0, padx=10, sticky="e") 

        self.row+=1

        self.more_info_button = tk.Button(self.analysis_frame, text="More information", command=self.show_more_info)
        self.more_info_button.grid(row=self.row, column=0, pady=10)

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
        button_frame.grid(row=11, column=0, columnspan=5, pady=10, sticky="e")  

        self.analysis_frame.grid_rowconfigure(0, weight=0)
        self.analysis_frame.grid_columnconfigure(0, weight=0)
        self.analysis_frame.grid_columnconfigure(1, weight=0)

        button_back = ttk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("GeneralConfigFrame"))
        button_back.grid(row=0, column=0, padx=50, pady=5, sticky="w")

        button_next = ttk.Button(button_frame, text="Next", command=self.next)
        button_next.grid(row=0, column=1, pady=5, sticky="e")

        button_frame.grid_columnconfigure(0, weight=0)
        button_frame.grid_columnconfigure(1, weight=0)    

    def show_more_info(self):
        tk.messagebox.showinfo("Dataset analysis information", 
                        "Performing a dataset analysis helps to understand the patterns and identify the underlying information in the images and their labels.\n\n" +
                        "This step is useful if you don’t have in-depth knowledge of the dataset or if you simply want to learn more about it.\n\n" +
                        "It is also a good practice as it can serve as a debugging phase to ensure images are correctly annotated and there are no corrupt files.\n\n" +
                        "Additionally, the analysis creates useful graphs that display information about the objects and classes in the dataset. In particular, the graphs show how many objects of each class are present in the dataset images and calculate metrics regarding their sizes, including those of their bounding boxes and ellipses.\n\n" +
                        "If you already know this information or don’t find it useful, you can skip this process to speed up the application.")


    def next(self):
       
        self.analyze["analyze"] = self.analyze_var.get()
        self.config_data.update(self.analyze)
        self.controller.show_frame("ClusteringFrame")
        
