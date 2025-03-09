import tkinter as tk
from tkinter import ttk, messagebox

from src.forms.configfileloaderframe import ConfigFileLoaderFrame
from src.forms.trainingconfigframe import TrainingConfigFrame
from src.forms.reductionframe import ReductionFrame
from src.forms.clusteringconfigframe import ClusteringConfigFrame
from src.forms.analysisconfigframe import AnalysisConfigFrame
from src.forms.clusteringframe import ClusteringFrame
from src.forms.generalconfigframe import GeneralConfigFrame
from src.forms.introductionframe import IntroductionFrame
from src.forms.reductionconfigframe import ReductionConfigFrame

class Wizard(tk.Tk):
   def __init__(self, *args, **kwargs):
      tk.Tk.__init__(self, *args, **kwargs)

      self.title("SegDAN Wizard")
      self.window_width = 800
      self.window_height = 750

      screen_width = self.winfo_screenwidth()
      screen_height = self.winfo_screenheight()

      x = (screen_width // 2) - (self.window_width // 2)
      y = (screen_height // 2) - (self.window_height // 2)

      self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

      self.resizable(False, False)

      self.config_data = {
         "cluster_images": tk.BooleanVar(value=False),
         "reduce_images": tk.BooleanVar(value=False),
         
      }

      self.container = ttk.Frame(self)
      self.container.grid(row=1, column=0, sticky="nsew")
      self.container.grid_rowconfigure(0, weight=1)
      self.container.grid_columnconfigure(0, weight=1)
      self.final_dict = {}
      
      self.steps = [
         "Introduction", 
         "General configuration", 
         "Analysis", 
         "Clustering", "Clustering configuration", 
         "Reduction", "Reduction configuration", 
         "Training",
         "ConfigFileLoader"
      ]

      self.frames = {}  
      self.current_step = 0

      self.create_breadcrumb()

      steps_classes = [IntroductionFrame, GeneralConfigFrame, AnalysisConfigFrame, ClusteringFrame, ClusteringConfigFrame, ReductionFrame, ReductionConfigFrame, TrainingConfigFrame, ConfigFileLoaderFrame]

      for Step in steps_classes:
         frame = Step(self.container, self, self.config_data, self.final_dict)
         self.frames[Step.__name__] = frame
         frame.grid(row=0, column=0, sticky="nsew")

      self.show_frame("IntroductionFrame")

      self.protocol("WM_DELETE_WINDOW", self.on_close)

   def create_breadcrumb(self):
      self.breadcrumb_frame = ttk.Frame(self)
      self.breadcrumb_frame.grid(row=0, column=0, padx=10, pady=(5,20))

      self.grid_rowconfigure(0, weight=0)  
      self.grid_rowconfigure(1, weight=1)  
      self.grid_columnconfigure(0, weight=1)  

      self.breadcrumb_labels = []
      for step in self.steps:
         if "configfileloader" in step.lower():
               continue
         
         if "configuration" in step.lower():
            step = step.replace("configuration", "\nconfiguration")
         label = ttk.Label(self.breadcrumb_frame, text=step, padding=(5, 2), justify="center")
         label.pack(side="left", padx=5)
         self.breadcrumb_labels.append(label)

      self.update_breadcrumb()

   def update_breadcrumb(self):
      for i, label in enumerate(self.breadcrumb_labels):
         if i == self.current_step:
            label.config(foreground="blue", font=("Arial", 10, "bold"))
         else:
            label.config(foreground="gray", font=("Arial", 10))

   def on_close(self):
      result = messagebox.askyesno("Close without saving?", "You haven't saved your configuration. Do you really want to exit?")
      if result:  
         self.quit()

   def show_frame(self, cont):
      #print(f"Switching to {cont}, current config_data: {self.config_data}")

      for i, step_name in enumerate(self.frames.keys()):
         if step_name == cont:
            self.current_step = i
            break

      frame = self.frames[cont]
      frame.tkraise()
      self.update_breadcrumb()