import tkinter as tk
from tkinter import ttk

from src.forms.reductionframe import ReductionFrame
from src.forms.clusteringconfigframe import ClusteringConfigFrame
from src.forms.analysisconfigframe import AnalysisConfigFrame
from src.forms.clusteringframe import ClusteringFrame
from src.forms.generalconfigframe import GeneralConfigFrame
from src.forms.introductionframe import IntroductionFrame

class Wizard(tk.Tk):
   def __init__(self, *args, **kwargs):
      tk.Tk.__init__(self, *args, **kwargs)

      self.title("SegDAN Wizard")
      self.geometry("1000x1000")

      self.config_data = {}

      # Create a container to hold wizard frames
      self.container = ttk.Frame(self)
      self.container.pack(side="top", fill="both", expand=True)
      self.container.grid_rowconfigure(0, weight=1)
      self.container.grid_columnconfigure(0, weight=1)

      self.frames = {}

      # Define wizard steps
      steps = [IntroductionFrame, GeneralConfigFrame, AnalysisConfigFrame, ClusteringFrame, ClusteringConfigFrame, ReductionFrame]

      for Step in steps:
         frame = Step(self.container, self, self.config_data)
         self.frames[Step.__name__] = frame
         frame.grid(row=0, column=0, sticky="nsew")

      self.show_frame("IntroductionFrame")

   def show_frame(self, cont):
      frame = self.frames[cont]
      frame.tkraise()
    
      if isinstance(frame, IntroductionFrame):
        self.geometry("600x500")
      elif isinstance(frame, GeneralConfigFrame):
        self.geometry("800x500")
      elif isinstance(frame, AnalysisConfigFrame):
        self.geometry("600x550")
      elif isinstance(frame, ClusteringFrame):
         self.geometry("600x550")
      elif isinstance(frame, ClusteringConfigFrame):
         self.geometry("800x525")
      elif isinstance(frame, ReductionFrame):
         self.geometry("600x550")