import tkinter as tk

class FormUtils():

    @staticmethod
    def toggle_widget(widget, variable):
        if variable.get():  
            widget.grid()  
        else:
            widget.grid_remove()  

    @staticmethod
    def toggle_label_entry(event, label, entry, comment, row, column):

        if event.get():
            label.grid(row=row, column=column, padx=5)
            entry.grid(row=row+1, column=column, padx=5)
            if comment is not None:
                comment.grid(row=row+2, column=column, padx=5)
        else:
            label.grid_forget()
            entry.grid_forget()
            if comment is not None:
                comment.grid_forget()

    @staticmethod
    def save_config(data):
        config_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                config_data[key] = {sub_key: sub_value.get() if isinstance(sub_value, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)) else sub_value
                                    for sub_key, sub_value in value.items()}
            else:
                config_data[key] = value.get() if isinstance(value, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)) else value

        print(config_data)