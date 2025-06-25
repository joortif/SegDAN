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
    def convert_value(value):
            if isinstance(value, tk.BooleanVar):
                return value.get()
            if isinstance(value, tk.StringVar):
                val = value.get()
                try:
                    return int(val)
                except ValueError:
                    pass
                try:
                    return float(val)
                except ValueError:
                    return val.lower()
            else:
                return value

    @staticmethod
    def save_config(data):
        config_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                config_data[key] = FormUtils.save_config(value)
            else:
                config_data[key] = FormUtils.convert_value(value) if isinstance(value, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)) else value


        return config_data

    @staticmethod
    def center_window(child, parent):
        parent.update_idletasks()  

        width = child.top.winfo_width()  
        height = child.top.winfo_height()  

        if width == 1 or height == 1:  
            width = child.top.winfo_reqwidth()
            height = child.top.winfo_reqheight()

        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (width // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (height // 2)

        child.top.geometry(f'+{x}+{y}')
