import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, colorchooser


class ColorConfigForm():
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Color dictionary configuration")

        self.colors = {}

        self.create_widgets()

    def create_widgets(self):
        self.row = 0

        tk.Label(self.top, text="Class id:").grid(row=self.row, column=0, padx=5, pady=5, sticky="w")
        self.class_id_entry = tk.Entry(self.top, width=20)
        self.class_id_entry.grid(row=self.row, column=1, padx=5, pady=5)

        tk.Button(self.top, text="Choose color", command=self.select_color).grid(row=self.row, column=2, padx=5, pady=5)

        self.color_display = tk.Label(self.top, width=4, height=2, relief="solid", bg="white")
        self.color_display.grid(row=self.row, column=3, padx=5, pady=5)
        self.color_display.grid_remove()

        self.row += 1

        tk.Button(self.top, text="Add class", command=self.add_class).grid(row=self.row, column=0, columnspan=3, pady=10)

        self.row += 1

        self.classes_list_label = tk.Label(self.top, text="Color-class dictionary:")
        self.classes_list_label.grid(row=self.row, column=0, columnspan=3, pady=5)

        self.row += 1

        self.classes_listbox = tk.Listbox(self.top, width=50, height=10, exportselection=0)
        self.classes_listbox.grid(row=self.row, column=0, columnspan=3, padx=5, pady=5)

        self.row += 1

        tk.Button(self.top, text="Save configuration", command=self.close_form).grid(row=self.row, column=0, columnspan=3, pady=10)

        self.selected_color = None

    def select_color(self):
        color_code = colorchooser.askcolor()[1]  
        if color_code:
            self.selected_color = self.hex_to_rgb(color_code)
            self.color_display.config(bg=color_code)
            self.color_display.grid()
    
    def add_class(self):
        class_id = self.class_id_entry.get()

        if not class_id or not self.selected_color:
            messagebox.showwarning("Warning", "Please, enter a valid class ID and its color.")
            return
        
        try:
            class_id = int(class_id)
        except ValueError:
            messagebox.showwarning("Warning", "Class ID must be a positive integer.")
            return

        if class_id in self.colors:
            for index in range(self.classes_listbox.size()):
                item_text = self.classes_listbox.get(index)
                if item_text.startswith(f"Class {class_id}:"):
                    self.classes_listbox.delete(index)
                    self.classes_listbox.insert(index, f"Class {class_id}: {self.selected_color}")
                    break  
        else:
            self.classes_listbox.insert(tk.END, f"Class {class_id}: {self.selected_color}")

        self.colors[class_id] = self.selected_color

        self.class_id_entry.delete(0, tk.END)
        self.selected_color = None
        

    def rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')  
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) 
        return rgb
    
    def close_form(self):
        self.top.destroy()
        if self.colors:  
            messagebox.showinfo("Color dictionary configuration", "Color dictionary saved successfully.")
        
