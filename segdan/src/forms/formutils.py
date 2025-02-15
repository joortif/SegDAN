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
            label.grid(row=row, column=column, padx=5, sticky="w")
            entry.grid(row=row+1, column=column, padx=5, sticky="w")
            if comment is not None:
                comment.grid(row=row+2, column=column, padx=5)
        else:
            label.grid_forget()
            entry.grid_forget()
            if comment is not None:
                comment.grid_forget()