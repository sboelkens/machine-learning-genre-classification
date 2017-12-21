import tkinter
from tkinter.filedialog import askopenfilename
root = tkinter.Tk()
root.withdraw()
filename = askopenfilename()
print(filename)
print(root.withdraw())