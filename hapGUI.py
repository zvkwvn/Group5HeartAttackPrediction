import numpy as np
import customtkinter
import tkinter
import tkinter.messagebox
from tkinter import *
from trainheartattack import DNA
from normalizedinputs import m,s
from testingheartattack import max,min


customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("500x350")
app.title('Heart Attack Prediction')

text_var = StringVar()

dna = DNA()
with open('newWeights.txt', 'r') as f:  # open and read data
    for y in range(6):
        for x in range(3):
            dna.weights[y][x] = float(f.readline())

with open('newWeights2.txt', 'r') as f:  # open and read data
    for x in range(3):
        dna.weights2[x] = float(f.readline())


def button_function():
    global age, sex, chol, hrate, diab, stress
    age = float(entry1.get())
    sex = 1 if entry2.get().lower() == 'Male' else 0
    chol = float(entry3.get())
    hrate = float(entry4.get())
    diab = float(entry5.get())
    stress = float(entry6.get())
    perform_prediction()


def perform_prediction():
    norm=[age, sex, chol, hrate, diab, stress]
    norm=(norm-m)/s
    pred = dna.forward(norm)
    print(pred)
    mid=((max+min)/2)+0.05
    if pred >= mid:
        text_var.set("You might experience a heart attack")
    else:
        text_var.set("You are healthy")


frame = customtkinter.CTkFrame(master=app)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Heart Attack Prediction", font=customtkinter.CTkFont(size=20, weight="bold"))
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Age")
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="Sex")
entry2.pack(pady=12, padx=10)

entry3 = customtkinter.CTkEntry(master=frame, placeholder_text="Cholesterol")
entry3.pack(pady=12, padx=10)

entry4 = customtkinter.CTkEntry(master=frame, placeholder_text="Heart Rate")
entry4.pack(pady=12, padx=10)

entry5 = customtkinter.CTkEntry(master=frame, placeholder_text="Diabetes")
entry5.pack(pady=12, padx=10)

entry6 = customtkinter.CTkEntry(master=frame, placeholder_text="Stress Level")
entry6.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Submit", command=button_function)
button.pack(pady=12, padx=10)

label1 = customtkinter.CTkLabel(master=frame, textvariable=text_var, width=12, height=25, corner_radius=8)
label1.pack(pady=12, padx=10)

app.mainloop()
