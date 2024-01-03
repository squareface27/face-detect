# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 00:47:52 2024

@author: Squareface
"""

from customtkinter import *
from tkinter import filedialog
import cv2 as cv

# Chargement des classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Initialisation du GUI avec la résolution 600x500px et empêchement le redimmensionnement de la fenêtre
app = CTk()
app.title("Face Detector")
app.geometry("600x500")
app.resizable(width=False, height=False)

# Thème sombre
set_appearance_mode("dark")

# Label pour l'échelle de rétrécissement
label_scale = CTkLabel(master=app, text="Échelle de rétrécissement", font=("Arial", 16))
label_scale.place(relx=0.5, rely=0.48, anchor=CENTER)

# Slider pour l'échelle
slider_scale = CTkSlider(master=app, from_=1, to=1.2)
slider_scale.set(1.1)
slider_scale.place(relx=0.5, rely=0.52, anchor=CENTER)

# Variable pour stocker la valeur du slider
slider_scale_value = DoubleVar(value=1.1)

# Mise à jour de la variable et affichage de la valeur du slider
def update_slider_scale_value(value):
    slider_scale_value.set(value)
    label_scale_value.configure(text=f"{value:.1f}")

slider_scale.configure(command=update_slider_scale_value)

# Label pour afficher la valeur du slider
label_scale_value = CTkLabel(master=app, text=f"{slider_scale.get():.1f}")
label_scale_value.place(relx=0.67, rely=0.52, anchor=W)


# Label pour le nombre de voisins
label_neighbor = CTkLabel(master=app, text="Nombre de voisins", font=("Arial", 16))
label_neighbor.place(relx=0.5, rely=0.28, anchor=CENTER)

# Slider pour le nombre de voisins
slider_neighbor = CTkSlider(master=app, from_=1, to=20)
slider_neighbor.set(5)
slider_neighbor.place(relx=0.5, rely=0.32, anchor=CENTER)

# Variable pour stocker la valeur du slider
slider_neighbor_value = DoubleVar(value=5)

# Mise à jour de la variable et affichage de la valeur du slider
def update_slider_neighbor_value(value):
    integer_value = round(float(value))
    slider_neighbor_value.set(integer_value)
    label_neighbor_value.configure(text=f"{integer_value}")

slider_neighbor.configure(command=update_slider_neighbor_value)

# Label pour afficher la valeur du slider
label_neighbor_value = CTkLabel(master=app, text=f"{round(slider_neighbor.get())}")
label_neighbor_value.place(relx=0.67, rely=0.32, anchor=W)


# Fonction appelée lorsque le bouton est cliqué
def import_image():
    global image_path
    image_path = filedialog.askopenfilename(title="Sélectionner une image", 
                                            filetypes=(("Fichiers image", "*.png;*.jpg;*.jpeg"), ("Tous les fichiers", "*.*")))
    if image_path:
        label_image_path.configure(text=image_path)

# Bouton pour importer une image
button_import = CTkButton(master=app, text="Importer une image", command=import_image)
button_import.place(relx=0.5, rely=0.7, anchor=CENTER)

# Label pour afficher le chemin de l'image
label_image_path = CTkLabel(master=app, text="Aucune image sélectionnée")
label_image_path.place(relx=0.5, rely=0.75, anchor=CENTER)



# Fonction détection des visages et des yeux

def face_detection():
    # Exécution de la détection des visages
    
    global image_path, slider_scale, slider_neighbor
    # Chargement des images
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    scale_value = slider_scale.get()
    neighbor_value = int(slider_neighbor.get())
    
    
    # detectMultiScale(image, échelle (%), nombre de voisins)
    faces = face_cascade.detectMultiScale(gray, scale_value, neighbor_value)

    # Affichage des visages
    i = 0
    for face in faces:
        x, y, w, h = face

        # Dessin du rectangle du visage sur l'image
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # extraction du visage
        face_img = img[y:y+h, x:x+w]


        # Détection des yeux dans chaque visage
        eyes = eye_cascade.detectMultiScale(face_img, 1.05, 5)

        # Affichage des yeux dans le visage
        j = 0
        for eye in eyes:
            ex, ey, ew, eh = eye

            # Dessin du rectangle de l'oeil sur le visage
            cv.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        i += 1

    # Enregistre l'image avec le visage encadré
    cv.imwrite('outputs/image.jpg', img)
    
    # Label pour le nombre de visage détecté
    label_number_faces = CTkLabel(master=app, text=f"{i} visage(s) a / ont été détecté(s).", font=("Arial", 16))
    label_number_faces.place(relx=0.5, rely=0.15, anchor=CENTER)



# Bouton pour détecter les visages et les yeux
button_detection = CTkButton(master=app, text="Détecter les visages et les yeux", command=face_detection)
button_detection.place(relx=0.5, rely=0.85, anchor=CENTER)

app.mainloop()