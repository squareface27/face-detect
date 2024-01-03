# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 00:47:52 2024

@author: Squareface
"""

import cv2 as cv

# Chargement des classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Chargement des images
img = cv.imread('dicaprio_sample.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Exécution de la détection des visages
# detectMultiScale(image, échelle (%), nombre de voisins)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Affichage des visages
i = 0
for face in faces:
    x, y, w, h = face
    
    # Dessin du rectangle du visage sur l'image
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # extraction du visage
    face_img = img[y:y+h, x:x+w]
    
    # Enregistrement de chaque visage extrait
    cv.imwrite('face{}.jpg'.format(i), face_img)
    
    # Détection des yeux dans chaque visage
    eyes = eye_cascade.detectMultiScale(face_img, 1.05, 5)
    
    # Affichage des yeux dans le visage
    j = 0
    for eye in eyes:
        ex, ey, ew, eh = eye
        
        # Dessin du rectangle de l'oeil sur le visage
        cv.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
        
        # extraction de l'oeil
        eye_img = face_img[ey:ey+eh, ex:ex+ew]
        
        # Enregistrement de chaque oeil extrait
        cv.imwrite('face{}_eye{}.jpg'.format(i, j), eye_img)
        j += 1
    
    i += 1
    
# Enregistre l'image avec le visage encadré
cv.imwrite('image.jpg', img)
