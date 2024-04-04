# Face Detector

## Description

Ce projet, "Face Detector", est un outil développé en quelques heures permettant de détecter les visages et les yeux dans les images en utilisant des techniques de vision par ordinateur.



## Fonctionnalités

- Détection des visages et des yeux avec des classificateurs en cascade.
- Interface utilisateur graphique pour une utilisation facile.
- Personnalisation des paramètres de détection.

## Technologies Utilisées

- Python
- OpenCV (cv2)
- CustomTkinter

## Installation et Configuration

1. **Clonage du Répertoire** :
   Clonez ce répertoire sur votre machine.

2. **Installation des Dépendances** :
   pip install -r requirements.txt
3. **Lancement du Programme** :
   python face_detector.py
4. **Utilisation du Programme** :
- Importez une image.
- Ajustez les paramètres de détection.
- Cliquez sur "Détecter les visages et les yeux" pour voir les résultats, l'image du résultat va se créer dan le dossier outputs du projet.

## Exemples d'Utilisation

L'interface permet d'ajuster les paramètres du nombres de voisins et de l'échelle de rétrécissement afin d'améliorer la détection et éliminer les faux positifs.
![image1](https://images.squareface.fr/projets/facedetect/image.png)

Une fois l'image importée, il suffit de cliquer sur le bouton "Détecter les visages et les yeux". Le nombre de visages détectés sera indiqué, et l'image avec les visages et les yeux encadrés sera créée dans le dossier 'outputs' du projet.
![image2](https://images.squareface.fr/projets/facedetect/image2.png)

