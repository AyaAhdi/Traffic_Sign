# Détection & Classification des Panneaux de Signalisation en Temps Réel

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-WebFramework-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)

Une application interactive de Deep Learning pour détecter et classifier les panneaux de signalisation routière en temps réel. Construite avec des réseaux de neurones convolutifs (CNN) sous TensorFlow/Keras et déployée via une interface web Flask moderne et réactive.






---

## 🚀 Fonctionnalités Principales

- **Détection en Temps Réel** : Analyse le flux vidéo de votre webcam en direct, entoure automatiquement les panneaux détectés et les classifie.
- **Analyseur d'Images** : Une superbe interface web par glisser-déposer (Drag & Drop) permettant de téléverser des images de panneaux et d'obtenir des prédictions instantanées avec barres de confiance.
- **Modèle de Haute Précision** : Entraîné sur le jeu de données allemand GTSRB (German Traffic Sign Recognition Benchmark) couvrant 43 classes.
- **Conversion TFLite** : Pipeline de conversion pour exporter le modèle au format TensorFlow Lite afin de l'intégrer sur des appareils embarqués ou mobiles.

---

## 🛠️ Démarrage Rapide

### 1. Installation
Installez l'ensemble des bibliothèques nécessaires :
```bash
pip install tensorflow opencv-python pandas scikit-learn matplotlib seaborn flask
```

### 2. Lancer l'Interface Web
Démarrez le serveur Flask local :
```bash
python app.py
```
Ouvrez l'adresse **`http://127.0.0.1:5000`** dans votre navigateur.

### 3. Lancer la Détection OpenCV Native
Démarrez l'outil autonome de test par caméra :
```bash
python Live_Video_testing.py
```
*Appuyez sur `q` pour quitter la fenêtre.*

---

## 🧠 Architecture du Modèle
Le modèle CNN traite des images converties en niveaux de gris et normalisées :
1. **Égalisation d'histogramme** pour compenser les variations de luminosité.
2. **Double couche de Conv2D** (60 filtres, noyau 5x5) + MaxPooling.
3. **Double couche de Conv2D** (30 filtres, noyau 3x3) + MaxPooling + Dropout.
4. **Couche Dense** (500 unités) + Dropout + Sortie Softmax (43 classes).

Consultez la [documentation.md](documentation.md) pour obtenir les détails complets de réplication et d'entraînement.

---

## 📁 Structure du Projet

```text
├── templates/
│   └── index.html             # Interface Web moderne en mode sombre (Glassmorphism)
├── app.py                     # Serveur Flask et logique de traitement backend
├── Live_Video_testing.py      # Script autonome OpenCV pour la détection webcam native
├── run_training.py            # Pipeline d'entraînement et de conversion local
├── training.ipynb             # Notebook Jupyter pour l'entraînement pas à pas (Colab/Local)
├── traffif_sign_model.h5      # Poids du modèle CNN pré-entraîné
├── documentation.md           # Guide de réplication technique étape par étape (FR)
├── guide.md                   # Manuel d'installation et d'exécution (FR)
└── README.md                  # Présentation du projet (Prêt pour votre GitHub public)
```

---

## 📝 Références & Dataset
- **Jeu de données** : [Kaggle GTSRB German Traffic Sign Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)


