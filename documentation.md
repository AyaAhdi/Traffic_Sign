# Documentation Technique - TrafficSign AI

Ce document présente l'architecture technique, les détails de modélisation et les étapes nécessaires pour reproduire ce projet de détection et classification en temps réel de panneaux de signalisation routière.

---

## 1. Description du Jeu de Données
Le projet utilise le jeu de données **GTSRB** (German Traffic Sign Recognition Benchmark) contenant :
- **43 classes** différentes (limites de vitesse, signalisations de danger, interdictions, etc.).
- Plus de **39 000 images** d'entraînement.
- Résolution variable, standardisée à **32x32 pixels** pendant le traitement.

---

## 2. Prétraitement des Images
Pour améliorer l'apprentissage du modèle face aux variations de luminosité et de contraste :
1. **Conversion en Niveaux de Gris** : Réduction de la dimensionnalité (couleurs non nécessaires pour la géométrie du symbole).
2. **Égalisation d'Histogramme** (`cv2.equalizeHist`) : Amélioration du contraste global pour compenser les ombres et surbrillances.
3. **Normalisation** : Division des valeurs de pixels par 255.0 pour les ramener dans l'intervalle `[0, 1]`.
4. **Augmentation des Données** (`ImageDataGenerator`) : Application de rotations, translations, cisaillements et zooms aléatoires pour éviter le surapprentissage.

---

## 3. Architecture du Modèle CNN
Le modèle est un réseau de neurones convolutifs (CNN) construit avec Keras :

| Couche | Type | Filtres / Neurones | Taille Noyau | Fonction d'activation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Convolution 2D | 60 | 5x5 | ReLU |
| 2 | Convolution 2D | 60 | 5x5 | ReLU |
| 3 | Max Pooling 2D | - | 2x2 | - |
| 4 | Convolution 2D | 30 | 3x3 | ReLU |
| 5 | Convolution 2D | 30 | 3x3 | ReLU |
| 6 | Max Pooling 2D | - | 2x2 | - |
| 7 | Dropout | Taux : 0.5 | - | - |
| 8 | Aplatissement | - | - | - |
| 9 | Dense (Fully Conn.)| 500 | - | ReLU |
| 10 | Dropout | Taux : 0.5 | - | - |
| 11 | Sortie Dense | 43 | - | Softmax (Probabilités) |

- **Optimiseur** : Adam avec un taux d'apprentissage (learning rate) initial de `0.001`.
- **Fonction de perte (Loss)** : `categorical_crossentropy` (car classification multi-classe exclusive).

---

## 4. Étapes pour Reproduire le Projet

### Étape 1 : Préparation de l'Environnement et des Fichiers
1. Téléchargez le jeu de données GTSRB sur [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Organisez les fichiers comme suit :
   ```text
   TrafficSign-ai/
   ├── Dataset/
   │   └── Dataset/
   │       ├── Train/
   │       │   ├── 0/ (Images classe 0)
   │       │   └── ... (jusqu'à 42)
   │       └── Train.csv
   └── TrafficSign/
       ├── run_training.py
       └── app.py
   ```

### Étape 2 : Entraînement du Modèle
Exécutez le script d'entraînement pour charger les images, appliquer les prétraitements, et lancer l'apprentissage :
```bash
python run_training.py
```
*Le script génère les graphiques de perte/précision (`confusionmatrix_graph.png`), la matrice de confusion (`confusionmatrix.png`), le modèle Keras (`traffic_sign_model.h5`) et le modèle compressé TensorFlow Lite (`converted_model.tflite`).*

### Étape 3 : Intégration du Modèle
Le fichier `.h5` généré est lu par `app.py` pour réaliser les prédictions en direct sur le flux vidéo ou sur des fichiers image téléversés.
