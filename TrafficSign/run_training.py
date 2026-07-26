import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Configurer les chemins d'accès locaux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Dataset", "Dataset"))
META_DIR = os.path.join(DATASET_DIR, "Meta")
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
LABEL_FILE = os.path.join(DATASET_DIR, "Train.csv")

print(f"Base Directory: {BASE_DIR}")
print(f"Dataset Directory: {DATASET_DIR}")

# 2. Générer un jeu de données fictif si le dossier Train est manquant
if not os.path.exists(TRAIN_DIR):
    print("\n[!] Le dossier 'Train' d'images d'entraînement est introuvable.")
    print("[*] Génération automatique d'un dossier 'Train' fictif à partir de 'Meta' pour valider le code...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    # Parcourir les 43 classes
    for i in range(43):
        class_meta_path = os.path.join(META_DIR, f"{i}.png")
        if os.path.exists(class_meta_path):
            img = cv2.imread(class_meta_path)
            class_train_dir = os.path.join(TRAIN_DIR, str(i))
            os.makedirs(class_train_dir, exist_ok=True)
            
            # Générer 10 images transformées pour chaque classe
            for j in range(10):
                # Légères distorsions/variations
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                
                # Bruit de brillance aléatoire
                value = np.random.randint(-30, 30)
                if value >= 0:
                    lim = 255 - value
                    rotated[rotated > lim] = 255
                    rotated[rotated <= lim] += value
                else:
                    lim = -value
                    rotated[rotated < lim] = 0
                    rotated[rotated >= lim] -= lim
                
                output_path = os.path.join(class_train_dir, f"mock_{j}.png")
                cv2.imwrite(output_path, rotated)
        else:
            print(f"[Warning] Image exemple manquante pour la classe {i}: {class_meta_path}")
    print("[*] Génération terminée. Dossier 'Train' de test prêt.")

# 3. Chargement des images et des étiquettes (Labels)
count = 0
images = []
labels = []
classes_list = [str(x) for x in range(43)]
noOfClasses = len(classes_list)

print("\n[*] Chargement des images...")
for x in range(noOfClasses):
    class_path = os.path.join(TRAIN_DIR, str(x))
    if os.path.exists(class_path):
        imglist = os.listdir(class_path)
        for y in imglist:
            img = cv2.imread(os.path.join(class_path, y))
            if img is not None:
                img = cv2.resize(img, (32, 32))
                images.append(img)
                labels.append(x)
        print(x, end=" ")
print("\n[*] Chargement terminé.")

images = np.array(images)
classNo = np.array(labels)

if len(images) == 0:
    raise ValueError("Aucune image n'a pu être chargée. Vérifiez les chemins d'accès.")

# Séparation des données en Train/Validation/Test (80% / 10% / 10%)
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2, random_state=42)
Y_tests = y_test
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("\nFormes des données (Data Shapes) :")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Validation: {X_validation.shape}, {y_validation.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# 4. Prétraitement des images
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalisation
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Redimensionnement des dimensions pour le canal de gris (Gray channel 1)
X_train = X_train.reshape(-1, 32, 32, 1)
X_validation = X_validation.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

# Augmentation de données
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

# Encodage One-hot des labels
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# 5. Définition du modèle CNN
def seq_Model():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    
    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = seq_Model()
print(model.summary())

# 6. Entraînement du modèle (paramètres réduits pour validation rapide)
batch_size_val = 32
epochs_val = 5  # Réduit à 5 époques pour un test rapide en local
steps_per_epoch_val = len(X_train) // batch_size_val

print(f"\n[*] Lancement de l'entraînement pour {epochs_val} époques...")
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=max(1, steps_per_epoch_val),
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# 7. Tracé des courbes de performance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entraînement')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Courbe de Perte (Loss)')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entraînement')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision (Accuracy)')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusionmatrix_graph.png'), dpi=150)
plt.close()

# 8. Évaluation sur l'ensemble de test
score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# 9. Matrice de Confusion
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(Y_tests, y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=False, cmap='Blues', fmt='g')  # annot=False pour éviter de surcharger avec 43 classes
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig(os.path.join(BASE_DIR, 'confusionmatrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# 10. Sauvegarde et conversion du modèle
model_path = os.path.join(BASE_DIR, 'traffic_sign_model.h5')
model.save(model_path)
print(f"\n[+] Modèle sauvegardé avec succès sous : {model_path}")

print("[*] Conversion du modèle au format TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(BASE_DIR, 'converted_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"[+] Modèle TFLite sauvegardé avec succès sous : {tflite_path}")
print("\n[OK] Execution terminee avec succes !")
