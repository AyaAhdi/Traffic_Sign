# Guide d'Exécution - TrafficSign AI

Ce guide décrit l'installation des dépendances et les différentes manières de lancer le projet sur votre système local.

---

## 1. Prérequis Système
- **Python** (version 3.8 à 3.12 recommandée)
- Une webcam fonctionnelle (pour les tests de détection en temps réel)

---

## 2. Installation des Dépendances
Ouvrez un terminal ou PowerShell, placez-vous dans le répertoire du projet, puis installez les bibliothèques requises :

```bash
pip install tensorflow opencv-python pandas scikit-learn matplotlib seaborn flask
```

---

## 3. Options d'Exécution

### Option A : Lancer l'Interface Web interactive (Recommandé)
Cette option fournit une interface visuelle moderne accessible depuis votre navigateur pour tester la webcam et téléverser des images.

1. Démarrez le serveur Flask :
   ```bash
   cd TrafficSign
   python app.py
   ```
2. Ouvrez votre navigateur et accédez à :
   - **`http://127.0.0.1:5000`** (Local)
   - L'adresse IP locale affichée dans la console (pour tester depuis un autre appareil connecté au même réseau WiFi).

---

### Option B : Lancer le test en direct via OpenCV
Idéal pour tester rapidement la webcam dans une fenêtre native sans utiliser de navigateur Web.

1. Lancez le script de test vidéo :
   ```bash
   python Live_Video_testing.py
   ```
2. Appuyez sur la touche **`q`** du clavier dans la fenêtre vidéo pour arrêter le programme.

---

### Option C : Lancer l'Entraînement ou la Validation
Pour ré-entraîner l'intelligence artificielle ou exécuter le code pas à pas :

- **Via Jupyter Notebook** : Ouvrez VS Code ou Jupyter Notebook et lancez [training.ipynb](training.ipynb).
- **Via Console** : Lancez directement le script d'entraînement pré-configuré :
  ```bash
  python run_training.py
  ```

---

## 4. Résolution des Problèmes Fréquents

### L'application Web n'affiche pas le nom du panneau
- **Raison** : Vous avez peut-être généré un modèle temporaire de test (mock) qui a écrasé le vrai modèle.
- **Solution** : Supprimez ou renommez le fichier `traffic_sign_model.h5` généré temporairement afin que le serveur charge par défaut votre modèle pré-entraîné complet (`traffif_sign_model.h5`).

### La webcam ne s'ouvre pas
- **Raison** : Une autre application utilise déjà la webcam, ou l'index de la caméra est incorrect.
- **Solution** : Fermez Teams, Zoom ou toute autre application de messagerie. Si vous possédez plusieurs caméras, modifiez la ligne `cv2.VideoCapture(0)` dans `app.py` ou `Live_Video_testing.py` en remplaçant `0` par `1` ou `2`.
