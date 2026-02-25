import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_tifinagh_mnist(root_path):
    """Charge le dataset Tifinagh-MNIST depuis un dossier."""
    images, labels = [], []
    class_names = sorted(os.listdir(root_path), key=lambda x: int(x))

    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(root_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            try:
                img = Image.open(os.path.join(class_folder, img_file)).convert("L")
                img = img.resize((28, 28))
                images.append(np.array(img))
                labels.append(label_idx)
            except Exception:
                pass

    return np.array(images), np.array(labels), class_names


def normalize(images):
    """Normalise les pixels de [0,255] vers [0,1]."""
    return images / 255.0


def reshape_for_cnn(images):
    """Ajoute la dimension canal pour le CNN : (N,28,28) → (N,28,28,1)."""
    return images.reshape(-1, 28, 28, 1)


def encode_labels(labels, num_classes=33):
    """One-hot encoding des labels."""
    return to_categorical(labels, num_classes=num_classes)


def split_dataset(X, y, labels, test_size=0.15, val_size=0.176, random_state=42):
    """
    Split en train / validation / test.
    Défaut : 70% train, 15% val, 15% test.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(root_path, save_dir=None):
    """
    Pipeline complet de prétraitement :
    Chargement → Normalisation → Reshape → Encodage → Split.
    """
    images, labels, class_names = load_tifinagh_mnist(root_path)
    images_norm  = normalize(images)
    X            = reshape_for_cnn(images_norm)
    y            = encode_labels(labels)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, labels)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "X_val.npy"),   X_val)
        np.save(os.path.join(save_dir, "X_test.npy"),  X_test)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)
        np.save(os.path.join(save_dir, "y_val.npy"),   y_val)
        np.save(os.path.join(save_dir, "y_test.npy"),  y_test)
        print(f"Données sauvegardées dans {save_dir}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names