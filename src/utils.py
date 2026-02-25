import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

TIFINAGH_MAPPING = [
     'ⴰ','ⴱ','ⵛ','ⴷ','ⴻ','ⴼ','ⴳ',
    'ⵀ','ⵉ','ⵊ','ⴽ','ⵍ','ⵎ','ⵏ','ⵇ',
    'ⵔ','ⵙ','ⵜ','ⵓ','ⵡ','ⵢ','ⵅ','ⵣ','ⵃ','ⵚ','ⴹ','ⵟ',
    'ⵄ','ⵖ','ⵥ','ⴳⵯ','ⴽⵯ','ⵕ',
]


def plot_training_curves(history, save_path=None):
    """Trace les courbes accuracy et loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(history.history[metric],     label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Validation')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True)
    plt.suptitle("Courbes d'entraînement — CNN Tifinagh", fontsize=13)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Affiche la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TIFINAGH_MAPPING, yticklabels=TIFINAGH_MAPPING)
    plt.title("Matrice de Confusion — CNN Tifinagh MNIST", fontsize=14)
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()


def print_classification_report(y_true, y_pred):
    """Affiche le rapport de classification complet."""
    print(classification_report(y_true, y_pred, target_names=TIFINAGH_MAPPING))


def pipeline_ocr_correction(images_sequence, model, dictionnaire, correction_module):
    """
    Pipeline complet : Image → CNN → Mot reconnu → Correction NLP → Mot final.
    """
    predictions    = model.predict(images_sequence, verbose=0)
    indices_predits = np.argmax(predictions, axis=1)
    mot_reconnu    = ''.join([TIFINAGH_MAPPING[i] for i in indices_predits])
    if mot_reconnu in dictionnaire:
        return mot_reconnu, mot_reconnu, False
    mot_final, _   = correction_module(mot_reconnu, dictionnaire)
    return mot_reconnu, mot_final, True