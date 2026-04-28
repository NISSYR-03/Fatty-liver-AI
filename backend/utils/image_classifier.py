"""
backend/utils/image_classifier.py
CNN (ResNet50 transfer learning) for liver ultrasound classification.
Falls back to heuristic if TensorFlow is not installed.
"""

import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as kimg
    TF_OK = True
except ImportError:
    TF_OK = False

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "..", "..", "ml_training", "saved_models")
CNN_PATH  = os.path.join(MODEL_DIR, "cnn_liver.h5")
LABELS    = ["Normal","Mild Fatty","Moderate Fatty","Severe Fatty"]
RISK_MAP  = {"Normal":"Low","Mild Fatty":"Medium",
             "Moderate Fatty":"High","Severe Fatty":"High"}
IMG_SIZE  = (224, 224)

_model = None   # lazy singleton


def _build_model():
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(224,224,3))
    base.trainable = False
    x   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(4,   activation="softmax")(x)
    m   = tf.keras.Model(base.input, out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def _get_model():
    global _model
    if _model is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        if os.path.exists(CNN_PATH):
            _model = tf.keras.models.load_model(CNN_PATH)
        else:
            _model = _build_model()
            _model.save(CNN_PATH)
    return _model


def _heuristic(img_path):
    """Brightness-based fallback — brighter = more echogenic = fattier."""
    gray = np.array(Image.open(img_path).convert("L"), dtype=float)
    mb   = gray.mean()
    if   mb < 80:  cls = 0
    elif mb < 120: cls = 1
    elif mb < 155: cls = 2
    else:          cls = 3
    p = [0.05]*4
    p[cls] = 0.65
    r = (1-0.65)/3
    for i in range(4):
        if i != cls: p[i] = r
    return {
        "class_index":   cls,
        "class_label":   LABELS[cls],
        "risk_level":    RISK_MAP[LABELS[cls]],
        "probabilities": {LABELS[i]: round(p[i],3) for i in range(4)},
        "method":        "heuristic (TF not installed)",
    }


def classify_ultrasound(img_path: str) -> dict:
    if not TF_OK:
        return _heuristic(img_path)
    model = _get_model()
    img   = kimg.load_img(img_path, target_size=IMG_SIZE)
    x     = preprocess_input(np.expand_dims(kimg.img_to_array(img), 0))
    pred  = model.predict(x, verbose=0)[0]
    cls   = int(np.argmax(pred))
    return {
        "class_index":   cls,
        "class_label":   LABELS[cls],
        "risk_level":    RISK_MAP[LABELS[cls]],
        "probabilities": {LABELS[i]: round(float(pred[i]),3) for i in range(4)},
        "method":        "ResNet50 CNN",
    }
