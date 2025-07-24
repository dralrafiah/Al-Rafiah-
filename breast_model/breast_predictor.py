import tensorflow as tf
from tensorflow import keras
import numpy as np
from huggingface_hub import hf_hub_download
import os

# Custom L2 Normalization Layer (needed for your models)
class L2Normalization(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

# Helper functions to recreate your model architectures
def create_binary_model():
    base_model = keras.applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
    intermediate_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    branch_outputs = []
    for output in intermediate_outputs:
        x = keras.layers.GlobalAveragePooling2D()(output)
        x = L2Normalization()(x)
        x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.BatchNormalization()(x)
        branch_outputs.append(x)
    
    fusion = keras.layers.Concatenate()(branch_outputs)
    x = keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(fusion)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.45)(x)
    final_output = keras.layers.Dense(2, activation='softmax')(x)
    
    model = keras.models.Model(inputs=base_model.input, outputs=final_output)
    return model

def create_subtype_model(num_classes):
    base_model = keras.applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    layer_names = ['conv3_block12_concat', 'conv4_block24_concat', 'conv5_block16_concat']
    intermediate_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    branch_outputs = []
    for output in intermediate_outputs:
        x = keras.layers.GlobalAveragePooling2D()(output)
        x = L2Normalization()(x)
        x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.BatchNormalization()(x)
        branch_outputs.append(x)
    
    fusion = keras.layers.Concatenate()(branch_outputs)
    x = keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(fusion)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.45)(x)
    final_output = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.models.Model(inputs=base_model.input, outputs=final_output)
    return model

# Global variables to store loaded models
_binary_model = None
_benign_model = None
_malignant_model = None

def load_models():
    global _binary_model, _benign_model, _malignant_model

    if _binary_model is None:
        try:
            print("Loading breast cancer models from Hugging Face...")

            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError("❌ Missing HF_TOKEN environment variable.")

            repo_id = "JawaherAlsharif/breast-cancer-model"

            _binary_model = create_binary_model()
            _benign_model = create_subtype_model(4)
            _malignant_model = create_subtype_model(4)

            _binary_model.load_weights(hf_hub_download(repo_id, "breast_binary_model.h5", token=token))
            _benign_model.load_weights(hf_hub_download(repo_id, "breast_benign_model.h5", token=token))
            _malignant_model.load_weights(hf_hub_download(repo_id, "breast_malignant_model.h5", token=token))

            print("✅ Breast cancer models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            _binary_model = _benign_model = _malignant_model = None

    return _binary_model, _benign_model, _malignant_model


# --- Subtype labels ---
benign_subtypes = ['Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma']
malignant_subtypes = ['Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma']

# --- Main prediction function (this is what your app will call) ---
def predict_breast_model(image):
    """
    Main prediction function for breast cancer classification
    
    Args:
        image: Preprocessed image array (shape: [1, 128, 128, 3])
    
    Returns:
        tuple: (result, subtype, confidence)
    """
    # Load models if not already loaded
    binary_model, benign_model, malignant_model = load_models()
    
    if binary_model is None or benign_model is None or malignant_model is None:
        return "Error", "Models not loaded", 0.0
        
    try:
        binary_pred = binary_model.predict(image, verbose=0)
        binary_class = np.argmax(binary_pred)

        if binary_class == 0:  # Benign
            subtype_pred = benign_model.predict(image, verbose=0)
            subtype_class = np.argmax(subtype_pred)
            subtype = benign_subtypes[subtype_class]
            return "Benign", subtype, binary_pred[0][binary_class]
        else:  # Malignant
            subtype_pred = malignant_model.predict(image, verbose=0)
            subtype_class = np.argmax(subtype_pred)
            subtype = malignant_subtypes[subtype_class]
            return "Malignant", subtype, binary_pred[0][binary_class]
            
    except Exception as e:
        return "Error", str(e), 0.0

# --- Enhanced prediction function with detailed output ---
def predict_breast_model_detailed(image):
    """
    Enhanced prediction function that returns detailed information for reports
    
    Args:
        image: Preprocessed image array (shape: [1, 128, 128, 3])
    
    Returns:
        dict: Detailed prediction results
    """
    # Load models if not already loaded
    binary_model, benign_model, malignant_model = load_models()
    
    if binary_model is None or benign_model is None or malignant_model is None:
        return {
            "error": True,
            "message": "Models not loaded",
            "classification": "Error",
            "subtype": "Models not loaded",
            "confidence": 0.0
        }
        
    try:
        # Binary prediction
        binary_pred = binary_model.predict(image, verbose=0)
        binary_class = np.argmax(binary_pred)
        binary_confidence = float(binary_pred[0][binary_class])

        if binary_class == 0:  # Benign
            subtype_pred = benign_model.predict(image, verbose=0)
            subtype_class = np.argmax(subtype_pred)
            subtype_confidence = float(subtype_pred[0][subtype_class])
            subtype = benign_subtypes[subtype_class]
            classification = "Benign"
        else:  # Malignant
            subtype_pred = malignant_model.predict(image, verbose=0)
            subtype_class = np.argmax(subtype_pred)
            subtype_confidence = float(subtype_pred[0][subtype_class])
            subtype = malignant_subtypes[subtype_class]
            classification = "Malignant"

        # Return detailed results
        return {
            "error": False,
            "classification": classification,
            "subtype": subtype,
            "confidence": binary_confidence,
            "subtype_confidence": subtype_confidence,
            "raw_binary_pred": binary_pred[0].tolist(),
            "raw_subtype_pred": subtype_pred[0].tolist(),
            "model_info": {
                "architecture": "DenseNet121 with hierarchical classification",
                "input_size": "128x128x3",
                "training_dataset": "BreaKHis",
                "feature_fusion": "Multi-scale with L2 normalization"
            }
        }
            
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "classification": "Error",
            "subtype": str(e),
            "confidence": 0.0
        }