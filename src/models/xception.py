"""
Xception model for deepfake detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
import logging

logger = logging.getLogger(__name__)


class XceptionModel:
    """Xception-based deepfake detector"""
    
    def __init__(self, 
                 input_shape: tuple = (299, 299, 3),
                 num_classes: int = 1):
        """
        Initialize Xception model
        
        Args:
            input_shape: Input image shape (Xception expects 299x299)
            num_classes: Number of output classes (1 for binary)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, freeze_base: bool = False) -> tf.keras.Model:
        """
        Build the Xception model
        
        Args:
            freeze_base: Whether to freeze base model weights
            
        Returns:
            Compiled Keras model
        """
        # Load base model
        base_model = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Freeze base model if requested
        if freeze_base:
            base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=self.input_shape)
        
        # Preprocessing (Xception expects values in [-1, 1])
        x = tf.keras.applications.xception.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=not freeze_base)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='xception_detector')
        
        self.model = model
        logger.info("Built Xception model")
        
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.0001,
                     optimizer: str = 'adam'):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Select optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'adamw':
            opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Compile
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("Model compiled successfully")
    
    def get_model(self) -> tf.keras.Model:
        """Get the built model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.summary()
    
    def unfreeze_base(self, from_layer: str = None):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            from_layer: Unfreeze from this layer onwards (None = all)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find base model layer
        base_model = None
        for layer in self.model.layers:
            if 'xception' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            logger.warning("Base model not found")
            return
        
        # Unfreeze layers
        if from_layer is None:
            base_model.trainable = True
            logger.info("Unfroze all base model layers")
        else:
            # Unfreeze from specific layer
            set_trainable = False
            for layer in base_model.layers:
                if layer.name == from_layer:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
            
            logger.info(f"Unfroze layers from {from_layer} onwards")


def create_xception_model(input_shape: tuple = (299, 299, 3),
                         freeze_base: bool = False,
                         learning_rate: float = 0.0001) -> tf.keras.Model:
    """
    Helper function to create and compile Xception model
    
    Args:
        input_shape: Input shape
        freeze_base: Whether to freeze base layers
        learning_rate: Learning rate
        
    Returns:
        Compiled model
    """
    model_builder = XceptionModel(input_shape=input_shape)
    model = model_builder.build_model(freeze_base=freeze_base)
    model_builder.compile_model(learning_rate=learning_rate)
    
    return model
