"""
EfficientNet model for deepfake detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
import logging

logger = logging.getLogger(__name__)


class EfficientNetModel:
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, 
                 variant: str = 'b0',
                 input_shape: tuple = (224, 224, 3),
                 num_classes: int = 1):
        """
        Initialize EfficientNet model
        
        Args:
            variant: EfficientNet variant ('b0', 'b4', etc.)
            input_shape: Input image shape
            num_classes: Number of output classes (1 for binary)
        """
        self.variant = variant
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, freeze_base: bool = False) -> tf.keras.Model:
        """
        Build the EfficientNet model
        
        Args:
            freeze_base: Whether to freeze base model weights
            
        Returns:
            Compiled Keras model
        """
        # Select base model
        if self.variant == 'b0':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.variant == 'b4':
            base_model = EfficientNetB4(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {self.variant}")
        
        # Freeze base model if requested
        if freeze_base:
            base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=self.input_shape)
        
        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)
        
        # Base model
        x = base_model(x, training=not freeze_base)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=f'efficientnet_{self.variant}')
        
        self.model = model
        logger.info(f"Built EfficientNet-{self.variant.upper()} model")
        
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
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
    
    def unfreeze_base(self, num_layers: int = None):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find base model layer
        base_model = None
        for layer in self.model.layers:
            if 'efficientnet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            logger.warning("Base model not found")
            return
        
        # Unfreeze layers
        if num_layers is None:
            base_model.trainable = True
            logger.info("Unfroze all base model layers")
        else:
            # Freeze all first
            base_model.trainable = False
            
            # Unfreeze last N layers
            for layer in base_model.layers[-num_layers:]:
                layer.trainable = True
            
            logger.info(f"Unfroze last {num_layers} layers of base model")


def create_efficientnet_model(variant: str = 'b0',
                              input_shape: tuple = (224, 224, 3),
                              freeze_base: bool = False,
                              learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Helper function to create and compile EfficientNet model
    
    Args:
        variant: EfficientNet variant
        input_shape: Input shape
        freeze_base: Whether to freeze base layers
        learning_rate: Learning rate
        
    Returns:
        Compiled model
    """
    model_builder = EfficientNetModel(variant=variant, input_shape=input_shape)
    model = model_builder.build_model(freeze_base=freeze_base)
    model_builder.compile_model(learning_rate=learning_rate)
    
    return model
