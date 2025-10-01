"""
Custom CNN architecture for deepfake detection
Lightweight model for faster inference
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger(__name__)


class CustomCNN:
    """Custom lightweight CNN for deepfake detection"""
    
    def __init__(self, 
                 input_shape: tuple = (224, 224, 3),
                 num_classes: int = 1):
        """
        Initialize custom CNN
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes (1 for binary)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self) -> tf.keras.Model:
        """
        Build custom CNN architecture
        
        Returns:
            Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Normalization
        x = layers.Rescaling(1./255)(inputs)
        
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 4
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='custom_cnn')
        
        self.model = model
        logger.info("Built custom CNN model")
        
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
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
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
        
        logger.info("Custom CNN model compiled successfully")
    
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


class AttentionCNN(CustomCNN):
    """Custom CNN with attention mechanism"""
    
    def __init__(self, 
                 input_shape: tuple = (224, 224, 3),
                 num_classes: int = 1):
        super().__init__(input_shape, num_classes)
    
    def attention_block(self, x, channels):
        """
        Attention block for feature refinement
        
        Args:
            x: Input tensor
            channels: Number of channels
            
        Returns:
            Attention-weighted tensor
        """
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
        max_pool = layers.Reshape((1, 1, channels))(max_pool)
        
        # Shared MLP
        mlp_avg = layers.Dense(channels // 8, activation='relu')(avg_pool)
        mlp_avg = layers.Dense(channels, activation='sigmoid')(mlp_avg)
        
        mlp_max = layers.Dense(channels // 8, activation='relu')(max_pool)
        mlp_max = layers.Dense(channels, activation='sigmoid')(mlp_max)
        
        channel_attention = layers.Add()([mlp_avg, mlp_max])
        
        # Apply attention
        x = layers.Multiply()([x, channel_attention])
        
        return x
    
    def build_model(self) -> tf.keras.Model:
        """
        Build custom CNN with attention
        
        Returns:
            Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Normalization
        x = layers.Rescaling(1./255)(inputs)
        
        # Block 1 with attention
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = self.attention_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2 with attention
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = self.attention_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 3 with attention
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = self.attention_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='attention_cnn')
        
        self.model = model
        logger.info("Built attention CNN model")
        
        return model


def create_custom_cnn(input_shape: tuple = (224, 224, 3),
                     with_attention: bool = False,
                     learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Helper function to create and compile custom CNN
    
    Args:
        input_shape: Input shape
        with_attention: Whether to use attention mechanism
        learning_rate: Learning rate
        
    Returns:
        Compiled model
    """
    if with_attention:
        model_builder = AttentionCNN(input_shape=input_shape)
    else:
        model_builder = CustomCNN(input_shape=input_shape)
    
    model = model_builder.build_model()
    model_builder.compile_model(learning_rate=learning_rate)
    
    return model
