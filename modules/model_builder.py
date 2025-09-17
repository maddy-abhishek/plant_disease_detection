from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

def build_model(num_classes):
    """
    Builds a transfer learning model using VGG16.

    Args:
        num_classes (int): The number of output classes for the final layer.

    Returns:
        tensorflow.keras.Model: The compiled Keras model.
    """
    # Load the VGG16 model, pre-trained on ImageNet, without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the base model so they are not trained
    for layer in base_model.layers:
        layer.trainable = False

    # Add our custom layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model