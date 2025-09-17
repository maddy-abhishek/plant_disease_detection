import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def create_data_generators(data_dir):
    """
    Creates training and validation data generators with augmentation.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: A tuple containing train_generator and validation_generator.
    """
    # Create an ImageDataGenerator for training with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of data for validation
    )

    # Create an ImageDataGenerator for validation (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Create the training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'  # Specify this is the training set
    )

    # Create the validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'  # Specify this is the validation set
    )
    
    return train_generator, validation_generator