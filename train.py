from modules.data_loader import create_data_generators
from modules.model_builder import build_model
import json

# --- Configuration ---
DATASET_PATH = 'path/to/your/PlantVillage/dataset' # IMPORTANT: Change this path
MODEL_SAVE_PATH = 'saved_model/plant_disease_model.h5'
EPOCHS = 10

def main():
    """Main function to run the training process."""
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators(DATASET_PATH)
    
    # Get the number of classes from the generator
    num_classes = len(train_gen.class_indices)
    print(f"Found {num_classes} classes.")

    # Save class indices to a file for later use in prediction
    class_indices = train_gen.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    print("Building model...")
    model = build_model(num_classes=num_classes)
    
    model.summary()
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_steps=val_gen.samples // val_gen.batch_size
    )
    
    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()