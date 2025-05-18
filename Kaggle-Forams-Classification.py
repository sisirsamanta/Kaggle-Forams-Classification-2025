

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import tifffile
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import glob
from scipy.ndimage import rotate

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Directories and file paths (update these to your dataset paths)
DATA_DIR = "/kaggle/input/forams-classification-2025"
LABELED_DIR = os.path.join(DATA_DIR, "volumes/volumes/labelled")
UNLABELED_DIR = os.path.join(DATA_DIR, "volumes/volumes/unlabelled")
LABELED_CSV = os.path.join(DATA_DIR, "labelled.csv")
UNLABELED_CSV = os.path.join(DATA_DIR, "unlabelled.csv")
OUTPUT_CSV = "/kaggle/working/submission.csv"

# Hyperparameters
NUM_CLASSES = 15  # 14 species + 1 unknown
BATCH_SIZE = 4
EPOCHS = 10
PSEUDO_EPOCHS = 5
CONFIDENCE_THRESHOLD = 0.9  # For pseudo-labeling and unknown class
INPUT_SHAPE = (128, 128, 128, 1)

# Load and preprocess a single volume
def load_volume(file_path):
    volume = tifffile.imread(file_path).astype(np.float32)
    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
    # Add channel dimension
    volume = volume[..., np.newaxis]
    return volume

# Data augmentation for labeled data
def augment_volume(volume):
    # Random rotation (90, 180, 270 degrees)
    angle = np.random.choice([0, 90, 180, 270])
    volume = rotate(volume, angle, axes=(0, 1), reshape=False)
    # Random flip
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=0)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=1)
    return volume

# Load labeled data
def load_labeled_data():
    # Check CSV existence
    if not os.path.exists(LABELED_CSV):
        raise FileNotFoundError(f"labelled.csv not found at {LABELED_CSV}")
    
    # Check directory existence
    if not os.path.exists(LABELED_DIR):
        raise FileNotFoundError(f"Labeled directory not found: {LABELED_DIR}")
    
    # List all .tif files
    labeled_files = glob.glob(os.path.join(LABELED_DIR, "*.tif"))
    print(f"Total .tif files in {LABELED_DIR}: {len(labeled_files)}")
    print(f"Sample filenames: {labeled_files[:5]}")
    
    # Load and inspect CSV
    df = pd.read_csv(LABELED_CSV)
    print(f"Rows in labelled.csv: {len(df)}")
    print(f"CSV columns: {list(df.columns)}")
    print("First 5 rows of labelled.csv:")
    print(df.head())
    print("Last 5 rows of labelled.csv:")
    print(df.tail())
    print("Unique labels:", df['label'].unique())
    print("Samples per label:", df['label'].value_counts().sort_index())
    
    if len(df) != 210:
        print(f"Warning: Expected 210 rows, found {len(df)}")
    
    file_paths = []
    labels = []
    missing_files = []
    
    # Try matching each ID
    for _, row in df.iterrows():
        # Extract 5-digit ID from filename (e.g., 'labelled_00000' -> '00000')
        filename = str(row['id'])  # Assuming 'id' column contains 'labelled_00000'
        if not filename.startswith('labelled_'):
            print(f"Warning: Unexpected filename format for row {row}: {filename}")
            missing_files.append(filename)
            continue
        file_id = filename[len('labelled_'):]  # Remove 'labelled_' prefix
        if not file_id.isdigit() or len(file_id) != 5:
            print(f"Warning: Invalid ID format for {filename}: {file_id}")
            missing_files.append(filename)
            continue
        
       
        file_path_pattern = os.path.join(LABELED_DIR, f"labelled_foram_{file_id}_sc_*.tif")
        #print(f"Searching for pattern: {file_path_pattern}")
        matching_files = glob.glob(file_path_pattern)
        
        if not matching_files:
            missing_files.append(file_id)
            print(f"No file found for ID {file_id}")
            continue
        if len(matching_files) > 1:
            print(f"Warning: Multiple files for ID {file_id}: {matching_files}")
        
        file_paths.append(matching_files[0])
        labels.append(row['label'])
    
    if missing_files:
        print(f"Warning: {len(missing_files)} IDs have no matching files: {missing_files[:5]}"
              f"{'...' if len(missing_files) > 5 else ''}")
    
    if not file_paths:
        raise ValueError("No valid labeled volumes found. Cannot proceed.")
    
    print(f"Successfully loaded {len(file_paths)} labeled volumes")
    
    volumes = [load_volume(fp) for fp in file_paths]
    labels = np.array(labels)
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    return np.array(volumes), labels
    
# Load unlabeled data
def load_unlabeled_data():
    df = pd.read_csv(UNLABELED_CSV)
    file_paths = []
    ids = []
    for _, row in df.iterrows():
        #print(f"Id: {str(row['id'])}")
        file_id = str(int(row['id'])).zfill(5)
        #print(f"file name : {file_id}")
        file_path = os.path.join(UNLABELED_DIR, f"foram_{file_id}_sc_*.tif")
        #print(f"Searching for pattern: {file_path}")
        file_path = glob.glob(file_path)
        if file_path:
            file_paths.append(file_path)
        ids.append(row['id'])
    
    volumes = [load_volume(fp) for fp in file_paths]
    return np.array(volumes), ids

# Build 3D CNN model
def build_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv3D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.Conv3D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.Conv3D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.GlobalAveragePooling3D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Pseudo-labeling function
def pseudo_labeling(model, unlabeled_volumes, confidence_threshold):
    preds = model.predict(unlabeled_volumes, batch_size=BATCH_SIZE)
    max_probs = np.max(preds, axis=1)
    pseudo_labels = np.argmax(preds, axis=1)
    
    # Select high-confidence predictions
    mask = max_probs >= confidence_threshold
    pseudo_volumes = unlabeled_volumes[mask]
    pseudo_y = tf.keras.utils.to_categorical(pseudo_labels[mask], num_classes=NUM_CLASSES)
    
    # Assign low-confidence predictions to "unknown" (label 14)
    pseudo_labels[max_probs < confidence_threshold] = 14
    return pseudo_volumes, pseudo_y, pseudo_labels


# Main pipeline
def main():
    # Load data
    print("Loading labeled data...")
    X_labeled, y_labeled = load_labeled_data()
    print("Loading unlabeled data...")
    X_unlabeled, unlabeled_ids = load_unlabeled_data()
    
    # Split labeled data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42
    )
    
    # Build and train initial model
    model = build_model()
    print("Training initial model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_preds_labels = np.argmax(val_preds, axis=1)
    val_true_labels = np.argmax(y_val, axis=1)
    f1 = f1_score(val_true_labels, val_preds_labels, average='weighted')
    print(f"Validation F1-score: {f1:.4f}")
    
    # Pseudo-labeling loop
    for iteration in range(2):  # Two iterations for simplicity
        print(f"Pseudo-labeling iteration {iteration + 1}...")
        pseudo_volumes, pseudo_y, all_pseudo_labels = pseudo_labeling(
            model, X_unlabeled, CONFIDENCE_THRESHOLD
        )
        
        if len(pseudo_volumes) == 0:
            print("No high-confidence pseudo-labels. Stopping.")
            break
        
        # Combine labeled and pseudo-labeled data
        X_combined = np.concatenate([X_train, pseudo_volumes], axis=0)
        y_combined = np.concatenate([y_train, pseudo_y], axis=0)
        
        # Retrain model
        model = build_model()  # Reset model to avoid overfitting
        model.fit(
            X_combined, y_combined,
            validation_data=(X_val, y_val),
            epochs=PSEUDO_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
            ]
        )
    
    # Final predictions for submission
    print("Generating final predictions...")
    final_preds = model.predict(X_unlabeled, batch_size=BATCH_SIZE)
    final_labels = np.argmax(final_preds, axis=1)
    final_probs = np.max(final_preds, axis=1)
    
    # Assign low-confidence predictions to "unknown" (label 14)
    final_labels[final_probs < CONFIDENCE_THRESHOLD] = 14
    
    # Create submission
    submission = pd.DataFrame({
        'id': unlabeled_ids,
        'label': final_labels
    })
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Submission saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

