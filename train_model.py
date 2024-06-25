import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import AlignIO
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Define file paths (replace with your actual file paths)
enzyme1_fasta = "CrtB.fasta"
enzyme2_fasta = "crtM.fasta"
alignment_file = "Align.aln"

# Hydrophobicity index and molecular weight tables for amino acids
hydrophobicity_index = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

molecular_weight = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15, 'Q': 146.14, 'E': 147.13, 
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 
    'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

# Polarity (P: polar, N: non-polar)
polarity = {'A': 'N', 'R': 'P', 'N': 'P', 'D': 'P', 'C': 'N', 'Q': 'P', 'E': 'P', 
             'G': 'N', 'H': 'P', 'I': 'N', 'L': 'N', 'K': 'P', 'M': 'N', 'F': 'N', 
             'P': 'N', 'S': 'P', 'T': 'P', 'W': 'N', 'Y': 'P', 'V': 'N'}

# Aromaticity (Y: Yes, N: No)
aromaticity = {'A': 'N', 'R': 'N', 'N': 'N', 'D': 'N', 'C': 'N', 'Q': 'N', 'E': 'N', 
               'G': 'N', 'H': 'N', 'I': 'N', 'L': 'N', 'K': 'N', 'M': 'N', 'F': 'Y', 
               'P': 'N', 'S': 'N', 'T': 'N', 'W': 'Y', 'Y': 'Y', 'V': 'N'}

# Acidity/Basicity (use pKa values)
acidity_basicity = {'A': 4.06, 'R': 12.48, 'N': 10.70, 'D': 3.86, 'C': 8.33, 'Q': 10.53, 'E': 4.26, 
                    'G': 5.97, 'H': 6.00, 'I': 6.02, 'L': 6.00, 'K': 10.54, 'M': 10.07, 'F': 3.90, 
                    'P': 10.47, 'S': 9.15, 'T': 9.10, 'W': 3.82, 'Y': 10.09, 'V': 7.39}

# Define charge information
charge_info = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 
               'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 
               'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# Read protein sequences in FASTA format
def read_fasta(file_path):
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
        return Seq("".join(lines[1:]))

seq1 = read_fasta(enzyme1_fasta)
seq2 = read_fasta(enzyme2_fasta)

# Perform pairwise alignment using Clustal Omega (command line tool)
# !clustalo -i {enzyme1_fasta} -i {enzyme2_fasta} -o {alignment_file} --outfmt=clu

# Read the alignment file
alignment = AlignIO.read(alignment_file, "clustal")

def assign_label(sequence_name):
    if sequence_name.startswith("B2"):
        return "Function1"
    elif sequence_name.startswith("M2"):
        return "Function2"
    else:
        raise ValueError("Unknown sequence name")

# Extract sequence names from the alignment object
sequence_names = [record.id for record in alignment]

def encode_sequence(sequence):
    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
    one_hot = np.zeros((len(sequence), len(amino_acid_order)))
    hydrophobicity = np.zeros(len(sequence))
    molecular_weight_array = np.zeros(len(sequence))
    charge_array = np.zeros(len(sequence))
    polarity_array = np.zeros(len(sequence))
    aromaticity_array = np.zeros(len(sequence))
    acidity_basicity_array = np.zeros(len(sequence))
    
    for i, aa in enumerate(sequence):
        if aa in amino_acid_order:
            one_hot[i, amino_acid_order.index(aa)] = 1
            hydrophobicity[i] = hydrophobicity_index.get(aa, 0)
            molecular_weight_array[i] = molecular_weight.get(aa, 0)
            charge_array[i] = charge_info.get(aa, 0)
            polarity_array[i] = polarity.get(aa, 'N') == 'P'
            aromaticity_array[i] = aromaticity.get(aa, 'N') == 'Y'
            acidity_basicity_array[i] = acidity_basicity.get(aa, 0)
    
    features = np.concatenate([
        one_hot, 
        hydrophobicity[:, np.newaxis], 
        molecular_weight_array[:, np.newaxis], 
        charge_array[:, np.newaxis], 
        polarity_array[:, np.newaxis], 
        aromaticity_array[:, np.newaxis], 
        acidity_basicity_array[:, np.newaxis]
    ], axis=1)
    
    return features

# Prepare data
X_encoded = np.array([encode_sequence(str(record.seq)) for record in alignment])
y = np.array([assign_label(name) for name in sequence_names])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Ensure X_train, X_test, y_train, y_test have correct data types
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = np.array(y_train)  # Ensure y_train is numpy array
y_test = np.array(y_test)    # Ensure y_test is numpy array

# Assuming X_train, X_test, y_train, y_test are already loaded and preprocessed

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Verify dtype after conversion
print("y_train dtype:", y_train.dtype)
print("y_test dtype:", y_test.dtype)

# Reshape data assuming it's 3D (samples, time_steps, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Verify the shapes after reshaping
print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_test_reshaped shape:", X_test_reshaped.shape)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)).reshape(X_train_reshaped.shape)
X_test_scaled = scaler.transform(X_test_reshaped.reshape(X_test_reshaped.shape[0], -1)).reshape(X_test_reshaped.shape)

# Verify the shapes and dtype after scaling
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("X_train_scaled dtype:", X_train_scaled.dtype)
print("X_test_scaled dtype:", X_test_scaled.dtype)

from tensorflow.keras.layers import Input

# Define input shape explicitly
input_shape = X_train_scaled.shape[1:]  # Shape should be (features, 1)

# Initialize Sequential model
model = Sequential()

# Add the first Conv2D layer with input shape
model.add(Input(shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Continue with the rest of your model architecture
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))  # Adjust for multi-class if needed

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Adjust for multi-class if needed
              metrics=['accuracy'])

# Print model summary to verify architecture
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=2)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)  # Apply threshold (0.5 for binary classification)

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

