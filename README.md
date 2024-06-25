### Protein Function Prediction Using Convolutional Neural Networks

This repository contains code for predicting the function of proteins using Convolutional Neural Networks (CNNs) implemented in TensorFlow/Keras. The model utilizes protein sequence data and various biochemical properties for classification.

#### Dataset and Requirements

- **Dataset:** The sequences of two enzymes (`CrtB` and `CrtM`) are aligned using Clustal Omega and stored in a file (`Align.aln`).
- **Biochemical Properties:** Amino acid sequences are encoded using one-hot encoding along with additional features such as hydrophobicity, molecular weight, charge, polarity, aromaticity, and acidity/basicity.
- **Labels:** Sequences are labeled based on their function (`Function1` or `Function2`).

#### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your/repository.git
   cd repository
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn numpy biopython scikit-learn
   ```

#### Usage

1. **Data Preparation:**
   - Update file paths (`CrtB.fasta`, `crtM.fasta`, `Align.aln`) according to your dataset.

2. **Model Training:**
   - Run the script to preprocess data, build the CNN model, train, and evaluate:
     ```bash
     python train_model.py
     ```

3. **Evaluation:**
   - Evaluate the model performance using accuracy, precision, recall, F1-score, classification report, and confusion matrix.

#### Code Structure

- **`train_model.py`:** Main script to preprocess data, build the CNN model, train, and evaluate.
- **`utils.py`:** Utility functions for reading FASTA files, encoding sequences, and assigning labels.
- **`requirements.txt`:** List of Python dependencies.

#### Results

After training, the model's performance metrics are displayed, including accuracy on the test set, classification report, and confusion matrix. Additionally, training history plots showing accuracy and loss over epochs are generated.


#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#### Acknowledgments

- This project utilizes the TensorFlow/Keras framework and various Python libraries for data processing and visualization.
- Special thanks to the developers of Clustal Omega for sequence alignment.

