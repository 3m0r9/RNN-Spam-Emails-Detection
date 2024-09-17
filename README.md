# RNN Spam Emails Detection

This project implements a Recurrent Neural Network (RNN) model to detect spam emails. The model is trained on text data and leverages the sequential nature of RNNs to effectively classify emails as either spam or not spam. The project highlights how deep learning techniques can be applied to natural language processing (NLP) tasks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

With the increasing volume of email communication, spam detection has become an important task to prevent phishing, fraud, and unwanted advertisements. This project builds a Recurrent Neural Network (RNN) model to classify emails as either spam or not spam using a dataset of labeled email texts. RNNs are particularly effective for NLP tasks due to their ability to learn from sequential data and long-term dependencies in text.

This project focuses on:
- Preprocessing text data for use in an RNN model.
- Building and training an RNN architecture (with LSTM layers) for spam classification.
- Evaluating the performance of the model.

## Dataset

The dataset used in this project contains labeled email texts categorized as either "spam" or "ham" (non-spam). It can be sourced from common spam email datasets such as the [Enron Email Dataset](https://www.kaggle.com/wcukierski/enron-email-dataset) or [SpamAssassin](https://www.kaggle.com/veleon/ham-and-spam-dataset).

- **Classes**: 
  - **Spam**: Unsolicited emails, often containing phishing attempts or advertisements.
  - **Ham**: Regular emails that are not considered spam.
- **Features**: Text content of emails, which is processed to feed into the RNN model.

## Data Preprocessing

Text data requires extensive preprocessing to make it suitable for use in machine learning models:
- **Tokenization**: Emails are split into tokens (words).
- **Lowercasing**: All text is converted to lowercase to standardize the data.
- **Stopword Removal**: Common words like "the", "is", and "in" are removed as they do not contribute significantly to the classification task.
- **Stemming/Lemmatization**: Words are reduced to their root forms to further reduce the vocabulary size.
- **Padding and Truncating**: Each email text is padded or truncated to a fixed length to ensure uniform input size for the RNN.

## Modeling

### Recurrent Neural Network (RNN)

An RNN model is built using Long Short-Term Memory (LSTM) layers, which are effective at capturing long-term dependencies in sequential data like text.

- **Embedding Layer**: Transforms word indices into dense vectors of fixed size, creating word embeddings.
- **LSTM Layers**: Learn patterns and dependencies in the sequence of words.
- **Dense Layers**: Map the learned features to the output, which is a binary classification (spam or not spam).
- **Activation**: The output layer uses a `sigmoid` activation function to produce a probability score for the binary classification.

### Libraries Used:
- TensorFlow / Keras
- NumPy
- NLTK (for text preprocessing)
- Matplotlib (for visualization)

## Evaluation

The performance of the RNN model is evaluated using the following metrics:
- **Accuracy**: The percentage of correct classifications.
- **Precision**: How many of the emails classified as spam were actually spam.
- **Recall**: How many of the actual spam emails were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall, giving a balance between the two.
- **Confusion Matrix**: To provide insights into the number of true positives, false positives, true negatives, and false negatives.

## Installation

To run this project on your local machine, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/RNN-Spam-Emails-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RNN-Spam-Emails-Detection
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset (Enron or SpamAssassin) is downloaded and placed in the `data/` directory.
2. Preprocess the dataset:
   ```bash
   python preprocess_data.py --input data/emails.csv --output data/processed_emails.csv
   ```
3. Train the RNN model:
   ```bash
   python train_model.py --input data/processed_emails.csv
   ```
4. Evaluate the model:
   ```bash
   python evaluate_model.py --input data/processed_emails.csv
   ```

## Results

The RNN model achieved the following results:
- **Accuracy**: 95% on the test set.
- **Precision**: 94%.
- **Recall**: 93%.
- **F1-Score**: 93.5%.

Further details, including confusion matrices and classification reports, are available in the `results/` directory.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
