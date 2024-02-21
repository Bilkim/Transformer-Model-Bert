# Project Description
This project focuses on sentiment analysis of airline tweets using Graph Neural Networks (GNN). It involves loading, preprocessing, and analyzing Twitter data to identify public sentiment towards various airlines. The project leverages deep learning models, particularly a GNN, to classify tweets into negative, neutral, or positive sentiments.

# Dependencies
Python 3.x
Libraries: numpy, pandas, networkx, matplotlib, seaborn, re, nltk, torch, torch-geometric, transformers, scikit-learn, tqdm, evaluate
A CUDA-compatible GPU for training the models
Dataset
The dataset used is the "Tweets.csv" file, which contains tweets related to airlines, along with their sentiments and other metadata.

# Features
Preprocessing of text data including cleaning and normalization
Visualization of sentiment distribution across airlines and reasons for negative sentiments
Utilization of BERT embeddings for feature extraction from textual data
Construction of a graph based on cosine similarity of tweet embeddings
Training a Graph Neural Network for sentiment classification
Evaluation of model performance using accuracy, precision, recall, and F1-score

# Setup and Installation
* Ensure all dependencies are installed using
  ``
  pip install <library>.
  ``
* Place the "Tweets.csv" file in the project directory.
* Run the script to perform sentiment analysis.
  
# Usage
To use this script:

Load the dataset using pandas.
Perform data preprocessing and exploratory data analysis.
Generate embeddings for the text data using a pretrained BERT model.
Construct a similarity graph based on the embeddings.
Define and train the GNN model on the constructed graph.
Evaluate the model on the test data.

# Model Architecture
The GNN model uses several GCNConv layers with dropout and ReLU activation functions, followed by a log-softmax layer for classification.

# Evaluation
The model is evaluated on a subset of the data using metrics like accuracy, precision, recall, and F1-score. The results are visualized to provide insights into the model's performance.

# N/B :
* The script contains several visualization blocks to understand data distribution and model performance.
* Adjust hyperparameters like similarity_threshold, hidden_dim, or num_epochs as needed for different dataset characteristics or to fine-tune the model.
* Ensure that CUDA is available for training the model efficiently.
