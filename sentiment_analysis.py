

# Loading libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoModel, AutoConfig, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset
from evaluate import load


import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


# In[4]:


# Loading the twitter airline data
data = pd.read_csv("Tweets.csv")

data.head()


# In[5]:


data.info()


# In[6]:


# Checking for null values
data.isnull().sum()


# In[7]:


data.duplicated().sum()


# In[8]:


# Dropping duplicates
data.drop_duplicates(inplace = True)


# In[9]:


data.duplicated().sum()


# In[10]:


data.shape


# In[11]:


data.describe().T


# In[12]:


# Filtering the dataframe for negative sentiments only
negative_df = data[data['airline_sentiment'] == 'negative']

# Plotting the chart for distribution of negative reasons
plt.figure(figsize=(7, 5))
sns.countplot(y='negativereason', data=negative_df,
              order=negative_df['negativereason'].value_counts().index,
              palette='viridis')
plt.title('Distribution of Reasons for Negative Sentiments')
plt.xlabel('Count')
plt.ylabel('Negative Reasons')
plt.tight_layout()
plt.show()


# The chart above visualizes the distribution of reasons for negative sentiments in airline tweets. It filters the dataset to focus only on tweets with negative sentiments and then plots the frequency of each reported reason for negativity.

# In[13]:


# Plotting the distribution of sentiments for all airlines
plt.figure(figsize=(10, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=data, palette='coolwarm')
plt.title('Sentiment Distribution Across All Airlines')
plt.xlabel('Airline')
plt.ylabel('Count of Sentiments')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()


# In[14]:


# Grouping data by airline and sentiment
sentiment_count_by_airline = data.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(10, 6))
sentiment_count_by_airline.plot(kind='bar', stacked=True, color=['green', 'blue', 'red'], figsize=(10, 6))
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.title('Airline-Specific Sentiment Analysis')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()


# 
# From the Airline specific sentiment chart shows the distribution among airlines, such as United Airlines having more negative sentiment and Virgin America having the least, along with Southwest and United Airlines having more positive sentiment.
# 

# In[15]:


# Function to preprocess the text
def preprocess_text(text):
    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Single character removal
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Converting to Lowercase
    text = text.lower()

    return text



# Apply the preprocessing to the 'text' column
data['text'] = data['text'].apply(preprocess_text)
data.head()


# In[16]:


# Basic preprocessing
data = data[['text', 'airline', 'airline_sentiment']]  # Selecting the relevant columns
data.dropna(inplace=True)  # Dropping the missing values
# Convert categorical labels to numerical
data['airline_sentiment'].replace(['negative', 'neutral', 'positive'], [0, 1, 2], inplace=True)

data


# In[17]:


data.info()


# In[18]:


# Split the dataset

train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
train_data = train_data[:2500]
test_data = test_data[:2500]


# In[19]:


# Tokenization and Model Loading
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
config = AutoConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", output_hidden_states=True)
bert_model = AutoModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment",  config=config)


# # Training the GNN Model

# In[20]:


# Generate Embeddings for Text Data
def generate_embeddings(texts, model, tokenizer, batch_size=8):
    model.to('cuda')
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)

train_embeddings = generate_embeddings(train_data['text'].tolist(), bert_model, tokenizer)
test_embeddings = generate_embeddings(test_data['text'].tolist(), bert_model, tokenizer)


# In[21]:


train_embeddings_np = train_embeddings.detach().cpu().numpy()

# Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(train_embeddings_np)

# Define a threshold for similarity to consider an edge
similarity_threshold = 0.9  # You can adjust this threshold

# Create an edge list
edge_list = []
for i in range(similarity_matrix.shape[0]):
    for j in range(i + 1, similarity_matrix.shape[1]):
        if similarity_matrix[i, j] > similarity_threshold:
            edge_list.append((i, j))


# In[21]:





# In[22]:


edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()


# In[23]:


train_labels = torch.tensor(train_data['airline_sentiment'].values, dtype=torch.long)
test_labels = torch.tensor(test_data['airline_sentiment'].values, dtype=torch.long)

# Create graph data objects and DataLoader for GNN
train_dataset = [Data(x=train_embeddings, edge_index=edge_index, y=train_labels)]
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = [Data(x=test_embeddings, edge_index=edge_index, y=test_labels)]
test_loader = DataLoader(test_dataset, batch_size=8)



# In[23]:





# In[24]:


# # Convert edge_index to a NetworkX graph
# G = nx.Graph()
# edge_index_np = edge_index.numpy()
# for i in range(edge_index_np.shape[1]):
#     source, target = edge_index_np[:, i]
#     G.add_edge(int(source), int(target))

# # Draw the graph
# pos = nx.kamada_kawai_layout(G)
# plt.figure(figsize=(15, 15))  # Increase figure size
# nx.draw_networkx_nodes(G, pos, node_size=50)  # Increase node size
# nx.draw_networkx_edges(G, pos, width=0.5)


# plt.title('Text Similarity-Based Graph', fontsize=15)
# plt.show()


# In[25]:


# Define the GNN Model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Initialize GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first convolution and activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply the second convolution and activation
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply the third convolution and activation
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.25, training=self.training)

        # Apply the fourth convolution
        x = self.conv4(x, edge_index)

        return torch.log_softmax(x, dim=1)

# Train_embeddings is your input embeddings
input_dim = train_embeddings.shape[1] # Input dimension
hidden_dim = 16 # Hidden dimension
output_dim = 3 # Output dimension (3 classes for sentiment)

gnn_model = GNN(input_dim, hidden_dim, output_dim).to('cuda')


# In[26]:


optimizer = AdamW(gnn_model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

num_epochs = 300
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

losses = []
progress_bar = tqdm(range(num_training_steps))
gnn_model.train()
for epoch in range(num_epochs):

    for batch in train_loader:
        batch.to('cuda')
        outputs = gnn_model(batch)
        loss = criterion(outputs, batch.y)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        progress_bar.update(1)


# In[27]:


plt.plot(losses)
plt.xlabel('Batch or Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


# In[28]:


def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to('cuda')
            output = model(data)
            _, predicted = torch.max(output, dim=1)

            y_true.extend(data.y.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)


    return accuracy, precision, recall, f1_score


# In[29]:


# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model.to('cuda')

# Evaluate the model
accuracy, precision, recall, f1_score = evaluate_model(gnn_model, test_loader, device)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1_score:.4f}")


# In[30]:


# Performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1_score]

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values, palette="viridis")
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)

# Display the plot
plt.show()

