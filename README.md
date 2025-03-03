# Tweet-o-mat: German Political Tweets Analysis & Classification
Tweet-o-mat is a Python-based project that fetches, processes, analyzes, and classifies tweets from German politicians. The project leverages a pretrained German BERT model to fine-tune a classification head and uses MongoDB as the data store. It also performs exploratory data analysis by plotting the occurrence of key political topics in tweets.

## Features
### Data Collection:
Fetch tweets from multiple German political figures using the Twikit library and store them in a local MongoDB database.

### Data Preprocessing:
Clean and preprocess tweet text by removing URLs, special characters, and extra spaces. Tokenize the text using Hugging Face’s transformers and add custom tokens to improve the model’s performance on political content.

### Exploratory Data Analysis:
Analyze the occurrence of topics (e.g., Migration, Economy, Climate, War) across tweets and generate visualizations (bar charts) to better understand trends across different political labels.

### Model Training:
Fine-tune a pretrained BERT model (bert-base-german-cased) by adding a classification head. The training pipeline uses PyTorch along with scikit-learn metrics to evaluate performance.

## Project Structure
### main.py:
The main entry point that ties together data fetching, preprocessing, exploratory analysis, and model training.

### api_data_loader.py:
Contains functions to log into Twitter via the Twikit library, fetch tweets from specified handles, and store them in MongoDB.

### data_preprocessing.py:
Provides functions for cleaning and tokenizing tweet text, converting labels to one-hot encoding, and processing MongoDB collections in batches.

### model_training.py:
Implements the model architecture that wraps a pretrained text model with a classification head. Also contains the training and evaluation loops.

## Prerequisites
Python: 3.8+
MongoDB: Running on localhost:27017
Twitter API Credentials: Required for Twikit (replace <auth_info_1> and <password> with your credentials)
## Dependencies:
PyTorch
Transformers
scikit-learn
pymongo
matplotlib
tqdm
Twikit

## Usage
### MongoDB:
Ensure you have a local MongoDB instance running on port 27017. 

### Configure API Credentials:
Update the placeholder values (<auth_info_1> and <password>) in api_data_loader.py with your actual Twitter API credentials.

**!!!Make sure the usage of twitter API using twikit is in line with the twitter TOS or your account might get suspended!!!**
### Run the Main Script:
python main.py

After training, the model weights will be saved as bert_base_model_v2.pt in the project directory.
