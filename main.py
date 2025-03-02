import re
import string

import torch
import torch.nn as nn
from pymongo import MongoClient
from tokenizers import Encoding
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding


def preprocess_text(texts):
    for i, text in enumerate(texts):
        text = text.lower()
        text = text.replace("–", "-")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß.,!?;:'\"()\s-]", "", text)
        text = re.sub(r"http\S+", "URL", text)
        texts[i] = text
    return texts


def tokenize_batch(batch):
    tokenized = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return tokenized


def process_mongodb_collection_in_batch(
    db_name,
    source_collection_name,
    target_collection_name,
    batch_size,
    X_transform_fns,
    y_transform_fns,
    db_fields,
    db_query={},
):
    dataset = MongoDataset(db_name, source_collection_name, db_query, db_fields)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_collection = MongoClient("mongodb://localhost:27017/")[db_name][
        target_collection_name
    ]

    for X_tuple, y_tuple in dataloader:
        X = [item for sublist in X_tuple for item in sublist]
        y = list(y_tuple)
        for transform_fn in X_transform_fns:
            X = transform_fn(X)
        for transform_fn in y_transform_fns:
            y = transform_fn(y)
        if isinstance(X, BatchEncoding):
            transformed_batch = []
            for i in range(len(X["input_ids"])):
                transformed_batch.append(
                    {
                        "input_ids": X["input_ids"][i].tolist(),
                        "attention_mask": X["attention_mask"][i].tolist(),
                        "token_type_ids": X["token_type_ids"][i].tolist(),
                        "label": y[i].tolist(),
                    }
                )
        elif isinstance(X, BaseModelOutputWithPoolingAndCrossAttentions):
            transformed_batch = []
            for i in range(len(X.last_hidden_state)):
                transformed_batch.append(
                    {
                        "last_hidden_state": X.last_hidden_state[i].tolist(),
                        "label": y[i],
                    }
                )
        else:
            transformed_batch = [{"text": x, "label": y} for x, y in zip(X, y)]
        target_collection.insert_many(transformed_batch)


def label_to_onehot_batch(batch):
    one_hot_batch = []
    label_to_onehot_dict = {
        "afd": [1, 0, 0, 0, 0],
        "cdu": [0, 1, 0, 0, 0],
        "die grünen": [0, 0, 1, 0, 0],
        "fdp": [0, 0, 0, 1, 0],
        "spd": [0, 0, 0, 0, 1],
    }
    for label in batch:
        one_hot_batch.append(label_to_onehot_dict[label])
    return one_hot_batch


class MongoDataset(Dataset):
    def __init__(self, db_name, collection_name, query={}, output_fields=None):
        """
        Pytorch Dataset class for loading data from a MongoDB collection.
        Args:
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the MongoDB collection.
            query (dict): Query to filter documents in the collection.
            output_fields (list): List of fields to retrieve from the documents.
                                 If None, retrieves all fields except "_id".
        """
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        if output_fields is None:
            self.output_fields = {"_id": 0}  # Exclude "_id" by default
        else:
            self.output_fields = {field: 1 for field in output_fields}
            self.output_fields["_id"] = 0  # Exclude "_id" by default

        self.ids = list(self.collection.find(query, {"_id": 1}))

        self.data = list(self.collection.find(query, self.output_fields))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        document = self.data[index]

        label = document.get("label", [])
        label_tensor = torch.tensor(
            label, dtype=torch.float32
        )  # needed to prevent pytorch dataset from stacking the labels into column tensors

        other_fields = tuple(
            document[field]
            for field in self.output_fields
            if field != "_id" and field != "label"
        )
        return other_fields, label_tensor


class PretrainedTextModelWithClassficationHead(nn.Module):
    def __init__(
        self,
        text_model,
        num_classes,
        train_embeddings=False,
        train_text_model_core=False,
    ):
        super().__init__()
        self.text_model = text_model
        for param in self.text_model.parameters():
            if train_text_model_core:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.text_model.embeddings.parameters():
            if train_embeddings:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.classifier = nn.Linear(text_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


MODEL_NAME = "bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

additional_tokens = [
    "URL",
    "weidel",
    "merz",
    "söder",
    "lindner",
    "habeck",
    "scholz",
    "afd",
    "cdu",
    "csu",
    "spd",
    "fdp",
    "bsw",
]


def check_words_in_tokenizer(words, tokenizer):
    for word in words:
        if word not in tokenizer.vocab.keys():
            print(f"Word {word} not in tokenizer")
        else:
            print(f"Word {word} in tokenizer")


def check_tokenization_of_words(words, tokenizer):
    for word in words:
        tokenized_word = tokenizer.tokenize(word)
        print(f"Tokenized word {word}: {tokenized_word}")


check_tokenization_of_words(additional_tokens, tokenizer)


def add_tokens_to_tokenizer(words, tokenizer, model):
    for word in words:
        tokenizer.add_tokens(word)
    model.resize_token_embeddings(len(tokenizer))


check_words_in_tokenizer(additional_tokens, tokenizer)

add_tokens_to_tokenizer(additional_tokens, tokenizer, model)

process_mongodb_collection_in_batch(
    db_name="tweet-o-mat",
    source_collection_name="tweets_onehot",
    target_collection_name="tweets_tokenized",
    batch_size=32,
    X_transform_fns=[tokenize_batch],
    y_transform_fns=[],
    db_fields=["text", "label"],
)

bert_model = PretrainedTextModelWithClassficationHead(
    text_model=model,
    num_classes=5,
    train_embeddings=True,
    train_text_model_core=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(
    [
        {
            "params": bert_model.text_model.embeddings.parameters(),
            "lr": 1e-4,
        },
        {"params": bert_model.classifier.parameters(), "lr": 1e-3},
    ]
)

dataset = MongoDataset(db_name="tweet-o-mat", collection_name="tokenized_tweets")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for epoch in range(10):
    print(f"Epoch {epoch}")
    bert_model.train()
    for input_ids, attention_mask, token_type_ids, y in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        y = y.to(device)
        logits = bert_model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, y)
        print(f"Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
