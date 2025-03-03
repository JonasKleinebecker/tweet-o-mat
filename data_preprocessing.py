import re

import torch
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding


def preprocess_text(texts):
    for i, text in enumerate(texts):
        text = text.replace("–", "-")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß.,!?;:'\"()\s-]", "", text)
        text = re.sub(r"http\S+", "URL", text)
        texts[i] = text
    return texts


def tokenize_batch(batch, tokenizer):
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

        other_fields = []
        for field in self.output_fields:
            if field != "_id" and field != "label":
                field_value = document[field]
                if isinstance(field_value, list):
                    field_value = torch.tensor(field_value, dtype=torch.long)
                other_fields.append(field_value)
        return tuple(other_fields), label_tensor
