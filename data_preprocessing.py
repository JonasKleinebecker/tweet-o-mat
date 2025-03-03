import re

import torch
from pymongo import MongoClient
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.tokenization_utils_base import BatchEncoding


def preprocess_text_batch(texts):
    """
    Preprocess the text batch by removing special characters, URLs, and extra spaces.
    """
    for i, text in enumerate(texts):
        text = text.replace("–", "-")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß.,!?;:'\"()\s-]", "", text)
        text = re.sub(r"http\S+", "URL", text)
        texts[i] = text
    return texts


def tokenize_batch(batch, tokenizer, max_length=128):
    """
    Tokenize the text batch using the provided tokenizer.
    """
    tokenized = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
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
    """
    Processes a MongoDB collection in batches, applies transformation functions to the data,
    and writes the transformed data to a target collection.

    This function is designed to handle large datasets by loading and processing data in
    batches. It supports transformations for both features (X) and labels (y), and can
    handle different types of output formats (e.g., `BatchEncoding`, `BaseModelOutputWithPoolingAndCrossAttentions`).

    Args:
        db_name (str): The name of the MongoDB database containing the source and target collections.
        source_collection_name (str): The name of the source MongoDB collection to read data from.
        target_collection_name (str): The name of the target MongoDB collection to write transformed data to.
        batch_size (int): The number of documents to process in each batch.
        X_transform_fns (list of callable): A list of transformation functions to apply to the features (X).
                                           Each function should accept a list of feature data and return
                                           transformed feature data.
        y_transform_fns (list of callable): A list of transformation functions to apply to the labels (y).
                                           Each function should accept a list of label data and return
                                           transformed label data.
        db_fields (list of str): A list of fields to retrieve from the source collection.
        db_query (dict, optional): A MongoDB query to filter documents in the source collection.
                                   Defaults to an empty dictionary, which retrieves all documents.

    Behavior:
        1. Loads data from the source MongoDB collection in batches using a PyTorch DataLoader.
        2. Applies the provided transformation functions to the features (X) and labels (y) in each batch.
        3. Handles different types of transformed outputs:
           - If the transformed features are of type `BatchEncoding` (e.g., from a Hugging Face tokenizer),
             the function extracts `input_ids`, `attention_mask`, and `token_type_ids` for each document.
           - If the transformed features are of type `BaseModelOutputWithPoolingAndCrossAttentions`
             (e.g., from a Hugging Face model), the function extracts the `last_hidden_state` for each document.
           - For other types of transformed features, the function assumes the data is in a simple text format.
        4. Writes the transformed data to the target MongoDB collection.

    Example:
        ```python
        # Define transformation functions
        def tokenize_text(texts):
            return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        def encode_labels(labels):
            return [label_encoder[label] for label in labels]

        # Process the MongoDB collection
        process_mongodb_collection_in_batch(
            db_name="my_database",
            source_collection_name="raw_tweets",
            target_collection_name="processed_tweets",
            batch_size=32,
            X_transform_fns=[tokenize_text],
            y_transform_fns=[encode_labels],
            db_fields=["text", "label"],
            db_query={"label": {"$exists": True}},
        )
        ```

    Notes:
        - The function assumes that the source collection contains documents with fields specified in `db_fields`.
        - The target collection will contain documents with transformed fields (e.g., `input_ids`, `attention_mask`,
          `last_hidden_state`, or `text`), depending on the transformations applied.
        - Ensure that the transformation functions are compatible with the data format in the source collection.
    """

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
    """
    Convert a batch of labels to one-hot encoding. Only works for the labels "afd", "cdu", "die grünen", "fdp", and "spd".
    """
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


def is_list_of_numbers(obj):
    """
    Check if an object is a list of numbers (int or float).
    """
    return isinstance(obj, list) and all(isinstance(item, (int, float)) for item in obj)


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
        if is_list_of_numbers(label):
            label = torch.tensor(
                label, dtype=torch.float32
            )  # needed to prevent pytorch dataset from stacking the labels into column tensors

        other_fields = []
        for field in self.output_fields:
            if field != "_id" and field != "label":
                field_value = document[field]
                if is_list_of_numbers(field_value):
                    field_value = torch.tensor(field_value, dtype=torch.long)
                other_fields.append(field_value)
        return tuple(other_fields), label
