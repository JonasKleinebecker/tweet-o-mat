import asyncio
from collections import defaultdict
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from api_data_loader import fetch_and_store_german_politics_tweets
from data_preprocessing import (
    MongoDataset,
    label_to_onehot_batch,
    preprocess_text_batch,
    process_mongodb_collection_in_batch,
    tokenize_batch,
)
from model_training import (
    PretrainedTextModelWithClassficationHead,
    add_tokens_to_tokenizer,
    check_tokenization_of_words,
    train_model_classification,
)


def prepare_tokenizer(tokenizer: AutoTokenizer, model: AutoModel) -> None:
    """
    Add additional tokens to the tokenizer and resize the model's embedding layer to match the new vocabulary size.
    """
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
    check_tokenization_of_words(additional_tokens, tokenizer)
    add_tokens_to_tokenizer(additional_tokens, tokenizer, model)
    check_tokenization_of_words(additional_tokens, tokenizer)


def calculate_percentage_of_tweets_mentioning_topic_per_label(
    dataset: MongoDataset, topic_words: List[str]
) -> tuple:
    """
    Calculate the percentage of tweets mentioning a specific topic for each label
    and the overall occurrence of the topic across all labels.

    Args:
        dataset: The dataset containing tweets with "text" and "label" fields.
        topic_words: A list of words associated with the topic.

    Returns:
        A tuple containing:
        - A dictionary where keys are labels and values are the percentage of tweets
          mentioning the topic for that label.
        - The overall percentage of tweets mentioning the topic across all labels.
    """
    topic_counts_by_label = defaultdict(int)
    total_counts_by_label = defaultdict(int)
    total_topic_count = 0

    for document in dataset.data:
        text = document["text"]
        label = document["label"]

        total_counts_by_label[label] += 1

        if any(word.lower() in text.lower() for word in topic_words):
            topic_counts_by_label[label] += 1
            total_topic_count += 1

    percentage_by_label = {}
    for label, total_count in total_counts_by_label.items():
        topic_count = topic_counts_by_label.get(label, 0)
        percentage = (topic_count / total_count) * 100
        percentage_by_label[label] = percentage

    total_tweets = len(dataset)
    overall_percentage = (total_topic_count / total_tweets) * 100

    return percentage_by_label, overall_percentage


def plot_overall_occurrence(topics: dict, overall_percentages: dict) -> None:
    """
    Plot a bar chart showing the overall percentage occurrence of each topic.
    """
    plt.figure(figsize=(10, 6))
    topic_names = list(topics.keys())
    percentages = list(overall_percentages.values())

    plt.bar(topic_names, percentages, color="skyblue")
    plt.xlabel("Topics")
    plt.ylabel("Percentage of Tweets")
    plt.title("Overall Percentage Occurrence of Topics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_topic_occurrence_per_label(
    topic_name: str, percentages_by_label: dict
) -> None:
    """
    Plot a bar chart showing the percentage occurrence of a topic for each label.
    """
    plt.figure(figsize=(8, 5))
    labels = list(percentages_by_label.keys())
    percentages = list(percentages_by_label.values())
    plt.bar(labels, percentages, color="lightgreen")
    plt.xlabel("Labels")
    plt.ylabel("Percentage of Tweets")
    plt.title(f"Occurrence of '{topic_name}' by Label")
    plt.tight_layout()
    plt.show()


def explore_data() -> None:
    """
    Explore the dataset and perform some basic analysis.
    """
    dataset = MongoDataset(
        db_name="tweet-o-mat",
        collection_name="tweets",
        output_fields=["text", "label"],
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset sample: \n{dataset[0]}")

    topics = {
        "Migration": [
            "Migration",
            "Immigration",
            "Migranten",
            "Migrant",
            "Einwanderer",
            "Einwanderung",
            "Zuwanderung",
            "Zuwanderer",
            "Migrationspolitik",
            "Remigration",
        ],
        "Wirtschaft": [
            "Wirtschaft",
            "Wirtschaftskrise",
            "Wirtschaftsstandort",
            "Wirtschaftspolitik",
            "Rezession",
        ],
        "Klima": [
            "Klimawandel",
            "Klimakrise",
            "Klimapolitik",
            "Klima",
            "Erderwärmung",
        ],
        "Krieg": [
            "Krieg",
            "Angriffskrieg",
            "Kriegsverbrechen",
            "Kriegsverbrecher",
            "Ukraine",
            "Russland",
        ],
    }

    overall_percentages = {}
    for topic_name, topic_words in topics.items():
        percentages_by_label, overall_percentage = (
            calculate_percentage_of_tweets_mentioning_topic_per_label(
                dataset, topic_words
            )
        )
        overall_percentages[topic_name] = overall_percentage

    plot_overall_occurrence(topics, overall_percentages)

    for topic_name, topic_words in topics.items():
        percentages_by_label, _ = (
            calculate_percentage_of_tweets_mentioning_topic_per_label(
                dataset, topic_words
            )
        )
        plot_topic_occurrence_per_label(topic_name, percentages_by_label)


def split_data_and_train_model(text_model: AutoModel) -> nn.Module:
    """
    Split the data and train the model using a specific text_model as a base.
    """
    bert_model = PretrainedTextModelWithClassficationHead(
        text_model=text_model,
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

    dataset = MongoDataset(
        db_name="tweet-o-mat",
        collection_name="tweets_tokenized",
        output_fields=["input_ids", "attention_mask", "token_type_ids", "label"],
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_model_classification(
        model=bert_model,
        trainloader=train_dataloader,
        testloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        test_interval=1,
        epochs=20,
        metric_fns=[accuracy_score],
    )
    return bert_model


def main() -> None:
    MODEL_NAME = "bert-base-german-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    asyncio.run(fetch_and_store_german_politics_tweets())

    explore_data()

    prepare_tokenizer(tokenizer, model)

    process_mongodb_collection_in_batch(  # preprocess and tokenize data
        db_name="tweet-o-mat",
        source_collection_name="tweets",
        target_collection_name="tweets_tokenized",
        batch_size=32,
        X_transform_fns=[
            preprocess_text_batch,
            partial(tokenize_batch, tokenizer=tokenizer),
        ],
        y_transform_fns=[label_to_onehot_batch],
        db_fields=["text", "label"],
    )

    trained_model = split_data_and_train_model(model)

    torch.save(trained_model.state_dict(), "bert_base_model_v2.pt")


if __name__ == "__main__":
    main()
