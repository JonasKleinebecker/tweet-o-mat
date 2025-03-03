import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


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

# process_mongodb_collection_in_batch(
#     db_name="tweet-o-mat",
#     source_collection_name="tweets_onehot",
#     target_collection_name="tweets_tokenized_new",
#     batch_size=4599,
#     X_transform_fns=[tokenize_batch],
#     y_transform_fns=[],
#     db_fields=["text", "label"],
# )

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

dataset = MongoDataset(
    db_name="tweet-o-mat",
    collection_name="tweets_tokenized_new",
    output_fields=["input_ids", "attention_mask", "token_type_ids", "label"],
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def train_step_classification(
    model: nn.Module,
    trainloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_fns=[],
):
    """
    Trains a PyTorch classification model for a single epoch.

    Args:
      model: A PyTorch model to train.
      data_loader: A DataLoader for the training data.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to use for gradient descent.
      device: A torch.device to run the model on.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed metrics.
    """
    model.train()
    metrics = {}
    metrics["loss"] = 0
    for metric_fn in metric_fns:
        metrics[metric_fn.__name__] = 0
    for X, y in tqdm(trainloader):
        X = tuple(t.to(device) for t in X)
        y = y.to(device)
        y_logits = model(*X)
        y_preds = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y)
        y = torch.argmax(y, dim=1)
        metrics["loss"] += loss.item()
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] += metric_fn(y.cpu(), y_preds.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for metric_fn in metric_fns:
        metrics[metric_fn.__name__] /= len(trainloader)
    metrics["loss"] /= len(trainloader)
    return metrics


def test_step_classification(
    model: nn.Module,
    testloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metric_fns=[],
):
    """
    Tests a PyTorch classification model on a dataset.

    Args:
      model: A PyTorch model to test.
      data_loader: A DataLoader for the test data.
      loss_fn: A PyTorch loss function to minimize.
      device: A torch.device to run the model on.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed metrics.
    """
    model.eval()
    with torch.inference_mode():
        metrics = {}
        metrics["loss"] = 0
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] = 0
        for X, y in tqdm(testloader):
            X = tuple(t.to(device) for t in X)
            y = y.to(device)
            y_logits = model(*X)
            y_preds = torch.argmax(y_logits, dim=1)
            y = torch.argmax(y, dim=1)
            loss = loss_fn(y_logits, y)
            metrics["loss"] += loss.item()
            for metric_fn in metric_fns:
                metrics[metric_fn.__name__] += metric_fn(y.cpu(), y_preds.cpu())
        for metric_fn in metric_fns:
            metrics[metric_fn.__name__] /= len(testloader)
        metrics["loss"] /= len(testloader)
    return metrics


def train_model_classification(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    test_interval: int,
    epochs: int,
    metric_fns=[],
):
    """
    Trains a PyTorch classification model.

    Args:
      model: A PyTorch model to train.
      trainloader: A DataLoader for the training data.
      testloader: A DataLoader for the test data.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to use for gradient descent.
      device: A torch.device to run the model on.
      epochs: The number of epochs to train for.
      test_interval: The number of epochs to wait before testing the model.
      metric_fns: A list of metric functions to compute. arguments are passed based on the sklearn convention (y_true,
      y_pred.)
    Returns:
      A dictionary of the computed training and test metrics as well as the time it took to train the model.
    """
    train_metrics = {}
    test_metrics = {}
    train_metrics["loss"] = []
    test_metrics["loss"] = []
    for metric_fn in metric_fns:
        train_metrics[metric_fn.__name__] = []
        test_metrics[metric_fn.__name__] = []

    train_start = time.time()

    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()
        print(f"Epoch {epoch}\n----------")
        print("Training:")
        metrics = train_step_classification(
            model, trainloader, loss_fn, optimizer, device, metric_fns
        )
        train_metrics["loss"].append(metrics["loss"])
        for metric_fn in metric_fns:
            train_metrics[metric_fn.__name__].append(metrics[metric_fn.__name__])
        for metric in train_metrics:
            print(f"    {metric}: {train_metrics[metric][-1]:.3f}")
        if epoch % test_interval == 0:
            print("Testing:")
            metrics = test_step_classification(
                model, testloader, loss_fn, device, metric_fns
            )
            test_metrics["loss"].append(metrics["loss"])
            for metric_fn in metric_fns:
                test_metrics[metric_fn.__name__].append(metrics[metric_fn.__name__])
            for metric in test_metrics:
                print(f"    {metric}: {test_metrics[metric][-1]:.3f}")
        epoch_time = time.time() - epoch_start
        print(f"Finished epoch in {epoch_time:.3f} seconds.")

    train_time = time.time() - train_start
    print(f"Finished training in {train_time:.3f} seconds.")
    return train_metrics, test_metrics, train_time


train_model_classification(
    model=bert_model,
    trainloader=train_dataloader,
    testloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    test_interval=1,
    epochs=5,
    metric_fns=[accuracy_score],
)

torch.save(bert_model.state_dict(), "bert_base_model_v1.pt")
