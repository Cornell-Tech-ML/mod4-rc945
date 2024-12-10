import random
import zipfile
import os
import embeddings
import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)

def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value

class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        return minitorch.conv1d(input, self.weights.value) + self.bias.value

class CNNSentimentKim(minitorch.Module):
    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.dropout = dropout
        self.conv1d1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.conv1d2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.conv1d3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])
        self.linear = Linear(self.feature_map_size, 1)

    def forward(self, embeddings):
        embeddings = embeddings.permute(0, 2, 1)
        conv1 = self.conv1d1.forward(embeddings).relu()
        conv2 = self.conv1d2.forward(embeddings).relu()
        conv3 = self.conv1d3.forward(embeddings).relu()
        out = (
            minitorch.max(conv1, 2) + minitorch.max(conv2, 2) + minitorch.max(conv3, 2)
        ).view(embeddings.shape[0], self.feature_map_size)
        out = minitorch.dropout(out, rate=self.dropout, ignore=not self.training)
        out = self.linear.forward(out)
        return out.sigmoid().view(embeddings.shape[0])

def get_predictions_array(y_true, model_output):
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array

def get_accuracy(predictions_array):
    correct = 0
    for (y_true, y_pred, logit) in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)

best_val = 0.0

def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")

class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]
                optim.step()
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()
            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0

def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)
    ys = dataset["label"][:N]
    return Xs, ys

def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))
    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")
    return (X_train, y_train), (X_val, y_val)

dataset = load_dataset("glue", "sst2")

class GloveEmbedding:
    def __init__(self, filepath, d_emb=50):
        self.filepath = filepath
        self.d_emb = d_emb
        self.word2emb = {}
        self.load_embeddings()

    def load_embeddings(self):
        print("Loading GloVe embeddings...")
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = [float(val) for val in values[1:]]
                self.word2emb[word] = vector
        print(f"Loaded {len(self.word2emb)} word vectors")

    def __contains__(self, word):
        return word in self.word2emb

    def emb(self, word):
        return self.word2emb[word]

try:
    glove_path = "glove/glove.6B.50d.txt"
    pretrained_embeddings = GloveEmbedding(glove_path, d_emb=50)
except FileNotFoundError:
    print("Error: GloVe embeddings file not found at:", glove_path)
    print("Please specify the correct path to glove.6B.50d.txt")
    exit(1)

(X_train, y_train), (X_val, y_val) = encode_sentiment_data(
    dataset,
    pretrained_embeddings,
    N_train=450,
    N_val=100
)

model_trainer = SentenceSentimentTrain(
    CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
)
model_trainer.train(
    (X_train, y_train),
    learning_rate=0.01,
    max_epochs=250,
    data_val=(X_val, y_val),
)