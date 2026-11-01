import nltk
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
# nltk.download('stopwords')
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
print(torch.version.cuda)  # Returns the CUDA version supported by PyTorch
print(torch.cuda.is_available())  # Checks if CUDA is available
vocab_to_int={}
import nltk
nltk.download('stopwords')

# Load the transfer learning tweet dataset
sentiment_df = pd.read_csv('finalSentimentdata2.csv')
# Checking if NVIDIA Graphics Card and CUDA is available
gpu_available = torch.cuda.is_available()

if gpu_available:
    print('Parallely Processing using CUDA')
else:
    print('No CUDA Detected')




# Structure of the Neural Network
class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding_layer(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        if gpu_available:
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden


# Pre-process the text and perform Stemming, Lemmatization and Stop-word removal
def text_preprocessing(text):
    remove_punctuation = [ch for ch in text if ch not in punctuation]
    remove_punctuation = "".join(remove_punctuation).split()
    filtered_text = [word.lower() for word in remove_punctuation if word.lower() not in stopwords.words('english')]
    return filtered_text


# Pad blank topken to keep the length of tweets consistent - mandatory to normalize and train the model
def pad_features(reviews_int, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, row in enumerate(reviews_int):
        if len(row) != 0:
            features[i, -len(row):] = np.array(row)[:seq_length]
    return features


# Convert the sentences into stream of tokens
def tokenize(tweet):
    test_ints = []
    with open("../vocab_int.json", "r") as f:
        vocab_to_int = json.load(f)
    test_ints.append([vocab_to_int[word] for word in tweet])
    return test_ints


# Predict the sentiment of the tweet - performs binary classification using the model inference
def sentiment(net, test_tweet, seq_length=50):
    print("\n--------------------------------------------------------------------------------------")
    print(f"\n Original input sentence: {test_tweet}")
    test_tweet = text_preprocessing(test_tweet)
    tokenized_tweet = tokenize(test_tweet)

    print(f"\n Pre-processed input sentence: {test_tweet}")
    # print(f"\nSentence converted into tokens:\n{tokenized_tweet}")

    padded_tweet = pad_features(tokenized_tweet, 50)
    feature_tensor = torch.from_numpy(padded_tweet)
    batch_size = feature_tensor.size(0)

    if gpu_available:
        feature_tensor = feature_tensor.cuda()

    h = net.init_hidden(batch_size)
    output, h = net(feature_tensor, h)

    predicted_sentiment = torch.round(output.squeeze())
    sentiment=""
    if predicted_sentiment == 1:
        print("\n Sentiment: Positive")
        sentiment="Positive"

    else:
        print("\n Sentiment: Negative")
        sentiment = "Negative"

    return test_tweet, sentiment


def createModel():
    sentiment_df.loc[:, 'text'] = sentiment_df['text'].apply(text_preprocessing)

    reviews_split = []
    for i, j in sentiment_df.iterrows():
        reviews_split.append(j['text'])

    words = []
    for review in reviews_split:
        for word in review:
            words.append(word)

    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    with open("../vocab_int.json", "w") as f:
        json.dump(vocab_to_int, f)

    encoded_reviews = []
    for review in reviews_split:
        encoded_reviews.append([vocab_to_int[word] for word in review])

    labels_to_int = []
    for i, j in sentiment_df.iterrows():
        if j['sentiment'] == 'joy':
            labels_to_int.append(1)
        else:
            labels_to_int.append(0)

    reviews_len = Counter([len(x) for x in encoded_reviews])
    non_zero_idx = [ii for ii, review in enumerate(encoded_reviews) if len(encoded_reviews) != 0]
    encoded_reviews = [encoded_reviews[ii] for ii in non_zero_idx]
    encoded_labels = np.array([labels_to_int[ii] for ii in non_zero_idx])

    seq_length = 50
    padded_features = pad_features(encoded_reviews, seq_length)

    # Split the dataset into Train (80%), Validation (10%) & Test (10%)
    batch_size = 1
    split_frac = 0.8
    split_idx = int(len(padded_features) * split_frac)

    training_x, remaining_x = padded_features[:split_idx], padded_features[split_idx:]
    training_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    # Transform the data into a Tensor datastructure
    train_data = TensorDataset(torch.from_numpy(training_x), torch.from_numpy(training_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    # Prepare the dataloader for Train, Test and Validation
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    # Embedding Dimension of Tokens
    embedding_dim = 400

    # Embedding Dimension of Hidden Layers
    hidden_dim = 256

    # Output of the model is binary (either Positive or Negative)
    output_size = 1

    # Number of hidden LSTM cells
    n_layers = 2
    vocab_size = len(vocab_to_int) + 1

    net = LSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    # Hyperparameters required for training of the network

    # Learning Rate
    lr = 0.001

    # Loss Function - Binary Cross Entropy
    criterion = nn.BCELoss()

    # Gradient Descent based Optimizer - ADAM (Adaptive LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Number of epochs to train the model
    epochs = 8
    count = 0

    # Step size
    print_every = 200
    clip = 5

    # Train the Neural Network
    # Off-load the model to CUDA
    if gpu_available:
        net.cuda()

    net.train()
    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            count += 1

            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()
            outputs, h = net(inputs, h)
            loss = criterion(outputs, labels.float())

            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), clip)
            optimizer.step()

            if count % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()

                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    if gpu_available:
                        inputs, labels = inputs.cuda(), labels.cuda()

                outputs, val_h = net(inputs, val_h)
                val_loss = criterion(outputs, labels.float())
                val_losses.append(val_loss.item())

                net.train()
                print(f"Epoch: {e + 1}/{epochs}.....", f"Step: {count}.....",
                      "Train Loss: {:.6f}......".format(loss.item()),
                      "Validation Loss: {:.6f}".format(np.mean(val_losses)))



    # Train the Neural Network
    test_losses = []
    num_correct = 0

    h = net.init_hidden(batch_size)
    net.eval()

    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])

        if gpu_available:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs, h = net(inputs, h)
        test_loss = criterion(outputs, labels.float())
        test_losses.append(test_loss.item())

        pred = torch.round(outputs)
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not gpu_available else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    test_acc = num_correct / len(test_loader.dataset)

    print("Average Test Loss: {:.4f}".format(np.mean(test_losses)))
    print("Average Test Accuracy: {:.4f}".format(test_acc))

    torch.save(net.state_dict(), "models/lstm_weights.pth")
    torch.save(net, "models/lstm_model.pth")

    return test_acc*100



    print("Model saved successfully")



def test_model(sample):
    model = torch.load("models/lstm_model.pth")


    sample_1 = 'Many lost their jobs because of covid and it is highly dangerous'
    #sentiment(model, sample_1)
    pro_tweet, senti=sentiment(model, sample)
    return pro_tweet, senti


#createModel()
"""
pro_tweet, senti=test_model("kashdga")
print(pro_tweet)
print(senti)
"""








