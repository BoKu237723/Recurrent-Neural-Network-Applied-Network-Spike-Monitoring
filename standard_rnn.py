import numpy as np
from collections import defaultdict
import string

class RNNWordPredictor:
    def __init__(self, hidden_size = 64, learning_rate = 0.05):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0

    def preprocess_text(self, text):
        translator = str.maketrans('','', string.punctuation)
        text = text.translate(translator).lower()
        return text

    def build_vocab(self, text):
        word_counts = defaultdict(int)
        
        cleaned_text = self.preprocess_text(text)
        words = cleaned_text.split()
        for word in words:
            word_counts[word] += 1

        self.word_to_index = {'<UNK>':0}
        self.index_to_word = {0 : '<UNK>'}

        for word in word_counts:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_index)

        return self.vocab_size
    
    def text_to_sequence(self, text):
        cleaned_text = self.preprocess_text(text)
        words = cleaned_text.split()
        sequence = []
        for word in words:
            sequence.append(self.word_to_index.get(word, 0))
        return sequence

    def prepare_trining_data(self, texts, sequence_length = 5):
        x = []
        y = []

        sequence = self.text_to_sequence(texts)
        
        for i in range(len(sequence) - sequence_length -1):
            x.append(sequence[i:i+sequence_length - 1])
            y.append(sequence[i+sequence_length])
        return np.array(x), np.array(y)

    def init_parameters(self):
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

    def softmax(self, ys):
        exp_ys = np.exp(ys - np.max(ys))
        return exp_ys / exp_ys.sum()

    def forward(self, inputs, h_prev):
        xs = {} # inputs (one-hot vector)
        hs = {} # hidden states
        ys = {} # output
        ps = {} # probabilities (prediction)

        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size,1))
            xs[t][inputs[t]] = 1

            # ht = tanh(Whh*ht-1 + Wxh*xt)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

            ys[t] = np.dot(self.Why, hs[t]) + self.by

            ps[t] = self.softmax(ys[t])

        return xs, hs, ps
            
    def backwards(self, xs, hs, ps, targets):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext

            # Hidden layer gradients (tanh derivative)
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

            for d in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(d, -5, 5, out=d)

        return dWxh, dWhh, dWhy, dbh, dby

    def train(self, X, y, epochs = 50, verbose = True):
        self.init_parameters()

        h_prev = np.zeros((self.hidden_size, 1))

        losses = []

        for epoch in range(epochs):
            total_loss = 0
            h_prev = np.zeros((self.hidden_size, 1))

            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(len(X_shuffled)):
                inputs = X_shuffled[i]
                target = y_shuffled[i]

                # Forward Pass
                xs, hs, ps = self.forward(inputs, h_prev)
                loss = -np.log(ps[len(ps)-1][target, 0])
                total_loss += loss

                dWxh, dWhh, dWhy, dbh, dby = self.backwards(xs, hs, ps, [target])

                self.Wxh -= self.learning_rate * dWxh
                self.Whh -= self.learning_rate * dWhh
                self.Why -= self.learning_rate * dWhy
                self.bh -= self.learning_rate * dbh
                self.by -= self.learning_rate * dby

                h_prev = hs[len(inputs)-1]
            avg_loss = total_loss / len(X_shuffled)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 2 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        return losses


    def predict_next_word(self, context_words, temperature = 0.8):
        context_indices = []
        for word in context_words:
            if word in self.word_to_index:
                context_indices.append(self.word_to_index[word])
            else:
                context_indices.append(0) # UNK token

        h = np.zeros((self.hidden_size, 1))

        for idx in context_indices:
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        
        y = np.dot(self.Why, h) + self.by
        y = y / temperature
        probs = self.softmax(y)

        next_idx = np.random.choice(range(self.vocab_size), p = probs.ravel())

        return self.index_to_word[next_idx]
        

    def generate_text(self, start_words, num_words = 10, temperature = 0.8):
        generated = start_words.copy()
        context = start_words.copy()
        
        for _ in range(num_words):
            next_word = self.predict_next_word(context, temperature)
            generated.append(next_word)

        return ' '.join(generated)

def main():
    with open("simulation.txt", "r", encoding="utf-8") as f:
        training_texts = f.read()
    rnn = RNNWordPredictor(hidden_size=32, learning_rate=0.05)

    print("Building vocabulary...")
    vocab_size = rnn.build_vocab(training_texts)
    print(f"Vocabulary size: {vocab_size}")

    print("Preparing training data...")
    X, y = rnn.prepare_trining_data(training_texts, sequence_length=3)
    print(f"Training samples: {len(X)}")

    print("Training RNN...")
    losses = rnn.train(X, y, epochs = 30, verbose = True)

    print("\nTesting predictions:")
    test_contexts = [
        ["the", "quick"]
    ]

    for context in test_contexts:
        next_word = rnn.predict_next_word(context, temperature=0.5)
        print(f"Context: {context} -> Next word: {next_word}")
    
    print("\nGenerating text:")
    for start in [["the"], ["a"], ["to"]]:
        generated = rnn.generate_text(start, num_words=10, temperature=0.5)
        print(f"Start: {start[0]} -> {generated}")
        
if __name__ == "__main__":
    main()