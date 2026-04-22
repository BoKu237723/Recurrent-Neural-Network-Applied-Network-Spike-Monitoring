import numpy as np
from collections import defaultdict
import datetime

class NetworkTrafficRNN:
    def __init__(self, hidden_size=64, learning_rate=0.01, sequence_length=12):
        """
        Initialize RNN for network traffic prediction
        hidden_size: number of hidden units
        learning_rate: learning rate for training
        sequence_length: number of time steps to look back (default 12 = 1 hour at 5-min intervals)
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.input_size = 1  # Single value: traffic in MB
        self.output_size = 1  # Single value: predicted traffic in MB
        
        # Normalization parameters
        self.mean = 0
        self.std = 1
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize RNN parameters using Xavier initialization"""
        # Input to hidden weights
        self.Wxh = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2.0 / self.input_size)
        # Hidden to hidden weights
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        # Hidden to output weights
        self.Why = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        
        # Biases
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.output_size, 1))
        
        # For Adam optimizer
        self.m_Wxh, self.v_Wxh = np.zeros_like(self.Wxh), np.zeros_like(self.Wxh)
        self.m_Whh, self.v_Whh = np.zeros_like(self.Whh), np.zeros_like(self.Whh)
        self.m_Why, self.v_Why = np.zeros_like(self.Why), np.zeros_like(self.Why)
        self.m_bh, self.v_bh = np.zeros_like(self.bh), np.zeros_like(self.bh)
        self.m_by, self.v_by = np.zeros_like(self.by), np.zeros_like(self.by)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def normalize(self, data):
        """Normalize data to zero mean and unit variance"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std < 1e-6:
            self.std = 1
        return (data - self.mean) / self.std
    
    def denormalize(self, data):
        """Convert normalized data back to original scale and round to integers"""
        denormalized = data * self.std + self.mean
        return np.round(denormalized).astype(int)  # Round to nearest integer
    
    def load_data(self, filename):
        """Load network traffic data from file"""
        with open(filename, 'r') as f:
            content = f.read()
            data = [int(x) for x in content.split() if x.strip()]
        
        return np.array(data)
    
    def prepare_training_data(self, data):
        """Prepare sequences for training"""
        X = []
        y = []
        
        # Normalize the data
        normalized_data = self.normalize(data)
        
        # Create sequences
        for i in range(len(normalized_data) - self.sequence_length):
            X.append(normalized_data[i:i+self.sequence_length])
            y.append(normalized_data[i+self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def forward(self, inputs, h_prev):
        """
        Forward pass through RNN
        inputs: sequence of input values (shape: sequence_length x 1)
        h_prev: previous hidden state
        """
        xs = {}  # inputs
        hs = {}  # hidden states
        ys = {}  # outputs
        
        hs[-1] = np.copy(h_prev)
        
        for t in range(len(inputs)):
            # Input at time t (reshape to column vector)
            xs[t] = np.array([[inputs[t]]])
            
            # Hidden state update: h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            
            # Output: y_t = Why * h_t + by
            ys[t] = np.dot(self.Why, hs[t]) + self.by
        
        return xs, hs, ys
    
    def backward(self, xs, hs, ys, target):
        """
        Backward pass through time (BPTT)
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        # Loop backwards through time
        for t in reversed(range(len(xs))):
            # Output error (MSE derivative)
            dy = ys[t] - target[t] if t == len(xs) - 1 else np.zeros_like(ys[t])
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # Hidden state error
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Gradient through tanh
            dh_raw = (1 - hs[t] * hs[t]) * dh
            
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            dWhh += np.dot(dh_raw, hs[t-1].T)
            
            # Error for previous time step
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        # Clip gradients to prevent explosion
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -5, 5, out=d)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def train(self, X, y, epochs=50, batch_size=32, verbose=True):
        """
        Train the RNN using Adam optimizer
        """
        n_samples = len(X)
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Initialize gradients
                dWxh_sum = np.zeros_like(self.Wxh)
                dWhh_sum = np.zeros_like(self.Whh)
                dWhy_sum = np.zeros_like(self.Why)
                dbh_sum = np.zeros_like(self.bh)
                dby_sum = np.zeros_like(self.by)
                
                batch_loss = 0
                
                # Process each sequence in batch
                for j in range(len(batch_X)):
                    # Reset hidden state for each sequence
                    h_prev = np.zeros((self.hidden_size, 1))
                    
                    # Forward pass
                    xs, hs, ys = self.forward(batch_X[j], h_prev)
                    
                    # Calculate loss (MSE)
                    target = np.array([[batch_y[j]]])
                    loss = np.mean((ys[len(ys)-1] - target) ** 2)
                    batch_loss += loss
                    
                    # Prepare target sequence for backward pass
                    target_seq = [None] * len(ys)
                    target_seq[len(target_seq)-1] = target
                    
                    # Backward pass
                    dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ys, target_seq)
                    
                    # Accumulate gradients
                    dWxh_sum += dWxh
                    dWhh_sum += dWhh
                    dWhy_sum += dWhy
                    dbh_sum += dbh
                    dby_sum += dby
                
                # Average gradients over batch
                batch_size_actual = len(batch_X)
                dWxh_sum /= batch_size_actual
                dWhh_sum /= batch_size_actual
                dWhy_sum /= batch_size_actual
                dbh_sum /= batch_size_actual
                dby_sum /= batch_size_actual
                
                # Update parameters using Adam
                self.t += 1
                
                # Update Wxh
                self.m_Wxh = self.beta1 * self.m_Wxh + (1 - self.beta1) * dWxh_sum
                self.v_Wxh = self.beta2 * self.v_Wxh + (1 - self.beta2) * (dWxh_sum ** 2)
                m_hat = self.m_Wxh / (1 - self.beta1 ** self.t)
                v_hat = self.v_Wxh / (1 - self.beta2 ** self.t)
                self.Wxh -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                # Update Whh
                self.m_Whh = self.beta1 * self.m_Whh + (1 - self.beta1) * dWhh_sum
                self.v_Whh = self.beta2 * self.v_Whh + (1 - self.beta2) * (dWhh_sum ** 2)
                m_hat = self.m_Whh / (1 - self.beta1 ** self.t)
                v_hat = self.v_Whh / (1 - self.beta2 ** self.t)
                self.Whh -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                # Update Why
                self.m_Why = self.beta1 * self.m_Why + (1 - self.beta1) * dWhy_sum
                self.v_Why = self.beta2 * self.v_Why + (1 - self.beta2) * (dWhy_sum ** 2)
                m_hat = self.m_Why / (1 - self.beta1 ** self.t)
                v_hat = self.v_Why / (1 - self.beta2 ** self.t)
                self.Why -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                # Update bh
                self.m_bh = self.beta1 * self.m_bh + (1 - self.beta1) * dbh_sum
                self.v_bh = self.beta2 * self.v_bh + (1 - self.beta2) * (dbh_sum ** 2)
                m_hat = self.m_bh / (1 - self.beta1 ** self.t)
                v_hat = self.v_bh / (1 - self.beta2 ** self.t)
                self.bh -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                # Update by
                self.m_by = self.beta1 * self.m_by + (1 - self.beta1) * dby_sum
                self.v_by = self.beta2 * self.v_by + (1 - self.beta2) * (dby_sum ** 2)
                m_hat = self.m_by / (1 - self.beta1 ** self.t)
                v_hat = self.v_by / (1 - self.beta2 ** self.t)
                self.by -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                total_loss += batch_loss / batch_size_actual
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        
        return losses
    
    def predict_sequence(self, seed_sequence, num_predictions):
        """
        Predict future values given a seed sequence
        """
        predictions = []
        current_sequence = seed_sequence.tolist() 
        
        for _ in range(num_predictions):
            # Use the last 'sequence_length' values as context
            context = current_sequence[-self.sequence_length:]
            
            # Forward pass through RNN
            h = np.zeros((self.hidden_size, 1))
            for t in range(len(context)):
                x = np.array([[context[t]]])
                h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            
            # Generate prediction
            y = np.dot(self.Why, h) + self.by
            prediction = y[0, 0]
            
            predictions.append(prediction)
            current_sequence.append(prediction)
        
        return np.array(predictions)
    
    def predict_week(self, historical_data, lookback_hours=24):
        """
        Predict network traffic for the next week
        historical_data: full historical data array
        lookback_hours: number of hours to look back (default 24)
        """
        # Calculate how many 5-min intervals to look back
        lookback_intervals = lookback_hours * (60 // 5)  # 12 intervals per hour
        
        # Use the last 'lookback_intervals' as seed
        seed = historical_data[-lookback_intervals:]
        seed_normalized = self.normalize(seed)
        
        # Predict for one week (7 days * 288 intervals per day)
        num_predictions = 7 * 288
        predictions_normalized = self.predict_sequence(seed_normalized, num_predictions)
        
        # Denormalize predictions (automatically rounds to integers)
        predictions = self.denormalize(predictions_normalized)
        
        return predictions

def main():
    print("=" * 60)
    print("Network Traffic Prediction using RNN")
    print("=" * 60)
    
    # Initialize RNN
    rnn = NetworkTrafficRNN(
        hidden_size=128,
        learning_rate=0.001,
        sequence_length=24  # Look back 2 hours (24 * 5 min intervals)
    )
    
    # Load data
    print("\n1. Loading network traffic data...")
    try:
        data = rnn.load_data("network_traffic_5min_2weeks.txt")
        print(f"   Loaded {len(data)} data points (2 weeks of 5-min intervals)")
        print(f"   Data range: {np.min(data)} MB to {np.max(data)} MB")
        print(f"   Average: {np.mean(data):.1f} MB")
    except FileNotFoundError:
        print("   Error: network_traffic_5min_2weeks.txt not found!")
        print("   Please run data_generation.py first to generate the data.")
        return
    
    # Prepare training data
    print("\n2. Preparing training data...")
    X, y = rnn.prepare_training_data(data)
    print(f"   Created {len(X)} training sequences")
    print(f"   Sequence length: {rnn.sequence_length} intervals ({rnn.sequence_length * 5} minutes)")
    
    # Train the model
    print("\n3. Training RNN model...")
    print("   This may take a few minutes...")
    losses = rnn.train(X, y, epochs=50, batch_size=64, verbose=True)
    print("   Training completed!")
    
    # Predict next week
    print("\n4. Predicting network traffic for next week...")
    predictions = rnn.predict_week(data, lookback_hours=24)
    print(f"   Generated {len(predictions)} predictions (7 days)")
    
    # Save predictions to file
    output_file = "predicted_pattern.txt"
    print(f"\n5. Saving predictions to {output_file}...")
    
    with open(output_file, "w") as f:
        # Write predictions as integers
        for i, value in enumerate(predictions):
            f.write(f"{int(value)} ")  # Ensure integer output
    
    # Print summary statistics (all integers now)
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted max traffic: {np.max(predictions)} MB")
    print(f"Predicted min traffic: {np.min(predictions)} MB")
    print(f"Predicted average: {np.mean(predictions):.1f} MB")
    
    # Calculate daily averages
    print("\nDaily average predictions:")
    for day in range(7):
        start_idx = day * 288
        end_idx = (day + 1) * 288
        daily_avg = np.mean(predictions[start_idx:end_idx])
        print(f"  Day {day + 1}: {daily_avg:.1f} MB per 5-min interval")
    
    # Identify peak periods
    peak_24h_windows = []
    for i in range(0, len(predictions) - 288, 12):  # Slide every hour
        window_avg = np.mean(predictions[i:i+288])
        peak_24h_windows.append(window_avg)
    
    if peak_24h_windows:
        max_peak = max(peak_24h_windows)
        print(f"\nHighest predicted 24-hour average: {max_peak:.1f} MB per 5-min interval")
    
    print("\n" + "=" * 60)
    print(f"Predictions saved to {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()