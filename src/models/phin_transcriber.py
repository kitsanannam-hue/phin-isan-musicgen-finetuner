"""
PhinTranscriber model architecture for Thai Isan Lute (Phin) Music Transcription

Implements a CNN-RNN-Attention architecture optimized for transcribing 
Thai 7-tone scale music played on the Phin lute.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhinTranscriber(nn.Module):
    """
    CNN-RNN-Attention model for transcribing Thai Isan Phin lute music.
    Designed to handle the unique characteristics of the 7-tone Thai scale system.
    """
    
    def __init__(self, n_freq_bins=120, n_time_steps=500, n_classes=88, 
                 cnn_channels=[64, 128, 256], rnn_units=256, dropout_rate=0.3):
        """
        Initialize the PhinTranscriber model.
        
        Args:
            n_freq_bins (int): Number of frequency bins in the input CQT
            n_time_steps (int): Number of time steps in the input CQT
            n_classes (int): Number of output classes (MIDI notes)
            cnn_channels (list): Number of channels for each CNN layer
            rnn_units (int): Number of units in RNN layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(PhinTranscriber, self).__init__()
        
        self.n_freq_bins = n_freq_bins
        self.n_time_steps = n_time_steps
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = 1  # Input is 1 channel (magnitude spectrogram)
        
        for i, out_channels in enumerate(cnn_channels):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample time and frequency
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the size after CNN layers
        # Each MaxPool2d reduces both dimensions by half
        n_cnn_layers = len(cnn_channels)
        cnn_out_freq = n_freq_bins // (2 ** n_cnn_layers)
        cnn_out_time = n_time_steps // (2 ** n_cnn_layers)
        rnn_input_size = cnn_out_freq * cnn_channels[-1]
        
        # RNN layers for temporal modeling
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True  # Bidirectional LSTM for context in both directions
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_units * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output layer for note classification
        self.output_layer = nn.Linear(rnn_units * 2, n_classes)  # *2 for bidirectional
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, freq_bins, time_steps)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, time_steps, n_classes)
        """
        batch_size = x.size(0)
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, freq_bins, time_steps)
        
        # Apply CNN layers
        x = self.cnn(x)  # (batch, channels, freq_bins_reduced, time_steps_reduced)
        
        # Reshape for RNN: (batch, time_steps, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, freq_bins)
        batch_size, time_steps, channels, freq_bins = x.size()
        x = x.contiguous().view(batch_size, time_steps, channels * freq_bins)
        
        # Apply RNN layers
        rnn_out, _ = self.rnn(x)  # (batch, time_steps, rnn_units*2)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)  # (batch, time_steps, rnn_units*2)
        
        # Apply dropout
        attn_out = self.dropout(attn_out)
        
        # Apply output layer
        output = self.output_layer(attn_out)  # (batch, time_steps, n_classes)
        
        # Apply sigmoid activation to get note activation probabilities
        output = torch.sigmoid(output)
        
        return output
    
    def transcribe(self, cqt_features):
        """
        Transcribe CQT features to note events.
        
        Args:
            cqt_features (np.ndarray): CQT spectrogram from feature_extraction module
        
        Returns:
            dict: Transcription result with note events and metadata
        """
        # Convert numpy array to torch tensor
        if isinstance(cqt_features, np.ndarray):
            cqt_tensor = torch.tensor(cqt_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        else:
            cqt_tensor = cqt_features.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            output = self(cqt_tensor)
        
        # Convert to numpy for further processing
        activations = output.squeeze(0).numpy()  # Remove batch dimension
        
        # Extract note events from activations
        note_events = self.extract_note_events_from_activations(activations)
        
        return {
            'note_events': note_events,
            'activations': activations,
            'model_params': {
                'n_freq_bins': self.n_freq_bins,
                'n_time_steps': self.n_time_steps,
                'n_classes': self.n_classes
            }
        }
    
    def extract_note_events_from_activations(self, activations, threshold=0.5):
        """
        Extract note events from the model's output activations.
        
        Args:
            activations (np.ndarray): Model output activations (time_steps, n_classes)
            threshold (float): Activation threshold for note detection
        
        Returns:
            list: List of note events (start_time, end_time, pitch, velocity)
        """
        note_events = []
        
        # For each MIDI note (class)
        for pitch_idx in range(activations.shape[1]):
            # Find time steps where this note is active
            active_frames = np.where(activations[:, pitch_idx] > threshold)[0]
            
            if len(active_frames) == 0:
                continue
            
            # Group consecutive frames into note events
            start_frame = active_frames[0]
            for i in range(1, len(active_frames)):
                if active_frames[i] != active_frames[i-1] + 1:  # Gap in activation
                    # End current note
                    end_frame = active_frames[i-1]
                    velocity = np.mean(activations[start_frame:end_frame+1, pitch_idx])
                    
                    # Convert frames to time (assuming 10ms per frame)
                    start_time = start_frame * 0.01
                    end_time = end_frame * 0.01
                    
                    note_events.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'pitch': pitch_idx + 21,  # MIDI note number (A0 = 21)
                        'velocity': int(velocity * 127)  # Convert to MIDI velocity (0-127)
                    })
                    
                    # Start new note
                    start_frame = active_frames[i]
            
            # Handle the last note
            end_frame = active_frames[-1]
            velocity = np.mean(activations[start_frame:end_frame+1, pitch_idx])
            start_time = start_frame * 0.01
            end_time = end_frame * 0.01
            
            note_events.append({
                'start_time': start_time,
                'end_time': end_time,
                'pitch': pitch_idx + 21,  # MIDI note number (A0 = 21)
                'velocity': int(velocity * 127)  # Convert to MIDI velocity (0-127)
            })
        
        return note_events


class CQTDataset(torch.utils.data.Dataset):
    """
    Dataset class for CQT features to be used with PyTorch DataLoader.
    """
    
    def __init__(self, cqt_features_list, labels_list=None):
        """
        Initialize the dataset.
        
        Args:
            cqt_features_list (list): List of CQT features (numpy arrays)
            labels_list (list): Optional list of labels for supervised learning
        """
        self.cqt_features = cqt_features_list
        self.labels = labels_list
    
    def __len__(self):
        return len(self.cqt_features)
    
    def __getitem__(self, idx):
        features = self.cqt_features[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(features, dtype=torch.float32)


def create_model():
    """
    Factory function to create a PhinTranscriber model with default parameters.
    
    Returns:
        PhinTranscriber: Initialized model instance
    """
    return PhinTranscriber()


# Example usage
if __name__ == "__main__":
    # Create model instance
    model = create_model()
    
    # Print model summary
    print("PhinTranscriber Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example forward pass with dummy data
    dummy_input = torch.randn(1, 120, 500)  # (batch, freq_bins, time_steps)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")