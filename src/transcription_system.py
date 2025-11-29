"""
Comprehensive Thai Isan Music Transcription System

This module integrates all components for creating high-quality training data
for Thai Isan music transcription, with special focus on accurately capturing
every musical note and preserving the unique characteristics of the 7-tone
scale system and Phin lute patterns.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
from typing import List, Dict, Tuple, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.setup.environment import setup_environment
from src.data_pipeline.download import download_youtube_audio
from src.data_pipeline.feature_extraction import extract_phin_features
from src.data_pipeline.thai_isan_analysis import create_detailed_transcription_report
from src.data_pipeline.training_data_preparer import ThaiIsanTrainingDataPreparer
from src.models.phin_transcriber import PhinTranscriber
from src.evaluation.transcription_eval import evaluate_transcription, evaluate_midi_accuracy
from src.utils.constants import CQT_PARAMS, THAI_7_TONE_RATIOS, TRANSCRIPTION_PARAMS


class ThaiIsanTranscriptionSystem:
    """
    Complete system for Thai Isan music transcription with focus on 
    accurate note capture and cultural preservation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the transcription system.
        
        Args:
            model_path: Optional path to a pre-trained model
        """
        # Setup environment
        self.setup_info = setup_environment()
        
        # Initialize model
        self.model = PhinTranscriber()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from: {model_path}")
        
        # Initialize training data preparer
        self.data_preparer = ThaiIsanTrainingDataPreparer()
        
        print("Thai Isan Transcription System initialized successfully")
    
    def download_thai_isan_audio(self, urls: List[str], output_dir: str = "./audio_sources") -> List[str]:
        """
        Download Thai Isan music audio from YouTube URLs.
        
        Args:
            urls: List of YouTube URLs
            output_dir: Directory to save audio files
        
        Returns:
            List of paths to downloaded audio files
        """
        audio_paths = []
        
        for i, url in enumerate(urls):
            print(f"Downloading audio {i+1}/{len(urls)}: {url}")
            try:
                audio_path = download_youtube_audio(url, output_dir, f"thai_isan_{i+1:03d}")
                audio_paths.append(audio_path)
                print(f"Downloaded: {audio_path}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                continue
        
        return audio_paths
    
    def preprocess_audio_for_training(self, audio_paths: List[str]) -> Dict[str, List[str]]:
        """
        Preprocess audio files for training data preparation.
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            Dictionary with train/validation/test splits
        """
        return self.data_preparer.prepare_training_data(audio_paths)
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Transcription result
        """
        print(f"Transcribing: {audio_path}")
        
        # Extract features optimized for Thai music
        features = extract_phin_features(audio_path)
        print(f"Extracted features with shape: {features.shape}")
        
        # Run transcription
        result = self.model.transcribe(features)
        print(f"Transcription completed with {len(result['note_events'])} note events")
        
        return result
    
    def analyze_audio_cultural_characteristics(self, audio_path: str) -> Dict:
        """
        Analyze cultural characteristics of Thai Isan music.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Analysis report
        """
        return create_detailed_transcription_report(audio_path)
    
    def evaluate_transcription_quality(self, reference_midi_path: str, predicted_midi_path: str) -> Dict:
        """
        Evaluate the quality of a transcription against a reference.
        
        Args:
            reference_midi_path: Path to reference MIDI
            predicted_midi_path: Path to predicted MIDI
        
        Returns:
            Evaluation metrics
        """
        return evaluate_transcription(reference_midi_path, predicted_midi_path)
    
    def create_training_dataset(
        self, 
        audio_directory: str, 
        output_directory: str = "./thai_isan_training_data",
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Create a complete training dataset from audio files.
        
        Args:
            audio_directory: Directory containing Thai Isan audio files
            output_directory: Directory to store the prepared dataset
            validation_split: Fraction for validation
            test_split: Fraction for test
        
        Returns:
            Dictionary with train/validation/test splits
        """
        return self.data_preparer.prepare_training_data(
            self._get_audio_files(audio_directory),
            validation_split=validation_split,
            test_split=test_split
        )
    
    def _get_audio_files(self, directory: str) -> List[str]:
        """
        Get all audio files from a directory.
        
        Args:
            directory: Directory to search for audio files
        
        Returns:
            List of audio file paths
        """
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
        audio_paths = []
        
        for ext in audio_extensions:
            audio_paths.extend(Path(directory).glob(f"*{ext}"))
            audio_paths.extend(Path(directory).glob(f"**/*{ext}"))  # Include subdirectories
        
        return [str(path) for path in audio_paths]
    
    def save_transcription_as_midi(self, transcription_result: Dict, output_path: str):
        """
        Save transcription result as a MIDI file.
        
        Args:
            transcription_result: Result from transcribe_audio
            output_path: Path to save the MIDI file
        """
        import pretty_midi
        
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        
        # Create an instrument (Acoustic Grand Piano)
        instrument = pretty_midi.Instrument(program=0)
        
        # Add notes from transcription result
        for note_event in transcription_result['note_events']:
            start_time = note_event['start_time']
            end_time = note_event['end_time']
            pitch = int(note_event['pitch'])
            velocity = int(note_event['velocity'])
            
            # Create note object
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            
            # Add note to instrument
            instrument.notes.append(note)
        
        # Add instrument to MIDI object
        midi.instruments.append(instrument)
        
        # Write MIDI file
        midi.write(output_path)
        print(f"MIDI file saved to: {output_path}")
    
    def train_model(
        self, 
        train_data_dir: str, 
        validation_data_dir: str, 
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 0.001
    ):
        """
        Train the transcription model on prepared data.
        
        Args:
            train_data_dir: Directory with training data
            validation_data_dir: Directory with validation data
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        print("Starting model training...")
        
        # This is a simplified training loop - in practice, you'd want to implement
        # a more sophisticated training procedure with proper data loading, etc.
        
        # Set model to training mode
        self.model.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss()  # Binary cross-entropy for note activation
        
        # For demonstration purposes, we'll use random data
        # In practice, you'd load your prepared training data
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        
        for epoch in range(epochs):
            # This is a placeholder - actual implementation would load batches
            # from your training data
            
            # Simulate a training step
            dummy_features = torch.randn(batch_size, CQT_PARAMS['n_bins'], 500)
            dummy_targets = torch.zeros(batch_size, 500, 88)  # 88 keys piano roll
            
            # Forward pass
            outputs = self.model(dummy_features)
            
            # Calculate loss (this would compare outputs with actual targets)
            loss = criterion(outputs, dummy_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("Training completed!")
    
    def generate_detailed_report(self, audio_path: str) -> Dict:
        """
        Generate a detailed report about an audio file including cultural analysis.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Comprehensive analysis report
        """
        print(f"Generating detailed report for: {audio_path}")
        
        # Transcribe the audio
        transcription_result = self.transcribe_audio(audio_path)
        
        # Analyze cultural characteristics
        cultural_analysis = self.analyze_audio_cultural_characteristics(audio_path)
        
        # Create report
        report = {
            'audio_path': audio_path,
            'transcription_result': transcription_result,
            'cultural_analysis': cultural_analysis,
            'note_count': len(transcription_result['note_events']),
            'thai_scale_adherence': cultural_analysis['thai_scale_adherence'],
            'duration': cultural_analysis['pattern_analysis']['rhythmic_features']['mean_ioi'] * len(transcription_result['note_events']) if transcription_result['note_events'] else 0
        }
        
        return report


def main():
    """
    Main function demonstrating the Thai Isan Transcription System.
    """
    print("Thai Isan Music Transcription System")
    print("=" * 40)
    
    # Initialize the system
    system = ThaiIsanTranscriptionSystem()
    
    print("\nSystem components initialized:")
    print(f"- Environment setup: {system.setup_info['dependencies_installed']}")
    print(f"- GPU available: {system.setup_info['gpu_available']}")
    print(f"- Model initialized: {type(system.model).__name__}")
    
    print("\nSystem capabilities:")
    print("1. Download Thai Isan music from YouTube")
    print("2. Extract culturally-accurate features using CQT")
    print("3. Transcribe music with focus on 7-tone scale")
    print("4. Analyze cultural characteristics of Phin lute patterns")
    print("5. Prepare high-quality training data")
    print("6. Evaluate transcription quality")
    
    print("\nFor actual usage:")
    print("1. Download Thai Isan audio files or YouTube URLs")
    print("2. Use system.transcribe_audio() for transcription")
    print("3. Use system.analyze_audio_cultural_characteristics() for analysis")
    print("4. Use system.create_training_dataset() for training data preparation")
    
    return system


if __name__ == "__main__":
    system = main()