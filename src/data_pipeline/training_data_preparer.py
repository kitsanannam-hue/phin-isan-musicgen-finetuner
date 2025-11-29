"""
Training Data Preparation Module for Thai Isan Music Transcription

This module focuses on creating high-quality training data for Thai Isan music transcription,
with special attention to accurately capturing every musical note and preserving the
unique characteristics of the 7-tone scale system and Phin lute patterns.
"""
import os
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple
import json
import pickle
from collections import defaultdict

from ..data_pipeline.feature_extraction import extract_phin_features, extract_note_events
from ..data_pipeline.thai_isan_analysis import create_detailed_transcription_report
from ..utils.constants import CQT_PARAMS, AUDIO_PARAMS, THAI_7_TONE_RATIOS


class ThaiIsanTrainingDataPreparer:
    """
    Class for preparing high-quality training data for Thai Isan music transcription.
    Focuses on accurate note capture and preserving cultural musical characteristics.
    """
    
    def __init__(self, data_dir: str = "./training_data"):
        """
        Initialize the training data preparer.
        
        Args:
            data_dir (str): Directory to store training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Subdirectories for different data types
        (self.data_dir / "audio").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)
        (self.data_dir / "labels").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)
        
    def prepare_training_data(
        self, 
        audio_paths: List[str], 
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, List[str]]:
        """
        Prepare complete training data from audio files.
        
        Args:
            audio_paths: List of paths to Thai Isan audio files
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
        
        Returns:
            Dictionary with train/validation/test splits
        """
        print(f"Preparing training data from {len(audio_paths)} audio files...")
        
        # Process each audio file to extract features and labels
        processed_data = []
        for i, audio_path in enumerate(audio_paths):
            print(f"Processing {i+1}/{len(audio_paths)}: {audio_path}")
            
            try:
                # Extract features and labels
                features, labels, metadata = self._process_audio_file(audio_path)
                
                # Save features and labels
                base_name = Path(audio_path).stem
                feature_path = self.data_dir / "features" / f"{base_name}_features.npy"
                label_path = self.data_dir / "labels" / f"{base_name}_labels.json"
                metadata_path = self.data_dir / "metadata" / f"{base_name}_metadata.json"
                
                np.save(feature_path, features)
                with open(label_path, 'w') as f:
                    json.dump(labels, f)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                # Copy audio file to training data directory
                audio_dest = self.data_dir / "audio" / Path(audio_path).name
                if not audio_dest.exists():
                    import shutil
                    shutil.copy(audio_path, audio_dest)
                
                processed_data.append({
                    'audio_path': str(audio_dest),
                    'feature_path': str(feature_path),
                    'label_path': str(label_path),
                    'metadata_path': str(metadata_path),
                    'duration': metadata.get('duration', 0)
                })
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue
        
        # Split data into train/validation/test sets
        splits = self._split_data(processed_data, validation_split, test_split)
        
        print(f"Training data preparation complete!")
        print(f"Train: {len(splits['train'])}, Validation: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def _process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Process a single audio file to extract features and labels.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Tuple of (features, labels, metadata)
        """
        # Extract CQT features optimized for Thai music
        features = extract_phin_features(audio_path)
        
        # Extract detailed note events
        note_events = extract_note_events(audio_path)
        
        # Create labels from note events
        labels = self._create_labels_from_note_events(note_events, features.shape[1])
        
        # Create metadata
        metadata = self._create_metadata(audio_path, note_events)
        
        return features, labels, metadata
    
    def _create_labels_from_note_events(self, note_events: List[Dict], time_steps: int) -> Dict:
        """
        Create training labels from note events.
        
        Args:
            note_events: List of note events
            time_steps: Number of time steps in the features
        
        Returns:
            Dictionary containing labels
        """
        # Create a time-frequency activation matrix
        # Assuming 88 keys (MIDI range from C1 to G7)
        activation_matrix = np.zeros((time_steps, 88))
        
        # Convert time to frame indices
        for event in note_events:
            start_time = event['start_time']
            end_time = event['end_time']
            pitch = event['pitch']
            velocity = event['velocity']
            
            # Convert time to frame index
            start_frame = int(start_time * (time_steps / self._get_audio_duration(note_events[0]['audio_path'] if note_events else 1.0)))
            end_frame = int(end_time * (time_steps / self._get_audio_duration(note_events[0]['audio_path'] if note_events else 1.0)))
            
            # Convert MIDI pitch to piano key index (0-87)
            key_idx = pitch - 21  # MIDI note 21 is A0, our lowest
            
            if 0 <= key_idx < 88:
                # Apply velocity as activation value
                for frame in range(max(0, start_frame), min(time_steps, end_frame)):
                    activation_matrix[frame, key_idx] = velocity / 127.0  # Normalize to [0, 1]
        
        # Create onset labels (when notes start)
        onset_labels = np.zeros(time_steps)
        for event in note_events:
            start_time = event['start_time']
            start_frame = int(start_time * (time_steps / self._get_audio_duration(note_events[0]['audio_path'] if note_events else 1.0)))
            if 0 <= start_frame < time_steps:
                onset_labels[start_frame] = 1
        
        return {
            'activation_matrix': activation_matrix.tolist(),  # Convert to list for JSON serialization
            'onset_labels': onset_labels.tolist(),
            'note_events': note_events
        }
    
    def _create_metadata(self, audio_path: str, note_events: List[Dict]) -> Dict:
        """
        Create metadata for the audio file.
        
        Args:
            audio_path: Path to the audio file
            note_events: List of note events
        
        Returns:
            Metadata dictionary
        """
        # Load audio to get duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # Analyze the audio for Thai Isan characteristics
        analysis_report = create_detailed_transcription_report(audio_path)
        
        # Calculate musical statistics
        if note_events:
            pitches = [event['pitch'] for event in note_events]
            velocities = [event['velocity'] for event in note_events]
            
            stats = {
                'note_count': len(note_events),
                'pitch_range': max(pitches) - min(pitches) if pitches else 0,
                'avg_velocity': sum(velocities) / len(velocities) if velocities else 0,
                'note_density': len(note_events) / duration if duration > 0 else 0,
                'pitch_entropy': self._calculate_pitch_entropy(pitches),
                'thai_scale_adherence': analysis_report['thai_scale_adherence']
            }
        else:
            stats = {
                'note_count': 0,
                'pitch_range': 0,
                'avg_velocity': 0,
                'note_density': 0,
                'pitch_entropy': 0,
                'thai_scale_adherence': 0
            }
        
        return {
            'audio_path': audio_path,
            'duration': duration,
            'sample_rate': sr,
            'note_statistics': stats,
            'analysis_report': {
                'thai_scale_adherence': analysis_report['thai_scale_adherence'],
                'note_count': analysis_report['note_count']
            }
        }
    
    def _calculate_pitch_entropy(self, pitches: List[int]) -> float:
        """
        Calculate entropy of pitch distribution.
        
        Args:
            pitches: List of MIDI pitch values
        
        Returns:
            Entropy value
        """
        if not pitches:
            return 0
        
        unique_pitches, counts = np.unique(pitches, return_counts=True)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]  # Remove zero probabilities
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of an audio file.
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Duration in seconds
        """
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    
    def _split_data(self, data: List[Dict], validation_split: float, test_split: float) -> Dict[str, List[Dict]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: List of data entries
            validation_split: Fraction for validation
            test_split: Fraction for test
        
        Returns:
            Dictionary with train/validation/test splits
        """
        # Sort by duration to ensure similar distributions
        sorted_data = sorted(data, key=lambda x: x['duration'])
        
        n_total = len(sorted_data)
        n_test = int(n_total * test_split)
        n_val = int(n_total * validation_split)
        
        # Split the data
        test_data = sorted_data[:n_test]
        val_data = sorted_data[n_test:n_test + n_val]
        train_data = sorted_data[n_test + n_val:]
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def create_augmented_data(self, audio_paths: List[str], augmentations: List[str] = None) -> List[str]:
        """
        Create augmented versions of audio files to increase training data diversity.
        
        Args:
            audio_paths: List of original audio file paths
            augmentations: List of augmentation techniques to apply
        
        Returns:
            List of paths to augmented audio files
        """
        if augmentations is None:
            augmentations = ['time_stretch', 'pitch_shift', 'add_noise']
        
        augmented_paths = []
        
        for audio_path in audio_paths:
            y, sr = librosa.load(audio_path, sr=None)
            
            base_name = Path(audio_path).stem
            
            for aug_type in augmentations:
                try:
                    if aug_type == 'time_stretch':
                        # Time stretching (preserves pitch)
                        y_aug = librosa.effects.time_stretch(y, rate=1.1)
                    elif aug_type == 'pitch_shift':
                        # Pitch shifting (preserves tempo)
                        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
                    elif aug_type == 'add_noise':
                        # Add slight noise
                        noise = np.random.normal(0, 0.001, y.shape)
                        y_aug = y + noise
                        y_aug = np.clip(y_aug, -1.0, 1.0)  # Ensure values stay in range
                    else:
                        print(f"Unknown augmentation type: {aug_type}")
                        continue
                    
                    # Save augmented audio
                    aug_path = self.data_dir / "audio" / f"{base_name}_{aug_type}.wav"
                    librosa.output.write_wav(str(aug_path), y_aug, sr)
                    augmented_paths.append(str(aug_path))
                    
                except Exception as e:
                    print(f"Error applying {aug_type} to {audio_path}: {str(e)}")
                    continue
        
        return augmented_paths
    
    def save_dataset_info(self, splits: Dict[str, List[Dict]], dataset_name: str = "thai_isan_dataset"):
        """
        Save information about the dataset.
        
        Args:
            splits: Dictionary with train/validation/test splits
            dataset_name: Name of the dataset
        """
        dataset_info = {
            'name': dataset_name,
            'total_samples': sum(len(split) for split in splits.values()),
            'train_samples': len(splits['train']),
            'validation_samples': len(splits['validation']),
            'test_samples': len(splits['test']),
            'feature_shape': (CQT_PARAMS['n_bins'], None),  # Time dimension varies
            'label_type': 'piano_roll_with_onsets',
            'description': 'Training dataset for Thai Isan music transcription with focus on 7-tone scale system',
            'creation_date': str(np.datetime64('now'))
        }
        
        info_path = self.data_dir / f"{dataset_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset info saved to: {info_path}")
    
    def get_data_statistics(self, splits: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate statistics about the prepared data.
        
        Args:
            splits: Dictionary with train/validation/test splits
        
        Returns:
            Dictionary with data statistics
        """
        stats = {}
        
        for split_name, split_data in splits.items():
            durations = [item['duration'] for item in split_data]
            stats[split_name] = {
                'count': len(split_data),
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0
            }
        
        # Overall statistics
        all_durations = []
        for split_data in splits.values():
            all_durations.extend([item['duration'] for item in split_data])
        
        stats['overall'] = {
            'total_samples': sum(len(split_data) for split_data in splits.values()),
            'total_duration': sum(all_durations),
            'avg_duration': np.mean(all_durations) if all_durations else 0,
            'min_duration': min(all_durations) if all_durations else 0,
            'max_duration': max(all_durations) if all_durations else 0
        }
        
        return stats


def prepare_thai_isan_dataset(
    audio_directory: str, 
    output_directory: str = "./thai_isan_training_data",
    validation_split: float = 0.2,
    test_split: float = 0.1
) -> Dict[str, List[str]]:
    """
    Convenience function to prepare a complete Thai Isan music transcription dataset.
    
    Args:
        audio_directory: Directory containing Thai Isan audio files
        output_directory: Directory to store the prepared dataset
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
    
    Returns:
        Dictionary with train/validation/test splits
    """
    # Find all audio files in the directory
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
    audio_paths = []
    
    for ext in audio_extensions:
        audio_paths.extend(Path(audio_directory).glob(f"*{ext}"))
        audio_paths.extend(Path(audio_directory).glob(f"**/*{ext}"))  # Include subdirectories
    
    audio_paths = [str(path) for path in audio_paths]
    
    if not audio_paths:
        raise ValueError(f"No audio files found in {audio_directory}")
    
    print(f"Found {len(audio_paths)} audio files")
    
    # Initialize the data preparer
    preparer = ThaiIsanTrainingDataPreparer(data_dir=output_directory)
    
    # Prepare the training data
    splits = preparer.prepare_training_data(
        audio_paths,
        validation_split=validation_split,
        test_split=test_split
    )
    
    # Save dataset information
    preparer.save_dataset_info(splits)
    
    # Print statistics
    stats = preparer.get_data_statistics(splits)
    print("\nDataset Statistics:")
    print(f"Overall: {stats['overall']['total_samples']} samples, "
          f"{stats['overall']['total_duration']:.2f}s total duration")
    print(f"Train: {stats['train']['count']} samples, "
          f"{stats['train']['total_duration']:.2f}s total duration")
    print(f"Validation: {stats['validation']['count']} samples, "
          f"{stats['validation']['total_duration']:.2f}s total duration")
    print(f"Test: {stats['test']['count']} samples, "
          f"{stats['test']['total_duration']:.2f}s total duration")
    
    return splits


# Example usage
if __name__ == "__main__":
    print("Thai Isan Music Training Data Preparation")
    print("=" * 45)
    print("This module prepares high-quality training data for Thai Isan music transcription")
    print("with focus on accurate note capture and preservation of 7-tone scale characteristics.")
    
    # Example of how to use the preparer (commented out to avoid errors without real data)
    """
    # Example usage:
    audio_dir = "./raw_thai_isan_audio"  # Directory with raw audio files
    splits = prepare_thai_isan_dataset(
        audio_directory=audio_dir,
        output_directory="./prepared_thai_isan_data",
        validation_split=0.2,
        test_split=0.1
    )
    
    print(f"Dataset prepared with {len(splits['train'])} training samples")
    """