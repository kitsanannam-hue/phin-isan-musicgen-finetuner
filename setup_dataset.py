#!/usr/bin/env python3
"""
Thai Music AI Dataset Setup Script
Based on the Thai_Music_AI_Dataset_Project and Thai Isan Music Transcription System

This script sets up the complete dataset configuration for note/music processing
combining both projects for comprehensive Thai music AI training.
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ThaiMusicConfig:
    """Configuration for Thai Music AI Dataset"""
    dataset_name: str = "thai_music_ai_dataset"
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 128
    n_cqt: int = 84  # 7 octaves * 12 semitones for CQT
    thai_scale_degrees: int = 7  # Thai 7-tone system
    sequence_length: int = 512
    batch_size: int = 32
    
    # Thai-specific configurations
    thai_regions: List[str] = None
    thai_scales: List[str] = None
    melodic_patterns: List[str] = None
    
    def __post_init__(self):
        if self.thai_regions is None:
            self.thai_regions = ["central", "isan", "northern", "southern"]
        if self.thai_scales is None:
            self.thai_scales = [
                "lai_yai", "lai_noi", "lai_se", "lai_sutsanaen",
                "lai_pong_sai", "lai_soi", "thai_pentatonic"
            ]
        if self.melodic_patterns is None:
            self.melodic_patterns = [
                "pattern_12352", "pattern_532121", "pattern_232165",
                "pattern_561216", "pattern_13531", "pattern_24642"
            ]

class ThaiMusicDatasetSetup:
    """Setup and configure Thai Music AI Dataset"""
    
    def __init__(self, config: ThaiMusicConfig):
        self.config = config
        self.base_path = Path("/home/user/webapp")
        self.dataset_path = self.base_path / "dataset"
        self.audio_path = self.dataset_path / "audio"
        self.metadata_path = self.dataset_path / "metadata"
        self.features_path = self.dataset_path / "features"
        self.models_path = self.dataset_path / "models"
        
    def create_directory_structure(self):
        """Create the complete dataset directory structure"""
        logger.info("Creating dataset directory structure...")
        
        directories = [
            self.dataset_path,
            self.audio_path,
            self.metadata_path,
            self.features_path,
            self.models_path,
            self.audio_path / "raw",
            self.audio_path / "processed",
            self.audio_path / "synthetic",
            self.features_path / "spectrograms",
            self.features_path / "cqt",
            self.features_path / "mfcc",
            self.metadata_path / "annotations",
            self.metadata_path / "train",
            self.metadata_path / "val", 
            self.metadata_path / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def integrate_existing_data(self):
        """Integrate existing data from the extracted archive"""
        logger.info("Integrating existing data...")
        
        # Copy raw audio files
        raw_audio_src = self.base_path / "raw_audio"
        if raw_audio_src.exists():
            logger.info("Copying raw audio files...")
            for audio_file in raw_audio_src.rglob("*.wav"):
                if audio_file.is_file():
                    dst = self.audio_path / "raw" / audio_file.name
                    shutil.copy2(audio_file, dst)
                    logger.info(f"Copied: {audio_file.name}")
        
        # Copy synthetic audio files
        synthetic_src = self.base_path / "synthetic_audio"
        if synthetic_src.exists():
            logger.info("Copying synthetic audio files...")
            for audio_file in synthetic_src.rglob("*.wav"):
                if audio_file.is_file():
                    dst = self.audio_path / "synthetic" / audio_file.name
                    shutil.copy2(audio_file, dst)
                    logger.info(f"Copied: {audio_file.name}")
        
        # Copy spectrograms
        spectrograms_src = self.base_path / "spectrograms"
        if spectrograms_src.exists():
            logger.info("Copying spectrograms...")
            for spec_file in spectrograms_src.rglob("*.png"):
                if spec_file.is_file():
                    dst = self.features_path / "spectrograms" / spec_file.name
                    shutil.copy2(spec_file, dst)
                    logger.info(f"Copied: {spec_file.name}")
        
        # Copy analysis results
        analysis_files = [
            (self.base_path / "analysis_results.json", self.metadata_path / "analysis_results.json"),
            (self.base_path / "synthetic_metadata.json", self.metadata_path / "synthetic_metadata.json")
        ]
        
        for src, dst in analysis_files:
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied analysis file: {src.name}")
    
    def create_dataset_metadata(self):
        """Create comprehensive dataset metadata"""
        logger.info("Creating dataset metadata...")
        
        dataset_info = {
            "dataset_name": self.config.dataset_name,
            "version": "1.0",
            "description": "Thai Music AI Dataset combining Thai Isan Lute transcription and Thai Music AI Dataset project",
            "created_date": "2025-11-29",
            "audio_config": {
                "sample_rate": self.config.sample_rate,
                "hop_length": self.config.hop_length,
                "n_fft": self.config.n_fft,
                "n_mels": self.config.n_mels,
                "n_cqt": self.config.n_cqt
            },
            "thai_music_config": {
                "scale_degrees": self.config.thai_scale_degrees,
                "regions": self.config.thai_regions,
                "scales": self.config.thai_scales,
                "melodic_patterns": self.config.melodic_patterns
            },
            "dataset_stats": self._calculate_dataset_stats(),
            "file_structure": self._get_file_structure()
        }
        
        # Save main dataset info
        with open(self.metadata_path / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info("Dataset metadata created successfully")
    
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate basic dataset statistics"""
        stats = {
            "total_audio_files": 0,
            "total_spectrograms": 0,
            "regions": {},
            "scales": {},
            "duration_stats": {}
        }
        
        # Count audio files
        for audio_dir in ["raw", "synthetic", "processed"]:
            audio_path = self.audio_path / audio_dir
            if audio_path.exists():
                audio_files = list(audio_path.rglob("*.wav"))
                stats["total_audio_files"] += len(audio_files)
                logger.info(f"Found {len(audio_files)} {audio_dir} audio files")
        
        # Count spectrograms
        spec_path = self.features_path / "spectrograms"
        if spec_path.exists():
            spec_files = list(spec_path.rglob("*.png"))
            stats["total_spectrograms"] = len(spec_files)
            logger.info(f"Found {len(spec_files)} spectrograms")
        
        # Load existing analysis results if available
        analysis_file = self.metadata_path / "analysis_results.json"
        if analysis_file.exists():
            try:
                with open(analysis_file, "r") as f:
                    analysis_data = json.load(f)
                    if "files" in analysis_data:
                        stats["analysis_results"] = len(analysis_data["files"])
                        logger.info(f"Found {len(analysis_data['files'])} analyzed files")
            except Exception as e:
                logger.warning(f"Could not load analysis results: {e}")
        
        return stats
    
    def _get_file_structure(self) -> Dict:
        """Get the complete file structure"""
        structure = {}
        
        def scan_directory(path: Path, structure: Dict, base_path: Path):
            relative_path = path.relative_to(base_path)
            if path.is_file():
                return
            
            structure[str(relative_path)] = {
                "type": "directory",
                "contents": []
            }
            
            for item in path.iterdir():
                if item.is_file():
                    structure[str(relative_path)]["contents"].append({
                        "name": item.name,
                        "type": "file",
                        "size": item.stat().st_size
                    })
                elif item.is_dir():
                    structure[str(relative_path)]["contents"].append({
                        "name": item.name,
                        "type": "directory"
                    })
                    scan_directory(item, structure, base_path)
        
        scan_directory(self.dataset_path, structure, self.base_path)
        return structure
    
    def create_training_config(self):
        """Create training configuration files"""
        logger.info("Creating training configuration...")
        
        # Training config
        training_config = {
            "model_type": "thai_music_transformer",
            "sequence_length": self.config.sequence_length,
            "batch_size": self.config.batch_size,
            "learning_rate": 0.0001,
            "num_epochs": 100,
            "validation_split": 0.15,
            "test_split": 0.15,
            "early_stopping_patience": 10,
            "model_checkpoint_dir": str(self.models_path / "checkpoints"),
            "log_dir": str(self.models_path / "logs")
        }
        
        # Feature extraction config
        feature_config = {
            "audio_features": {
                "spectrogram": {
                    "enabled": True,
                    "n_mels": self.config.n_mels,
                    "hop_length": self.config.hop_length,
                    "n_fft": self.config.n_fft
                },
                "cqt": {
                    "enabled": True,
                    "n_bins": self.config.n_cqt,
                    "hop_length": self.config.hop_length,
                    "fmin": 32.7  # C1
                },
                "mfcc": {
                    "enabled": True,
                    "n_mfcc": 13,
                    "hop_length": self.config.hop_length
                }
            },
            "thai_specific": {
                "scale_quantization": True,
                "pattern_extraction": True,
                "rhythmic_analysis": True
            }
        }
        
        # Save configs
        with open(self.metadata_path / "training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)
        
        with open(self.metadata_path / "feature_config.json", "w") as f:
            json.dump(feature_config, f, indent=2)
        
        logger.info("Training configuration created successfully")
    
    def create_dataset_split(self):
        """Create train/val/test splits for the dataset"""
        logger.info("Creating dataset splits...")
        
        # Get all audio files
        all_files = []
        for audio_dir in ["raw", "synthetic"]:
            audio_path = self.audio_path / audio_dir
            if audio_path.exists():
                all_files.extend(list(audio_path.rglob("*.wav")))
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Save splits
        splits = {
            "train": [str(f) for f in train_files],
            "validation": [str(f) for f in val_files],
            "test": [str(f) for f in test_files]
        }
        
        with open(self.metadata_path / "dataset_splits.json", "w") as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Dataset splits created: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    def setup_complete_dataset(self):
        """Run the complete dataset setup process"""
        logger.info("Starting complete Thai Music AI Dataset setup...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Integrate existing data
        self.integrate_existing_data()
        
        # Create metadata
        self.create_dataset_metadata()
        
        # Create training config
        self.create_training_config()
        
        # Create dataset splits
        self.create_dataset_split()
        
        logger.info("✅ Thai Music AI Dataset setup completed successfully!")
        logger.info(f"Dataset location: {self.dataset_path}")
        logger.info(f"Total audio files: {self._calculate_dataset_stats()['total_audio_files']}")
        logger.info(f"Total spectrograms: {self._calculate_dataset_stats()['total_spectrograms']}")

def main():
    """Main function to setup the dataset"""
    # Create configuration
    config = ThaiMusicConfig()
    
    # Setup dataset
    setup = ThaiMusicDatasetSetup(config)
    setup.setup_complete_dataset()

if __name__ == "__main__":
    main()