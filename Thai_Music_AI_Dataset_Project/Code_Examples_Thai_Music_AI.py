"""
Thai Music AI Dataset Creation - Code Examples
Based on dissertation: "Jazz Orchestra Portraits of Thailand" by Tanarat Chaichana (2022)

This module provides practical code examples for:
1. MIDI processing and feature extraction
2. Thai music pattern recognition
3. Dataset preparation for AI training
4. Model implementations
"""

import os
import json
import numpy as np
import music21 as m21
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Part 1: Data Structures for Thai Music
# ============================================================================

class ThaiRegion(Enum):
    """Thai music regions as identified in dissertation"""
    CENTRAL = "central"  # Dontri Thai Doem (Thai Classical)
    ISAN = "isan"        # Northeast Thailand
    NORTHERN = "northern" # Northern Thailand
    SOUTHERN = "southern" # Southern Thailand

class ThaiScale(Enum):
    """Thai musical scales from dissertation Chapter 3-6"""
    # Isan scales (Chapter 4)
    LAI_THANG_YAO_YAI = ("lai_yai", ["A", "C", "D", "E", "G"])
    LAI_THANG_YAO_NOI = ("lai_noi", ["D", "F", "G", "A", "C"])
    LAI_THANG_YAO_SE = ("lai_se", ["E", "G", "A", "B", "D"])
    LAI_THANG_SAN_SUTSANAEN = ("lai_sutsanaen", ["C", "D", "E", "G", "A"])
    LAI_THANG_SAN_PONG_SAI = ("lai_pong_sai", ["F", "G", "A", "C", "D"])
    LAI_THANG_SAN_SOI = ("lai_soi", ["G", "A", "B", "D", "E"])
    
    # Thai Classical pentatonic (Chapter 3)
    THAI_PENTATONIC = ("thai_pentatonic", ["C", "D", "E", "G", "A"])
    
    def __init__(self, scale_name: str, notes: List[str]):
        self.scale_name = scale_name
        self.notes = notes

@dataclass
class ThaiMelodicPattern:
    """Thai melodic patterns (ลายเพลง) from dissertation"""
    name: str
    pattern: List[int]  # Scale degrees
    region: ThaiRegion
    description: str
    
    # Common patterns from dissertation Figure 3.5
    PATTERN_1 = [1, 2, 3, 5, 2]
    PATTERN_2 = [5, 3, 2, 1, 2, 3]
    PATTERN_3 = [2, 3, 2, 1, 6, 5]  # 6 is lower octave
    PATTERN_4 = [5, 6, 1, 2, 1, 6]

@dataclass
class ThaiRhythmicPattern:
    """Thai rhythmic patterns from dissertation"""
    name: str
    chan: str  # chan dio, song chan, sam chan
    pattern: List[float]  # Note durations
    cycle_length: int  # In measures
    region: ThaiRegion

class NathapCycle(Enum):
    """Nathap cycles from Chapter 3"""
    PROPKAI_SAM_CHAN = ("propkai_sam_chan", 8)  # 8-bar cycle
    PROPKAI_SONG_CHAN = ("propkai_song_chan", 4)  # 4-bar cycle
    PROPKAI_CHAN_DIO = ("propkai_chan_dio", 2)  # 2-bar cycle
    SONG_MAI_SAM_CHAN = ("song_mai_sam_chan", 4)
    SONG_MAI_SONG_CHAN = ("song_mai_song_chan", 2)
    SONG_MAI_CHAN_DIO = ("song_mai_chan_dio", 1)
    
    def __init__(self, cycle_name: str, bars: int):
        self.cycle_name = cycle_name
        self.bars = bars

# ============================================================================
# Part 2: MIDI Processing and Feature Extraction
# ============================================================================

class ThaiMusicMIDIProcessor:
    """Process MIDI files from dissertation with Thai music context"""
    
    def __init__(self):
        self.thai_scales = self._load_thai_scales()
        self.patterns = self._load_melodic_patterns()
        
    def _load_thai_scales(self) -> Dict:
        """Load Thai scales from dissertation"""
        return {
            "lai_yai": ["A", "C", "D", "E", "G"],
            "lai_noi": ["D", "F", "G", "A", "C"],
            "lai_se": ["E", "G", "A", "B", "D"],
            "lai_sutsanaen": ["C", "D", "E", "G", "A"],
            "thai_pentatonic": ["C", "D", "E", "G", "A"],
        }
    
    def _load_melodic_patterns(self) -> List[List[int]]:
        """Load common Thai melodic patterns from dissertation Figure 3.5"""
        return [
            [1, 2, 3, 5, 2],
            [5, 3, 2, 1, 2, 3],
            [2, 3, 2, 1, 6, 5],
            [5, 6, 1, 2, 1, 6],
        ]
    
    def load_midi(self, filepath: str) -> m21.stream.Score:
        """Load MIDI file using music21"""
        try:
            score = m21.converter.parse(filepath)
            return score
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_melody(self, score: m21.stream.Score) -> List[m21.note.Note]:
        """Extract melody line from score"""
        melody = []
        
        # Find the part with highest notes (usually melody)
        if score.parts:
            melody_part = score.parts[0]  # Assume first part is melody
            for element in melody_part.flatten().notesAndRests:
                if isinstance(element, m21.note.Note):
                    melody.append(element)
        
        return melody
    
    def identify_scale(self, melody: List[m21.note.Note]) -> Optional[str]:
        """Identify Thai scale from melody notes"""
        # Extract unique pitch classes
        pitch_classes = set()
        for note in melody:
            pitch_classes.add(note.pitch.name)
        
        # Compare with known Thai scales
        best_match = None
        best_score = 0
        
        for scale_name, scale_notes in self.thai_scales.items():
            # Calculate overlap
            overlap = len(pitch_classes.intersection(set(scale_notes)))
            score = overlap / len(scale_notes)
            
            if score > best_score:
                best_score = score
                best_match = scale_name
        
        return best_match if best_score > 0.6 else None
    
    def detect_patterns(self, melody: List[m21.note.Note]) -> List[Dict]:
        """Detect Thai melodic patterns in melody"""
        detected_patterns = []
        
        # Convert melody to scale degrees
        scale_degrees = self._to_scale_degrees(melody)
        
        # Search for patterns
        for pattern in self.patterns:
            positions = self._find_pattern_positions(scale_degrees, pattern)
            if positions:
                detected_patterns.append({
                    'pattern': pattern,
                    'positions': positions,
                    'count': len(positions)
                })
        
        return detected_patterns
    
    def _to_scale_degrees(self, melody: List[m21.note.Note]) -> List[int]:
        """Convert melody notes to scale degrees"""
        if not melody:
            return []
        
        # Use first note as tonic
        tonic = melody[0].pitch.pitchClass
        scale_degrees = []
        
        for note in melody:
            degree = (note.pitch.pitchClass - tonic) % 12
            # Map to scale degree (simplified)
            scale_degrees.append(degree + 1)
        
        return scale_degrees
    
    def _find_pattern_positions(self, scale_degrees: List[int], 
                                pattern: List[int]) -> List[int]:
        """Find positions where pattern occurs"""
        positions = []
        pattern_len = len(pattern)
        
        for i in range(len(scale_degrees) - pattern_len + 1):
            if scale_degrees[i:i+pattern_len] == pattern:
                positions.append(i)
        
        return positions
    
    def analyze_rhythm(self, score: m21.stream.Score) -> Dict:
        """Analyze rhythmic structure for nathap cycles"""
        rhythm_info = {
            'time_signature': None,
            'tempo': None,
            'cycle_detected': None,
            'measure_count': 0
        }
        
        if not score.parts:
            return rhythm_info
        
        # Get time signature
        ts = score.parts[0].flatten().getElementsByClass(m21.meter.TimeSignature)
        if ts:
            rhythm_info['time_signature'] = f"{ts[0].numerator}/{ts[0].denominator}"
        
        # Get tempo
        tempos = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
        if tempos:
            rhythm_info['tempo'] = tempos[0].number
        
        # Count measures
        measures = score.parts[0].getElementsByClass(m21.stream.Measure)
        rhythm_info['measure_count'] = len(measures)
        
        # Detect cycle (nathap)
        measure_count = rhythm_info['measure_count']
        if measure_count % 8 == 0:
            rhythm_info['cycle_detected'] = 'propkai_sam_chan'
        elif measure_count % 4 == 0:
            rhythm_info['cycle_detected'] = 'propkai_song_chan'
        elif measure_count % 2 == 0:
            rhythm_info['cycle_detected'] = 'propkai_chan_dio'
        
        return rhythm_info
    
    def extract_features(self, filepath: str) -> Dict:
        """Extract all features from MIDI file"""
        score = self.load_midi(filepath)
        if not score:
            return None
        
        melody = self.extract_melody(score)
        scale = self.identify_scale(melody)
        patterns = self.detect_patterns(melody)
        rhythm = self.analyze_rhythm(score)
        
        return {
            'filename': os.path.basename(filepath),
            'scale': scale,
            'patterns': patterns,
            'rhythm': rhythm,
            'note_count': len(melody),
            'duration': float(score.highestTime)
        }

# ============================================================================
# Part 3: Dataset Preparation
# ============================================================================

class ThaiMusicDatasetBuilder:
    """Build dataset for AI training from dissertation MIDI files"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.processor = ThaiMusicMIDIProcessor()
        self.dataset = []
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create dataset directory structure"""
        dirs = [
            'raw_data/midi_transcriptions/central',
            'raw_data/midi_transcriptions/isan',
            'raw_data/midi_transcriptions/northern',
            'raw_data/midi_transcriptions/southern',
            'processed_data/symbolic',
            'processed_data/features',
            'processed_data/annotations',
            'training_data/train',
            'training_data/validation',
            'training_data/test',
            'metadata'
        ]
        
        for dir_path in dirs:
            full_path = os.path.join(self.output_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
    
    def process_midi_directory(self, input_dir: str, region: ThaiRegion):
        """Process all MIDI files in directory"""
        for filename in os.listdir(input_dir):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                filepath = os.path.join(input_dir, filename)
                features = self.processor.extract_features(filepath)
                
                if features:
                    features['region'] = region.value
                    self.dataset.append(features)
                    
                    # Save to appropriate region folder
                    self._save_processed_file(filepath, region)
        
        print(f"Processed {len(self.dataset)} files from {region.value}")
    
    def _save_processed_file(self, filepath: str, region: ThaiRegion):
        """Save processed file to region folder"""
        output_path = os.path.join(
            self.output_dir,
            'raw_data/midi_transcriptions',
            region.value,
            os.path.basename(filepath)
        )
        # Copy file (simplified - in practice use shutil.copy)
        # shutil.copy(filepath, output_path)
        pass
    
    def create_annotations(self, manual_annotations: Dict):
        """Create annotation files with Thai music elements"""
        for file_id, annotations in manual_annotations.items():
            annotation = {
                'file_id': file_id,
                'region': annotations.get('region'),
                'scale_type': annotations.get('scale_type'),
                'scale_notes': annotations.get('scale_notes'),
                'mode': annotations.get('mode'),
                'tempo': annotations.get('tempo'),
                'time_signature': annotations.get('time_signature'),
                'rhythmic_pattern': annotations.get('rhythmic_pattern'),
                'melodic_patterns': annotations.get('melodic_patterns', []),
                'ornamentations': annotations.get('ornamentations', []),
                'instruments': annotations.get('instruments', []),
                'techniques': annotations.get('techniques', {}),
                'cultural_context': annotations.get('cultural_context', {})
            }
            
            # Save annotation
            output_path = os.path.join(
                self.output_dir,
                'processed_data/annotations',
                f'{file_id}.json'
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    def augment_data(self, augmentation_factor: int = 5):
        """Augment dataset with variations"""
        augmented_dataset = []
        
        for data in self.dataset:
            # Original
            augmented_dataset.append(data)
            
            # Create variations
            for i in range(augmentation_factor - 1):
                augmented = self._create_variation(data, i)
                augmented_dataset.append(augmented)
        
        self.dataset = augmented_dataset
        print(f"Augmented dataset to {len(self.dataset)} examples")
    
    def _create_variation(self, data: Dict, variation_id: int) -> Dict:
        """Create a variation of the data"""
        # In practice, this would:
        # 1. Transpose (carefully, respecting Thai scales)
        # 2. Change tempo (80%-120%)
        # 3. Add ornamentations
        # 4. Vary rhythmic density
        
        variation = data.copy()
        variation['variation_id'] = variation_id
        variation['is_augmented'] = True
        
        return variation
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        # Shuffle
        np.random.shuffle(self.dataset)
        
        n = len(self.dataset)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': self.dataset[:train_end],
            'validation': self.dataset[train_end:val_end],
            'test': self.dataset[val_end:]
        }
        
        # Save splits
        for split_name, split_data in splits.items():
            output_path = os.path.join(
                self.output_dir,
                'training_data',
                split_name,
                'dataset.json'
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"Split: Train={len(splits['train'])}, "
              f"Val={len(splits['validation'])}, "
              f"Test={len(splits['test'])}")
        
        return splits
    
    def save_metadata(self):
        """Save dataset metadata"""
        metadata = {
            'total_files': len(self.dataset),
            'regions': self._count_by_region(),
            'scales': self._count_by_scale(),
            'creation_date': '2025-11-27',
            'source': 'Jazz Orchestra Portraits of Thailand (Tanarat Chaichana, 2022)',
            'purpose': 'AI music generation training dataset'
        }
        
        output_path = os.path.join(self.output_dir, 'metadata', 'dataset_info.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _count_by_region(self) -> Dict:
        """Count files by region"""
        counts = {}
        for data in self.dataset:
            region = data.get('region', 'unknown')
            counts[region] = counts.get(region, 0) + 1
        return counts
    
    def _count_by_scale(self) -> Dict:
        """Count files by scale"""
        counts = {}
        for data in self.dataset:
            scale = data.get('scale', 'unknown')
            counts[scale] = counts.get(scale, 0) + 1
        return counts

# ============================================================================
# Part 4: Model Implementation Skeleton
# ============================================================================

class ThaiMusicTransformer:
    """
    Thai Music Transformer Model
    Based on "Music Transformer" (Huang et al., 2018) with custom embeddings
    """
    
    def __init__(self, 
                 vocab_size: int = 128,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_regions: int = 4,
                 num_scales: int = 20):
        """
        Args:
            vocab_size: MIDI note vocabulary (0-127)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_regions: Number of Thai regions (4)
            num_scales: Number of Thai scales
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # This is a skeleton - full implementation would use PyTorch
        print("Initializing Thai Music Transformer...")
        print(f"  - Model dimension: {d_model}")
        print(f"  - Attention heads: {nhead}")
        print(f"  - Layers: {num_layers}")
        print(f"  - Regions: {num_regions}")
        print(f"  - Scales: {num_scales}")
    
    def forward(self, x, region_id, scale_id, pattern_id):
        """
        Forward pass with Thai music context
        
        Args:
            x: Input sequence (batch_size, seq_len)
            region_id: Thai region ID (batch_size,)
            scale_id: Thai scale ID (batch_size,)
            pattern_id: Melodic pattern ID (batch_size,)
        
        Returns:
            output: Generated sequence
        """
        # Skeleton implementation
        # Full implementation would include:
        # 1. Note embeddings
        # 2. Positional embeddings
        # 3. Regional embeddings
        # 4. Scale embeddings
        # 5. Pattern embeddings
        # 6. Transformer encoder
        # 7. Transformer decoder
        # 8. Output projection
        pass
    
    def generate(self, 
                 prompt: List[int],
                 region: str,
                 scale: str,
                 max_length: int = 256,
                 temperature: float = 1.0):
        """
        Generate Thai music sequence
        
        Args:
            prompt: Initial notes
            region: Thai region ('central', 'isan', etc.)
            scale: Thai scale name
            max_length: Maximum sequence length
            temperature: Sampling temperature
        
        Returns:
            generated_sequence: List of MIDI notes
        """
        print(f"Generating {region} style music in {scale} scale...")
        # Skeleton - full implementation would do actual generation
        return []

# ============================================================================
# Part 5: Usage Examples
# ============================================================================

def example_1_process_single_midi():
    """Example: Process a single MIDI file"""
    print("\n=== Example 1: Process Single MIDI File ===\n")
    
    processor = ThaiMusicMIDIProcessor()
    
    # Process "Lai Ka Ten Kon" from dissertation Appendix C
    features = processor.extract_features("lai_ka_ten_kon.mid")
    
    if features:
        print("Extracted Features:")
        print(json.dumps(features, indent=2))

def example_2_build_dataset():
    """Example: Build complete dataset"""
    print("\n=== Example 2: Build Dataset ===\n")
    
    builder = ThaiMusicDatasetBuilder(output_dir="./Thai_Music_AI_Dataset")
    
    # Process MIDI files by region
    builder.process_midi_directory("./appendix_c/isan/", ThaiRegion.ISAN)
    builder.process_midi_directory("./appendix_c/central/", ThaiRegion.CENTRAL)
    builder.process_midi_directory("./appendix_c/northern/", ThaiRegion.NORTHERN)
    builder.process_midi_directory("./appendix_c/southern/", ThaiRegion.SOUTHERN)
    
    # Add manual annotations (from dissertation analysis)
    manual_annotations = {
        'lai_ka_ten_kon': {
            'region': 'isan',
            'scale_type': 'lai_thang_san',
            'scale_notes': ['C', 'D', 'E', 'G', 'A'],
            'mode': 'major_pentatonic',
            'tempo': 120,
            'time_signature': '4/4',
            'rhythmic_pattern': 'chan_dio',
            'instruments': ['phin', 'khaen'],
            'techniques': {
                'phin': ['E_A_E_tuning', 'drone_pattern'],
                'khaen': ['pentatonic_fingering']
            }
        }
    }
    
    builder.create_annotations(manual_annotations)
    
    # Augment data
    builder.augment_data(augmentation_factor=5)
    
    # Split dataset
    builder.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Save metadata
    builder.save_metadata()
    
    print("\nDataset build complete!")

def example_3_train_model():
    """Example: Initialize and train model (skeleton)"""
    print("\n=== Example 3: Model Training (Skeleton) ===\n")
    
    # Initialize model
    model = ThaiMusicTransformer(
        vocab_size=128,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    
    # In practice, would load dataset and train
    print("\nTraining would proceed with:")
    print("1. Load dataset from ./Thai_Music_AI_Dataset/training_data/")
    print("2. Create data loaders")
    print("3. Define loss function (CrossEntropyLoss)")
    print("4. Define optimizer (AdamW)")
    print("5. Training loop with validation")
    print("6. Save checkpoints")

def example_4_generate_music():
    """Example: Generate Thai music (skeleton)"""
    print("\n=== Example 4: Music Generation (Skeleton) ===\n")
    
    model = ThaiMusicTransformer(vocab_size=128)
    
    # Generate Isan-style music
    prompt = [60, 62, 64, 67, 69]  # C-D-E-G-A (pentatonic)
    generated = model.generate(
        prompt=prompt,
        region='isan',
        scale='lai_sutsanaen',
        max_length=256,
        temperature=1.0
    )
    
    print("Generated sequence:", generated)
    print("\n(Full implementation would convert to MIDI and audio)")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Thai Music AI Dataset Creation - Code Examples")
    print("Based on: Jazz Orchestra Portraits of Thailand (Chaichana, 2022)")
    print("=" * 70)
    
    # Run examples
    example_1_process_single_midi()
    example_2_build_dataset()
    example_3_train_model()
    example_4_generate_music()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("For full implementation, refer to the documentation.")
    print("=" * 70)
