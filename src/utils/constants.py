"""
Constants for Thai Isan Lute (Phin) Music Transcription

This module contains constants related to the Thai 7-tone musical scale system
and other parameters specific to Thai Isan music transcription.
"""

# Thai 7-tone scale frequencies (relative to a reference pitch)
# Thai traditional music uses a heptatonic scale system that differs from Western 12-TET
# These ratios are based on traditional Thai music theory
THAI_7_TONE_RATIOS = {
    # Ratios relative to the fundamental (tonic)
    0: 1.0,          # Tonic (Do)
    1: 1.125,        # Second (Re) - 9/8 ratio
    2: 1.25,         # Third (Mi) - 5/4 ratio
    3: 1.333,        # Fourth (Fa) - 4/3 ratio (approximately)
    4: 1.5,          # Fifth (So) - 3/2 ratio
    5: 1.667,        # Sixth (La) - 5/3 ratio (approximately)
    6: 1.789,        # Seventh (Ti) - Augmented sixth (approximately)
}

# Common fundamental frequencies for Thai Isan music (in Hz)
# These are typical tuning frequencies for the Phin lute
COMMON_FUNDAMENTALS = {
    'C': 261.63,  # Standard C
    'D': 293.66,  # Standard D
    'Eb': 311.13, # Standard Eb
    'F': 349.23,  # Standard F
    'G': 392.00,  # Standard G
    'A': 440.00,  # Standard A (often used as reference)
    'Bb': 466.16, # Standard Bb
}

# Default parameters for Constant-Q Transform optimized for Thai music
CQT_PARAMS = {
    'sr': 22050,        # Sample rate
    'fmin': 65.41,      # Frequency of C2 (lowest note on Phin)
    'n_bins': 120,      # Number of bins covering 5 octaves
    'bins_per_octave': 24,  # Higher resolution for non-12-TET systems
    'filter_scale': 2,  # Filter scale for better frequency resolution
}

# MIDI note mapping for Thai 7-tone scale
# Maps Thai scale degrees to MIDI note numbers
THAI_SCALE_TO_MIDI = {
    # This is a simplified mapping - in practice, microtonal adjustments may be needed
    'tonic': 60,      # C4
    'second': 62,     # D4 (may be slightly flat compared to Thai tuning)
    'third': 64,      # E4 (may be slightly flat compared to Thai tuning)
    'fourth': 65,     # F4
    'fifth': 67,      # G4
    'sixth': 69,      # A4 (may be slightly flat compared to Thai tuning)
    'seventh': 71,    # B4
}

# Phin lute specific parameters
PHIN_PARAMS = {
    'string_count': 3,  # Traditional Phin has 3 strings
    'open_strings': ['D3', 'A3', 'D4'],  # Common tuning for Thai Phin
    'max_frets': 15,    # Typical number of frets on a Phin
    'vibrato_range': 0.5,  # Typical vibrato range in semitones
}

# Audio processing parameters optimized for Phin sound characteristics
AUDIO_PARAMS = {
    'sample_rate': 22050,
    'hop_length': 512,      # Hop length for feature extraction
    'n_fft': 2048,          # FFT window size
    'duration_threshold': 0.05,  # Minimum note duration (50ms)
    'amplitude_threshold': 0.01, # Minimum amplitude threshold
    'harmonic_threshold': 0.3,   # Threshold for harmonic detection
}

# Transcription parameters
TRANSCRIPTION_PARAMS = {
    'onset_threshold': 0.3,    # Onset detection threshold
    'pitch_tolerance': 0.2,    # Pitch tolerance for note matching (in semitones)
    'tempo_range': (40, 200),  # Valid tempo range (BPM)
    'max_polyphony': 3,        # Maximum number of simultaneous notes (Phin is typically monophonic with occasional harmony)
}