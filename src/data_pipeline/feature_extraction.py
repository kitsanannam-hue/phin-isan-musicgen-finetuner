"""
Feature extraction module for Thai Isan Lute (Phin) Music Transcription

Contains functions for extracting audio features optimized for Thai 7-tone scale system
and Phin lute characteristics.
"""
import librosa
import numpy as np
from scipy import signal
from ..utils.constants import CQT_PARAMS, AUDIO_PARAMS, THAI_7_TONE_RATIOS


def extract_phin_features(audio_path, sr=CQT_PARAMS['sr']):
    """
    Extract Constant-Q Transform features optimized for Thai Isan music.
    This function is specifically designed to handle the 7-tone scale system
    and unique characteristics of the Phin lute.

    Args:
        audio_path (str): Path to the audio file
        sr (int): Target sample rate

    Returns:
        np.ndarray: Normalized CQT spectrogram (frequency bins, time)
    """
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Normalize audio to optimal levels for Thai Isan music characteristics
    y = normalize_audio(y, sr)
    
    # Apply bandpass filter to focus on Phin lute frequencies
    y = apply_bandpass_filter(y, sr)
    
    # Compute Constant-Q Transform with parameters optimized for Thai 7-tone system
    # The high resolution (24 bins per octave) is crucial for capturing the microtonal
    # characteristics of Thai traditional music
    cqt = librosa.cqt(
        y,
        sr=sr,
        fmin=CQT_PARAMS['fmin'],
        n_bins=CQT_PARAMS['n_bins'],
        bins_per_octave=CQT_PARAMS['bins_per_octave'],
        filter_scale=CQT_PARAMS['filter_scale']
    )
    
    # Convert to magnitude and apply log scaling
    cqt_mag = np.abs(cqt)
    cqt_log = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    
    # Normalize to 0-1 range
    cqt_normalized = (cqt_log - np.min(cqt_log)) / (np.max(cqt_log) - np.min(cqt_log))
    
    return cqt_normalized


def extract_harmonic_features(y, sr):
    """
    Extract harmonic features specific to Phin lute timbre.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary of harmonic features
    """
    # Compute harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Extract spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_harmonic, sr=sr)[0]
    
    # Compute zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y_harmonic)[0]
    
    # Compute MFCCs for timbral characteristics
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mels=13)
    
    return {
        'spectral_centroids': spectral_centroids,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'zcr': zcr,
        'mfccs': mfccs
    }


def normalize_audio(y, sr):
    """
    Normalize audio to optimal levels for Thai Isan music characteristics.
    This normalization preserves the dynamic range important for traditional music.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
    
    Returns:
        np.ndarray: Normalized audio
    """
    # Apply a gentle normalization that preserves dynamic range
    y_max = np.max(np.abs(y))
    if y_max > 0:
        y = y / y_max * 0.8  # Normalize to 80% of full scale
    
    return y


def apply_bandpass_filter(y, sr, lowcut=60.0, highcut=8000.0, order=5):
    """
    Apply a bandpass filter to focus on Phin lute frequencies.
    The Phin lute typically produces fundamental frequencies in the range of 
    approximately 65Hz (C2) to 1047Hz (C6), with harmonics extending higher.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        order (int): Filter order
    
    Returns:
        np.ndarray: Filtered audio
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y_filtered = signal.lfilter(b, a, y)
    return y_filtered


def detect_pitch_contours(y, sr, hop_length=AUDIO_PARAMS['hop_length']):
    """
    Detect pitch contours in the audio, optimized for Thai Isan music.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        hop_length (int): Hop length for analysis
    
    Returns:
        np.ndarray: Array of fundamental frequencies over time
    """
    # Use librosa's pyin for pitch tracking (more accurate than simple yin)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, 
        fmin=65,  # C2, lowest note on Phin
        fmax=1000,  # Approximate upper limit for Phin fundamental frequencies
        hop_length=hop_length
    )
    
    return f0, voiced_flag, voiced_prob


def extract_note_events(audio_path, sr=CQT_PARAMS['sr']):
    """
    Extract note-level events from audio, optimized for Thai 7-tone scale.
    
    Args:
        audio_path (str): Path to audio file
        sr (int): Sample rate
    
    Returns:
        list: List of note events (start_time, end_time, pitch, amplitude)
    """
    # Load and preprocess audio
    y, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    y = normalize_audio(y, sr)
    y = apply_bandpass_filter(y, sr)
    
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=AUDIO_PARAMS['hop_length'])
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=AUDIO_PARAMS['hop_length'])
    
    # Extract pitch contours
    f0, voiced_flag, voiced_prob = detect_pitch_contours(y, sr)
    time_axis = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    
    # Group consecutive frames with similar pitch to form note events
    note_events = []
    current_note = None
    
    for i, (time, freq) in enumerate(zip(time_axis, f0)):
        if freq is not None and not np.isnan(freq):  # Valid pitch detected
            # Quantize to Thai 7-tone scale
            quantized_pitch = quantize_to_thai_scale(freq)
            
            if current_note is None:
                # Start new note
                current_note = {
                    'start_time': time,
                    'pitch': quantized_pitch,
                    'amplitude': np.abs(y[int(time * sr)])  # Approximate amplitude
                }
            else:
                # Check if pitch changed significantly
                if abs(current_note['pitch'] - quantized_pitch) > 0.5:  # Significant change
                    # End current note and start new one
                    current_note['end_time'] = time
                    note_events.append(current_note)
                    
                    current_note = {
                        'start_time': time,
                        'pitch': quantized_pitch,
                        'amplitude': np.abs(y[int(time * sr)])
                    }
        else:
            # No valid pitch, end current note if exists
            if current_note is not None:
                current_note['end_time'] = time
                note_events.append(current_note)
                current_note = None
    
    # Close the last note if still active
    if current_note is not None:
        current_note['end_time'] = time_axis[-1]  # End at the end of the audio
        note_events.append(current_note)
    
    return note_events


def quantize_to_thai_scale(frequency, reference_freq=440.0):
    """
    Quantize a frequency to the nearest note in the Thai 7-tone scale.
    
    Args:
        frequency (float): Input frequency to quantize
        reference_freq (float): Reference frequency (A4)
    
    Returns:
        float: Quantized frequency according to Thai 7-tone scale
    """
    # Calculate the interval relative to the reference frequency
    interval_ratio = frequency / reference_freq
    
    # Find the closest Thai scale degree
    closest_degree = 0
    min_difference = float('inf')
    
    for degree, ratio in THAI_7_TONE_RATIOS.items():
        # Adjust ratio relative to A4
        expected_freq = reference_freq * ratio / THAI_7_TONE_RATIOS[4]  # Scale relative to fifth (3/2 ratio)
        difference = abs(expected_freq - frequency)
        
        if difference < min_difference:
            min_difference = difference
            closest_degree = degree
    
    # Return the quantized frequency
    return reference_freq * THAI_7_TONE_RATIOS[closest_degree] / THAI_7_TONE_RATIOS[4]


def extract_rhythm_features(y, sr):
    """
    Extract rhythm features specific to Thai Isan music patterns.
    
    Args:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary of rhythm features
    """
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Detect tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    
    # Compute beat times
    beats = librosa.frames_to_time(
        librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)[1],
        sr=sr
    )
    
    # Extract rhythmic patterns
    # Calculate inter-beat intervals
    if len(beats) > 1:
        ibi = np.diff(beats)  # Inter-beat intervals
        mean_ibi = np.mean(ibi)
        std_ibi = np.std(ibi)
    else:
        mean_ibi = 0
        std_ibi = 0
    
    return {
        'tempo': tempo,
        'beats': beats,
        'mean_ibi': mean_ibi,
        'std_ibi': std_ibi,
        'onset_env': onset_env
    }