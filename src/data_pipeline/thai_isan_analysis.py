"""
Thai Isan Music Analysis Module

This module focuses on accurately capturing every musical note in Thai Isan music,
with special attention to the 7-tone scale system and traditional Phin lute patterns.
"""
import librosa
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from collections import defaultdict
import matplotlib.pyplot as plt
from ..data_pipeline.feature_extraction import extract_note_events, quantize_to_thai_scale
from ..utils.constants import THAI_7_TONE_RATIOS, CQT_PARAMS, AUDIO_PARAMS


def analyze_thai_scale_adherence(audio_path, sr=CQT_PARAMS['sr']):
    """
    Analyze how well the audio adheres to the Thai 7-tone scale system.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Sample rate
    
    Returns:
        dict: Analysis results including scale adherence metrics
    """
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Extract pitch contours
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, 
        fmin=65,  # C2, lowest note on Phin
        fmax=1000,  # Approximate upper limit for Phin fundamental frequencies
        hop_length=AUDIO_PARAMS['hop_length']
    )
    
    # Filter out unvoiced frames
    valid_f0 = f0[voiced_flag]
    
    if len(valid_f0) == 0:
        return {
            'scale_adherence': 0.0,
            'detected_frequencies': [],
            'thai_scale_matches': 0,
            'total_notes': 0
        }
    
    # Quantize frequencies to Thai scale
    quantized_frequencies = []
    scale_matches = 0
    
    for freq in valid_f0:
        if not np.isnan(freq):
            quantized_freq = quantize_to_thai_scale(freq)
            quantized_frequencies.append(quantized_freq)
            
            # Check if it's close to a Thai scale frequency
            if abs(freq - quantized_freq) / freq < 0.05:  # 5% tolerance
                scale_matches += 1
    
    adherence = scale_matches / len(valid_f0) if len(valid_f0) > 0 else 0
    
    return {
        'scale_adherence': adherence,
        'detected_frequencies': valid_f0,
        'quantized_frequencies': quantized_frequencies,
        'thai_scale_matches': scale_matches,
        'total_notes': len(valid_f0),
        'frequency_distribution': np.histogram(valid_f0, bins=50)[0] if len(valid_f0) > 0 else []
    }


def extract_phin_playing_techniques(audio_path, sr=CQT_PARAMS['sr']):
    """
    Extract features related to Phin lute playing techniques.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary of extracted playing technique features
    """
    y, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Compute STFT for detailed analysis
    D = librosa.stft(y, n_fft=AUDIO_PARAMS['n_fft'], hop_length=AUDIO_PARAMS['hop_length'])
    magnitude, phase = librosa.magphase(D)
    
    # Extract spectral features
    spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=AUDIO_PARAMS['hop_length'])[0]
    
    # Detect onsets (strumming/picking patterns)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=AUDIO_PARAMS['hop_length'])
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=AUDIO_PARAMS['hop_length'])
    
    # Detect pitches
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, 
        fmin=65,  # C2
        fmax=1000,  # C6
        hop_length=AUDIO_PARAMS['hop_length']
    )
    
    # Analyze vibrato (common in Phin playing)
    vibrato_features = detect_vibrato(f0, voiced_flag, sr)
    
    return {
        'spectral_centroids': spectral_centroids,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'zero_crossing_rate': zero_crossing_rate,
        'onset_times': onset_times,
        'f0': f0,
        'voiced_flag': voiced_flag,
        'vibrato_features': vibrato_features,
        'onset_count': len(onset_times)
    }


def detect_vibrato(f0, voiced_flag, sr, hop_length=AUDIO_PARAMS['hop_length']):
    """
    Detect vibrato characteristics in the pitch contour.
    
    Args:
        f0 (np.ndarray): Fundamental frequency contour
        voiced_flag (np.ndarray): Boolean array indicating voiced frames
        sr (int): Sample rate
        hop_length (int): Hop length used in analysis
    
    Returns:
        dict: Vibrato detection results
    """
    if len(f0) == 0:
        return {
            'vibrato_present': False,
            'vibrato_rate': 0.0,
            'vibrato_extent': 0.0
        }
    
    # Only consider voiced frames
    voiced_f0 = f0[voiced_flag]
    
    if len(voiced_f0) < 10:  # Need enough points for analysis
        return {
            'vibrato_present': False,
            'vibrato_rate': 0.0,
            'vibrato_extent': 0.0
        }
    
    # Remove DC component and detrend
    detrended_f0 = signal.detrend(voiced_f0)
    
    # Compute FFT to find vibrato frequency
    fft_result = np.fft.fft(detrended_f0)
    freqs = np.fft.fftfreq(len(detrended_f0), d=hop_length/sr)
    
    # Look for energy in typical vibrato range (4-8 Hz for Phin)
    vibrato_range = (4, 8)
    vibrato_mask = (freqs >= vibrato_range[0]) & (freqs <= vibrato_range[1])
    
    if np.any(vibrato_mask):
        vibrato_energy = np.abs(fft_result[vibrato_mask])
        max_idx = np.argmax(vibrato_energy)
        vibrato_rate = freqs[vibrato_mask][max_idx]
        
        # Calculate vibrato extent (deviation from mean)
        vibrato_extent = np.std(detrended_f0) * 100  # In cents
        
        return {
            'vibrato_present': True,
            'vibrato_rate': vibrato_rate,
            'vibrato_extent': vibrato_extent
        }
    else:
        return {
            'vibrato_present': False,
            'vibrato_rate': 0.0,
            'vibrato_extent': 0.0
        }


def accurate_note_detection(audio_path, sr=CQT_PARAMS['sr']):
    """
    Perform highly accurate note detection for Thai Isan music.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Sample rate
    
    Returns:
        list: List of detected notes with precise timing and pitch
    """
    y, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Apply bandpass filter to focus on Phin frequencies
    nyq = 0.5 * sr
    low = 65 / nyq  # C2
    high = 1000 / nyq  # C6
    b, a = signal.butter(4, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)
    
    # Compute Constant-Q Transform for better frequency resolution
    cqt = librosa.cqt(
        y_filtered,
        sr=sr,
        fmin=CQT_PARAMS['fmin'],
        n_bins=CQT_PARAMS['n_bins'],
        bins_per_octave=CQT_PARAMS['bins_per_octave']
    )
    
    # Convert to magnitude
    cqt_mag = np.abs(cqt)
    
    # Apply non-negative matrix factorization to separate notes
    from sklearn.decomposition import NMF
    
    # Transpose to (time, freq) format
    cqt_mag_t = cqt_mag.T
    
    # Use NMF to decompose spectrogram into note-like components
    n_components = min(20, cqt_mag_t.shape[1] // 10)  # Reasonable number of components
    if n_components < 1:
        n_components = 1
    
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
    W = model.fit_transform(cqt_mag_t)
    H = model.components_
    
    # Find peaks in the activation matrix W
    note_events = []
    time_resolution = AUDIO_PARAMS['hop_length'] / sr  # Time per frame
    
    for i in range(W.shape[1]):  # For each component
        activations = W[:, i]
        # Find peaks in activation
        peaks, properties = find_peaks(
            activations, 
            height=np.max(activations) * 0.3,  # At least 30% of max
            distance=int(0.1 / time_resolution)  # At least 100ms apart
        )
        
        for peak_idx in peaks:
            start_time = peak_idx * time_resolution
            # Find the dominant frequency bin for this component
            freq_bin = np.argmax(H[i, :])
            # Convert to frequency
            freq = librosa.cqt_frequencies(
                n_bins=CQT_PARAMS['n_bins'],
                fmin=CQT_PARAMS['fmin'],
                bins_per_octave=CQT_PARAMS['bins_per_octave']
            )[freq_bin]
            
            # Quantize to Thai scale
            quantized_freq = quantize_to_thai_scale(freq)
            
            note_events.append({
                'start_time': start_time,
                'frequency': quantized_freq,
                'midi_note': librosa.hz_to_midi(quantized_freq),
                'activation': activations[peak_idx],
                'component': i
            })
    
    # Sort by time
    note_events.sort(key=lambda x: x['start_time'])
    
    # Group nearby events that likely represent the same note
    grouped_notes = group_similar_notes(note_events, time_tolerance=0.1, freq_tolerance=0.5)
    
    return grouped_notes


def group_similar_notes(note_events, time_tolerance=0.1, freq_tolerance=0.5):
    """
    Group similar note events that likely represent the same musical note.
    
    Args:
        note_events (list): List of note events from accurate_note_detection
        time_tolerance (float): Time tolerance for grouping (in seconds)
        freq_tolerance (float): Frequency tolerance for grouping (in Hz)
    
    Returns:
        list: Grouped note events
    """
    if not note_events:
        return []
    
    grouped = []
    current_group = [note_events[0]]
    
    for event in note_events[1:]:
        last_event = current_group[-1]
        
        # Check if this event is close enough to the last in the group
        time_diff = abs(event['start_time'] - last_event['start_time'])
        freq_diff = abs(event['frequency'] - last_event['frequency'])
        
        if time_diff <= time_tolerance and freq_diff <= freq_tolerance:
            # Add to current group
            current_group.append(event)
        else:
            # Finalize current group and start new one
            if current_group:
                grouped.append(merge_note_group(current_group))
            current_group = [event]
    
    # Add the last group
    if current_group:
        grouped.append(merge_note_group(current_group))
    
    return grouped


def merge_note_group(note_group):
    """
    Merge a group of similar note events into a single note.
    
    Args:
        note_group (list): List of similar note events
    
    Returns:
        dict: Merged note event
    """
    if len(note_group) == 1:
        return note_group[0]
    
    # Calculate average properties
    start_times = [n['start_time'] for n in note_group]
    frequencies = [n['frequency'] for n in note_group]
    activations = [n['activation'] for n in note_group]
    
    avg_start_time = np.mean(start_times)
    avg_frequency = np.mean(frequencies)
    avg_activation = np.mean(activations)
    
    return {
        'start_time': avg_start_time,
        'frequency': avg_frequency,
        'midi_note': librosa.hz_to_midi(avg_frequency),
        'activation': avg_activation,
        'duration': max(start_times) - min(start_times)
    }


def analyze_phin_patterns(audio_path, sr=CQT_PARAMS['sr']):
    """
    Analyze specific Phin lute playing patterns in the audio.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Sample rate
    
    Returns:
        dict: Analysis of Phin-specific patterns
    """
    # Extract note events
    note_events = extract_note_events(audio_path, sr)
    
    # Analyze melodic patterns
    melodic_patterns = detect_melodic_patterns(note_events)
    
    # Analyze rhythmic patterns
    rhythmic_features = extract_rhythmic_features(note_events)
    
    # Analyze pitch relationships
    pitch_relationships = analyze_pitch_relationships(note_events)
    
    return {
        'note_events': note_events,
        'melodic_patterns': melodic_patterns,
        'rhythmic_features': rhythmic_features,
        'pitch_relationships': pitch_relationships,
        'note_count': len(note_events),
        'note_density': len(note_events) / get_audio_duration(audio_path)
    }


def detect_melodic_patterns(note_events):
    """
    Detect common melodic patterns in Thai Isan music.
    
    Args:
        note_events (list): List of note events
    
    Returns:
        dict: Detected melodic patterns
    """
    if len(note_events) < 2:
        return {'patterns': [], 'common_intervals': []}
    
    # Calculate intervals between consecutive notes
    intervals = []
    for i in range(1, len(note_events)):
        prev_note = note_events[i-1]
        curr_note = note_events[i]
        interval = curr_note['pitch'] - prev_note['pitch']
        intervals.append(interval)
    
    # Find common intervals (characteristic of Thai music)
    unique_intervals, counts = np.unique(intervals, return_counts=True)
    common_intervals = list(zip(unique_intervals, counts))
    common_intervals.sort(key=lambda x: x[1], reverse=True)  # Sort by frequency
    
    # Detect common melodic patterns (sequences of intervals)
    patterns = []
    for length in range(2, min(6, len(intervals))):  # Look for patterns of length 2-5
        for i in range(len(intervals) - length + 1):
            pattern = tuple(intervals[i:i+length])
            patterns.append(pattern)
    
    # Count pattern occurrences
    pattern_counts = defaultdict(int)
    for pattern in patterns:
        pattern_counts[pattern] += 1
    
    # Get most common patterns
    common_patterns = [(pattern, count) for pattern, count in pattern_counts.items()]
    common_patterns.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'common_intervals': common_intervals[:10],  # Top 10 intervals
        'common_patterns': common_patterns[:10],    # Top 10 patterns
        'total_patterns': len(set(patterns)),
        'interval_histogram': np.histogram(intervals, bins=20)[0] if intervals else []
    }


def extract_rhythmic_features(note_events):
    """
    Extract rhythmic features specific to Thai Isan music.
    
    Args:
        note_events (list): List of note events
    
    Returns:
        dict: Rhythmic features
    """
    if len(note_events) < 2:
        return {'tempo': 0, 'rhythmic_variability': 0, 'note_durations': []}
    
    # Calculate inter-onset intervals
    ioi = []
    for i in range(1, len(note_events)):
        ioi.append(note_events[i]['start_time'] - note_events[i-1]['start_time'])
    
    # Calculate note durations
    note_durations = [event['end_time'] - event['start_time'] for event in note_events]
    
    # Estimate tempo (notes per minute)
    if ioi:
        mean_ioi = np.mean(ioi)
        tempo = 60.0 / mean_ioi if mean_ioi > 0 else 0
    else:
        tempo = 0
    
    # Calculate rhythmic variability
    rhythmic_variability = np.std(ioi) / np.mean(ioi) if ioi and np.mean(ioi) > 0 else 0
    
    return {
        'tempo': tempo,
        'inter_onset_intervals': ioi,
        'note_durations': note_durations,
        'rhythmic_variability': rhythmic_variability,
        'mean_ioi': np.mean(ioi) if ioi else 0,
        'std_ioi': np.std(ioi) if ioi else 0
    }


def analyze_pitch_relationships(note_events):
    """
    Analyze pitch relationships in the note sequence.
    
    Args:
        note_events (list): List of note events
    
    Returns:
        dict: Pitch relationship analysis
    """
    if not note_events:
        return {'pitch_range': 0, 'pitch_histogram': [], 'common_pitches': []}
    
    pitches = [event['pitch'] for event in note_events]
    
    # Calculate pitch range
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    
    # Create pitch histogram
    unique_pitches, counts = np.unique(pitches, return_counts=True)
    pitch_histogram = list(zip(unique_pitches, counts))
    
    # Find most common pitches
    pitch_histogram.sort(key=lambda x: x[1], reverse=True)
    common_pitches = pitch_histogram[:10]  # Top 10 most common pitches
    
    return {
        'pitch_range': pitch_range,
        'pitch_histogram': pitch_histogram,
        'common_pitches': common_pitches,
        'total_unique_pitches': len(unique_pitches),
        'pitch_entropy': calculate_entropy(counts)
    }


def calculate_entropy(counts):
    """
    Calculate entropy of a distribution.
    
    Args:
        counts (np.ndarray): Counts of occurrences
    
    Returns:
        float: Entropy value
    """
    if len(counts) == 0:
        return 0
    
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  # Remove zero probabilities
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def get_audio_duration(audio_path):
    """
    Get the duration of an audio file.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        float: Duration in seconds
    """
    y, sr = librosa.load(audio_path, sr=None)
    return len(y) / sr


def create_detailed_transcription_report(audio_path):
    """
    Create a comprehensive report of the transcription analysis.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Comprehensive analysis report
    """
    print(f"Creating detailed transcription report for: {audio_path}")
    
    # Analyze Thai scale adherence
    scale_analysis = analyze_thai_scale_adherence(audio_path)
    
    # Extract Phin playing techniques
    technique_features = extract_phin_playing_techniques(audio_path)
    
    # Perform accurate note detection
    accurate_notes = accurate_note_detection(audio_path)
    
    # Analyze Phin patterns
    pattern_analysis = analyze_phin_patterns(audio_path)
    
    # Compile report
    report = {
        'audio_path': audio_path,
        'scale_analysis': scale_analysis,
        'technique_features': technique_features,
        'accurate_notes': accurate_notes,
        'pattern_analysis': pattern_analysis,
        'note_count': len(accurate_notes),
        'thai_scale_adherence': scale_analysis['scale_adherence']
    }
    
    print(f"Analysis complete. Found {len(accurate_notes)} notes with {scale_analysis['scale_adherence']:.2%} Thai scale adherence.")
    
    return report


# Example usage function
def analyze_sample_audio(audio_path):
    """
    Example function to demonstrate the analysis capabilities.
    
    Args:
        audio_path (str): Path to the audio file to analyze
    """
    print("Thai Isan Music Analysis")
    print("=" * 30)
    
    # Create detailed report
    report = create_detailed_transcription_report(audio_path)
    
    # Print summary
    print(f"\nAnalysis Summary for: {audio_path}")
    print(f"Thai Scale Adherence: {report['thai_scale_adherence']:.2%}")
    print(f"Number of Detected Notes: {report['note_count']}")
    print(f"Estimated Tempo: {report['pattern_analysis']['rhythmic_features']['tempo']:.2f} BPM")
    print(f"Pitch Range: {report['pattern_analysis']['pitch_relationships']['pitch_range']} semitones")
    
    return report


if __name__ == "__main__":
    # Example usage (would need a real audio file)
    print("Thai Isan Music Analysis Module")
    print("This module provides tools for accurate note detection in Thai Isan music")
    print("with special focus on the 7-tone scale system and Phin lute patterns.")