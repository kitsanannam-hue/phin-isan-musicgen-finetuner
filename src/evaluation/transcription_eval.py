"""
Transcription evaluation module for Thai Isan Lute (Phin) Music Transcription

Implements evaluation metrics for measuring transcription accuracy,
specifically Onset F1 and Pitch F1 scores.
"""
import mir_eval
import pretty_midi
import numpy as np
from collections import defaultdict
from ..utils.constants import TRANSCRIPTION_PARAMS


def evaluate_transcription(reference_midi_path, predicted_midi_path):
    """
    Evaluate the quality of a transcription by comparing it to a reference.
    
    Args:
        reference_midi_path (str): Path to the reference MIDI file
        predicted_midi_path (str): Path to the predicted MIDI file
    
    Returns:
        tuple: (onset_f1, pitch_f1) scores
    """
    # Load reference and predicted MIDI files
    ref_midi = pretty_midi.PrettyMIDI(reference_midi_path)
    pred_midi = pretty_midi.PrettyMIDI(predicted_midi_path)
    
    # Extract note events from both MIDI files
    ref_intervals, ref_pitches = get_notes_from_midi(ref_midi)
    pred_intervals, pred_pitches = get_notes_from_midi(pred_midi)
    
    # Validate inputs
    if len(ref_intervals) == 0 or len(pred_intervals) == 0:
        return 0.0, 0.0  # Return 0 if no notes detected
    
    # Compute onset-based F1 score
    onset_precision, onset_recall, onset_f1 = mir_eval.onset.f_measure(
        ref_intervals[:, 0],  # Reference onset times
        pred_intervals[:, 0],  # Predicted onset times
        window=TRANSCRIPTION_PARAMS['onset_threshold']  # Tolerance window
    )
    
    # Compute pitch-based F1 score
    pitch_precision, pitch_recall, pitch_f1 = mir_eval.transcription.precision_recall_f1(
        ref_intervals, ref_pitches,
        pred_intervals, pred_pitches,
        offset_ratio=None  # Don't consider note offsets for basic pitch evaluation
    )
    
    return onset_f1, pitch_f1


def get_notes_from_midi(midi_data):
    """
    Extract note intervals and pitches from a PrettyMIDI object.
    
    Args:
        midi_data (pretty_midi.PrettyMIDI): MIDI data object
    
    Returns:
        tuple: (intervals, pitches) where intervals is (N, 2) array of [start, end] times
               and pitches is (N,) array of MIDI note numbers
    """
    intervals = []
    pitches = []
    
    # Process all instruments in the MIDI file
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Only consider non-drum instruments
            for note in instrument.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)
    
    if intervals:
        intervals = np.array(intervals)
        pitches = np.array(pitches)
    else:
        intervals = np.empty((0, 2))
        pitches = np.empty(0)
    
    return intervals, pitches


def evaluate_midi_accuracy(ref_midi, pred_midi):
    """
    Comprehensive evaluation of MIDI transcription accuracy.
    
    Args:
        ref_midi (pretty_midi.PrettyMIDI): Reference MIDI object
        pred_midi (pretty_midi.PrettyMIDI): Predicted MIDI object
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Extract note events
    ref_intervals, ref_pitches = get_notes_from_midi(ref_midi)
    pred_intervals, pred_pitches = get_notes_from_midi(pred_midi)
    
    metrics = {}
    
    # Onset detection evaluation
    if len(ref_intervals) > 0 and len(pred_intervals) > 0:
        onset_precision, onset_recall, onset_f1 = mir_eval.onset.f_measure(
            ref_intervals[:, 0],  # Reference onsets
            pred_intervals[:, 0],  # Predicted onsets
            window=TRANSCRIPTION_PARAMS['onset_threshold']
        )
        metrics['onset_precision'] = onset_precision
        metrics['onset_recall'] = onset_recall
        metrics['onset_f1'] = onset_f1
    else:
        metrics['onset_precision'] = 0.0
        metrics['onset_recall'] = 0.0
        metrics['onset_f1'] = 0.0
    
    # Pitch accuracy evaluation
    if len(ref_pitches) > 0 and len(pred_pitches) > 0:
        pitch_precision, pitch_recall, pitch_f1 = mir_eval.transcription.precision_recall_f1(
            ref_intervals, ref_pitches,
            pred_intervals, pred_pitches,
            offset_ratio=None
        )
        metrics['pitch_precision'] = pitch_precision
        metrics['pitch_recall'] = pitch_recall
        metrics['pitch_f1'] = pitch_f1
    else:
        metrics['pitch_precision'] = 0.0
        metrics['pitch_recall'] = 0.0
        metrics['pitch_f1'] = 0.0
    
    # Additional metrics
    metrics['ref_note_count'] = len(ref_pitches)
    metrics['pred_note_count'] = len(pred_pitches)
    
    # Calculate pitch accuracy (how many predicted pitches match reference pitches)
    if len(ref_pitches) > 0 and len(pred_pitches) > 0:
        # Find closest matches between ref and pred pitches
        pitch_matches = 0
        for pred_pitch in pred_pitches:
            # Find the closest reference pitch
            closest_ref_idx = np.argmin(np.abs(ref_pitches - pred_pitch))
            if abs(ref_pitches[closest_ref_idx] - pred_pitch) <= TRANSCRIPTION_PARAMS['pitch_tolerance']:
                pitch_matches += 1
        
        metrics['pitch_accuracy'] = pitch_matches / len(pred_pitches)
    else:
        metrics['pitch_accuracy'] = 0.0
    
    return metrics


def validate_transcription_quality(cqt_features, transcription_result):
    """
    Validate the quality of a transcription based on the input features.
    
    Args:
        cqt_features (np.ndarray): Input CQT features
        transcription_result (dict): Result from PhinTranscriber.transcribe()
    
    Returns:
        dict: Quality metrics for the transcription
    """
    note_events = transcription_result['note_events']
    
    quality_metrics = {
        'note_density': len(note_events) / (cqt_features.shape[1] * 0.01),  # Notes per second
        'pitch_range': 0,
        'avg_velocity': 0,
        'temporal_coverage': 0
    }
    
    if note_events:
        # Calculate pitch range
        pitches = [event['pitch'] for event in note_events]
        quality_metrics['pitch_range'] = max(pitches) - min(pitches)
        
        # Calculate average velocity
        velocities = [event['velocity'] for event in note_events]
        quality_metrics['avg_velocity'] = sum(velocities) / len(velocities)
        
        # Calculate temporal coverage (how much of the audio has notes)
        total_duration = max(event['end_time'] for event in note_events)
        note_duration = sum(event['end_time'] - event['start_time'] for event in note_events)
        quality_metrics['temporal_coverage'] = note_duration / total_duration if total_duration > 0 else 0
    
    return quality_metrics


def compute_thai_scale_accuracy(note_events, fundamental_hz=440.0):
    """
    Compute how well the transcription adheres to the Thai 7-tone scale.
    
    Args:
        note_events (list): List of note events from transcription
        fundamental_hz (float): Fundamental frequency for reference
    
    Returns:
        dict: Metrics related to Thai scale accuracy
    """
    if not note_events:
        return {
            'thai_scale_adherence': 0.0,
            'average_deviation': 0.0,
            'thai_scale_notes_ratio': 0.0
        }
    
    # Convert MIDI pitches to frequencies
    frequencies = [440.0 * (2 ** ((pitch - 69) / 12)) for event in note_events for pitch in [event['pitch']]]
    
    # Define Thai 7-tone scale frequencies based on fundamental
    thai_scale_ratios = [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.789]  # Simplified ratios
    thai_scale_freqs = [fundamental_hz * ratio for ratio in thai_scale_ratios]
    
    # Calculate how many notes are close to Thai scale frequencies
    scale_matches = 0
    total_deviation = 0.0
    
    for freq in frequencies:
        # Find the closest Thai scale frequency
        closest_scale_freq = min(thai_scale_freqs, key=lambda x: abs(x - freq))
        deviation = abs(freq - closest_scale_freq)
        
        # Consider it a match if deviation is within tolerance
        if deviation / freq < 0.05:  # 5% tolerance
            scale_matches += 1
        
        total_deviation += deviation / freq  # Normalize by frequency
    
    adherence = scale_matches / len(frequencies) if frequencies else 0
    avg_deviation = total_deviation / len(frequencies) if frequencies else 0
    
    return {
        'thai_scale_adherence': adherence,
        'average_deviation': avg_deviation,
        'thai_scale_notes_ratio': scale_matches / len(frequencies) if frequencies else 0
    }


def evaluate_batch_transcriptions(ref_midi_paths, pred_midi_paths):
    """
    Evaluate a batch of transcriptions and return aggregate metrics.
    
    Args:
        ref_midi_paths (list): List of reference MIDI file paths
        pred_midi_paths (list): List of predicted MIDI file paths
    
    Returns:
        dict: Aggregate evaluation metrics
    """
    if len(ref_midi_paths) != len(pred_midi_paths):
        raise ValueError("Reference and predicted MIDI lists must have the same length")
    
    onset_f1_scores = []
    pitch_f1_scores = []
    
    for ref_path, pred_path in zip(ref_midi_paths, pred_midi_paths):
        onset_f1, pitch_f1 = evaluate_transcription(ref_path, pred_path)
        onset_f1_scores.append(onset_f1)
        pitch_f1_scores.append(pitch_f1)
    
    return {
        'mean_onset_f1': np.mean(onset_f1_scores),
        'std_onset_f1': np.std(onset_f1_scores),
        'mean_pitch_f1': np.mean(pitch_f1_scores),
        'std_pitch_f1': np.std(pitch_f1_scores),
        'median_onset_f1': np.median(onset_f1_scores),
        'median_pitch_f1': np.median(pitch_f1_scores),
        'min_onset_f1': np.min(onset_f1_scores),
        'max_onset_f1': np.max(onset_f1_scores),
        'min_pitch_f1': np.min(pitch_f1_scores),
        'max_pitch_f1': np.max(pitch_f1_scores),
        'total_evaluations': len(onset_f1_scores)
    }