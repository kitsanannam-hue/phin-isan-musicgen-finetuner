"""
Main module for Thai Isan Lute (Phin) Music Transcription

This module provides the main entry point for the transcription system,
coordinating the data pipeline, model, and evaluation components.
"""
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.setup.environment import setup_environment
from src.data_pipeline.download import preprocess_audio_batch, extract_phin_features
from src.data_pipeline.feature_extraction import extract_note_events
from src.models.phin_transcriber import PhinTranscriber
from src.evaluation.transcription_eval import evaluate_transcription, evaluate_midi_accuracy
from src.utils.constants import CQT_PARAMS


def create_transcription_pipeline():
    """
    Create and return the complete transcription pipeline.
    
    Returns:
        dict: Components of the transcription pipeline
    """
    # Setup environment
    setup_summary = setup_environment()
    
    # Create model
    model = PhinTranscriber()
    
    # Return pipeline components
    return {
        'setup': setup_summary,
        'model': model,
        'constants': CQT_PARAMS
    }


def transcribe_audio_file(audio_path, model=None):
    """
    Transcribe a single audio file using the complete pipeline.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model (PhinTranscriber): Model instance to use for transcription
    
    Returns:
        dict: Transcription result
    """
    if model is None:
        model = PhinTranscriber()
    
    # Extract features
    print(f"Extracting features from {audio_path}...")
    cqt_features = extract_phin_features(audio_path)
    print(f"Features extracted with shape: {cqt_features.shape}")
    
    # Run transcription
    print("Running transcription...")
    transcription_result = model.transcribe(cqt_features)
    print(f"Transcription completed with {len(transcription_result['note_events'])} note events")
    
    return transcription_result


def save_transcription_as_midi(transcription_result, output_path):
    """
    Save transcription result as a MIDI file.
    
    Args:
        transcription_result (dict): Result from transcribe_audio_file
        output_path (str): Path to save the MIDI file
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


def process_youtube_urls(urls, model=None):
    """
    Process a list of YouTube URLs, downloading, transcribing, and evaluating.
    
    Args:
        urls (list): List of YouTube URLs
        model (PhinTranscriber): Model instance to use for transcription
    
    Returns:
        list: Results for each URL
    """
    if model is None:
        model = PhinTranscriber()
    
    results = []
    
    # Download audio files
    print("Downloading audio from YouTube URLs...")
    audio_paths = preprocess_audio_batch(urls)
    
    # Process each audio file
    for i, audio_path in enumerate(audio_paths):
        print(f"\nProcessing audio file {i+1}/{len(audio_paths)}: {audio_path}")
        
        # Transcribe audio
        transcription_result = transcribe_audio_file(audio_path, model)
        
        # Save as MIDI
        midi_path = audio_path.replace('.wav', '_transcribed.mid')
        save_transcription_as_midi(transcription_result, midi_path)
        
        # Store result
        result = {
            'audio_path': audio_path,
            'midi_path': midi_path,
            'transcription_result': transcription_result
        }
        results.append(result)
        
        print(f"Transcription saved to: {midi_path}")
    
    return results


def evaluate_transcription_quality(reference_midi_path, predicted_midi_path):
    """
    Evaluate the quality of a transcription against a reference.
    
    Args:
        reference_midi_path (str): Path to reference MIDI file
        predicted_midi_path (str): Path to predicted MIDI file
    
    Returns:
        dict: Evaluation metrics
    """
    # Evaluate using standard metrics
    onset_f1, pitch_f1 = evaluate_transcription(reference_midi_path, predicted_midi_path)
    
    # Load MIDI files for more detailed evaluation
    import pretty_midi
    ref_midi = pretty_midi.PrettyMIDI(reference_midi_path)
    pred_midi = pretty_midi.PrettyMIDI(predicted_midi_path)
    
    # Detailed evaluation
    detailed_metrics = evaluate_midi_accuracy(ref_midi, pred_midi)
    detailed_metrics['onset_f1'] = onset_f1
    detailed_metrics['pitch_f1'] = pitch_f1
    
    return detailed_metrics


def main():
    """
    Main function demonstrating the complete transcription pipeline.
    """
    print("Thai Isan Lute (Phin) Music Transcription Pipeline")
    print("=" * 55)
    
    # Create the pipeline
    pipeline = create_transcription_pipeline()
    print("Pipeline created successfully")
    
    # Example usage with a single audio file
    # Note: In practice, you would have actual Thai Isan music files
    print("\nExample: Transcribing a single audio file")
    print("-" * 40)
    
    # For demonstration, we'll skip actual processing since we don't have files
    # But in a real scenario, you would call:
    # transcription_result = transcribe_audio_file("path/to/thai_isan_audio.wav", pipeline['model'])
    
    print("Pipeline demonstration complete.")
    print("\nTo use the pipeline with real data:")
    print("1. Download Thai Isan music files or YouTube URLs")
    print("2. Call process_youtube_urls() with your URLs")
    print("3. Evaluate results with evaluate_transcription_quality()")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()