"""
Data pipeline module for Thai Isan Lute (Phin) Music Transcription

Handles downloading audio from YouTube and extracting Constant-Q Transform features
optimized for Thai 7-tone scale system.
"""
import os
import librosa
import numpy as np
import yt_dlp
from pathlib import Path
from scipy.signal import butter, lfilter
from .utils.constants import CQT_PARAMS, AUDIO_PARAMS


def download_youtube_audio(url, output_dir="./audio_sources", filename=None):
    """
    Download audio from YouTube and convert to WAV format.
    
    Args:
        url (str): YouTube URL to download
        output_dir (str): Directory to save the audio file
        filename (str): Custom filename (without extension)
    
    Returns:
        str: Path to the downloaded audio file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if filename is None:
        # Extract title from YouTube URL to use as filename
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
            }],
            'postprocessor_args': [
                '-ar', str(AUDIO_PARAMS['sample_rate'])
            ],
            'prefer_ffmpeg': True,
            'audioquality': '0',
            'extractaudio': True,
            'audioformat': 'wav',
        }
    else:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
            }],
            'postprocessor_args': [
                '-ar', str(AUDIO_PARAMS['sample_rate'])
            ],
            'prefer_ffmpeg': True,
            'audioquality': '0',
            'extractaudio': True,
            'audioformat': 'wav',
            'outtmpl': os.path.join(output_dir, filename + '.%(ext)s'),
        }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if filename is None:
            # Use the video title as the filename
            safe_title = "".join(c for c in info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filepath = os.path.join(output_dir, safe_title + '.wav')
        else:
            filepath = os.path.join(output_dir, filename + '.wav')
    
    return filepath


def normalize_audio(y, sr):
    """
    Normalize audio to optimal levels for Thai Isan music characteristics.
    
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
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    return y_filtered


def extract_phin_features(audio_path, sr=CQT_PARAMS['sr']):
    """
    Extract Constant-Q Transform features optimized for Thai Isan music.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int): Target sample rate
    
    Returns:
        np.ndarray: CQT spectrogram (time, frequency bins)
    """
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    
    # Normalize audio
    y = normalize_audio(y, sr)
    
    # Apply bandpass filter to focus on Phin frequencies
    y = apply_bandpass_filter(y, sr)
    
    # Compute Constant-Q Transform with parameters optimized for Thai 7-tone system
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


def preprocess_audio_batch(urls, output_dir="./audio_sources"):
    """
    Download and preprocess a batch of audio files from YouTube URLs.
    
    Args:
        urls (list): List of YouTube URLs
        output_dir (str): Directory to save the audio files
    
    Returns:
        list: Paths to the processed audio files
    """
    audio_paths = []
    
    for i, url in enumerate(urls):
        print(f"Processing audio {i+1}/{len(urls)}: {url}")
        
        try:
            # Download audio
            audio_path = download_youtube_audio(url, output_dir, f"thai_isan_{i+1:03d}")
            
            # Extract features to verify quality
            features = extract_phin_features(audio_path)
            print(f"Extracted features with shape: {features.shape}")
            
            audio_paths.append(audio_path)
            print(f"Successfully processed: {audio_path}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    return audio_paths


# Example usage
if __name__ == "__main__":
    # Example URLs for Thai Isan music (these are placeholders - you would need real URLs)
    sample_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder - replace with actual Thai Isan music
        "https://www.youtube.com/watch?v=P9pzm5b6FFY",  # Placeholder - replace with actual Thai Isan music
        "https://www.youtube.com/watch?v=Z0B7488J-iw"   # Placeholder - replace with actual Thai Isan music
    ]
    
    # Preprocess a batch of audio files
    # audio_paths = preprocess_audio_batch(sample_urls)
    print("Data pipeline module ready for use.")