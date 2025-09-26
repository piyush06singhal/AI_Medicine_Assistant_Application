"""
Audio Processing Module for AI Medical Assistant
Handles voice input for symptom description using speech-to-text.
"""

import streamlit as st
import speech_recognition as sr
import pyaudio
import wave
import tempfile
import os
from pathlib import Path
import logging
from typing import Optional, Tuple
import io
import base64
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing for voice-based symptom input."""
    
    def __init__(self):
        """Initialize audio processor."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        logger.info("Audio processor initialized")
    
    def record_audio(self, duration: int = 5) -> Optional[bytes]:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio data as bytes or None if failed
        """
        try:
            with self.microphone as source:
                logger.info(f"Recording audio for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                
                # Convert to bytes
                audio_data = audio.get_wav_data()
                return audio_data
                
        except sr.WaitTimeoutError:
            logger.warning("Audio recording timeout")
            return None
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
    
    def save_audio_to_file(self, audio_data: bytes, filename: str = None) -> str:
        """
        Save audio data to temporary file.
        
        Args:
            audio_data: Audio data as bytes
            filename: Optional filename
            
        Returns:
            Path to saved audio file
        """
        try:
            if not filename:
                filename = f"audio_{int(time.time())}.wav"
            
            temp_dir = Path("temp_audio")
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Audio saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None
    
    def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Tuple[str, float]:
        """
        Transcribe audio to text using speech recognition.
        
        Args:
            audio_data: Audio data as bytes
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Create AudioData object
            audio = sr.AudioData(audio_data, 16000, 2)  # 16kHz, 16-bit, stereo
            
            # Transcribe using Google Speech Recognition
            text = self.recognizer.recognize_google(audio, language=language)
            confidence = 0.8  # Google doesn't provide confidence, use default
            
            logger.info(f"Audio transcribed: {text[:50]}...")
            return text, confidence
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return "", 0.0
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "", 0.0
    
    def transcribe_audio_file(self, file_path: str, language: str = "en") -> Tuple[str, float]:
        """
        Transcribe audio from file.
        
        Args:
            file_path: Path to audio file
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=language)
                confidence = 0.8
                
                logger.info(f"File transcribed: {text[:50]}...")
                return text, confidence
                
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return "", 0.0
    
    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        Get duration of audio data.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Duration in seconds
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Get duration using wave module
            with wave.open(temp_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
            
            # Clean up
            os.unlink(temp_path)
            
            return duration
            
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0
    
    def convert_to_base64(self, audio_data: bytes) -> str:
        """
        Convert audio data to base64 string.
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Base64 encoded string
        """
        try:
            return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            return ""
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            temp_dir = Path("temp_audio")
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    file.unlink()
                temp_dir.rmdir()
                logger.info("Temporary audio files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

# Global instance
audio_processor = AudioProcessor()

def record_and_transcribe(duration: int = 5, language: str = "en") -> Tuple[str, float, Optional[bytes]]:
    """
    Convenience function to record and transcribe audio.
    
    Args:
        duration: Recording duration in seconds
        language: Language code for transcription
        
    Returns:
        Tuple of (transcribed_text, confidence, audio_data)
    """
    # Record audio
    audio_data = audio_processor.record_audio(duration)
    
    if audio_data is None:
        return "", 0.0, None
    
    # Transcribe audio
    text, confidence = audio_processor.transcribe_audio(audio_data, language)
    
    return text, confidence, audio_data

def transcribe_uploaded_audio(uploaded_file, language: str = "en") -> Tuple[str, float]:
    """
    Transcribe uploaded audio file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        language: Language code for transcription
        
    Returns:
        Tuple of (transcribed_text, confidence)
    """
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Transcribe
        text, confidence = audio_processor.transcribe_audio_file(str(file_path), language)
        
        # Clean up
        file_path.unlink()
        
        return text, confidence
        
    except Exception as e:
        logger.error(f"Error transcribing uploaded audio: {e}")
        return "", 0.0
