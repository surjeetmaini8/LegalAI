"""Voice interface module for speech-to-text and text-to-speech"""
import os
from pathlib import Path
from typing import Optional
import tempfile
import whisper
from gtts import gTTS
import numpy as np

class VoiceHandler:
    
    def __init__(self, whisper_model: str = "base", tts_lang: str = "en"):
        
        self.whisper_model_name = whisper_model
        self.tts_lang = tts_lang
        self.whisper_model = None
        
        print(f"Loading Whisper model: {whisper_model}...")
        self._load_whisper()
    
    def _load_whisper(self):
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_file_path: str) -> dict:
       
        if self.whisper_model is None:
            raise ValueError("Whisper model not initialized")
        
        try:
            result = self.whisper_model.transcribe(
                audio_file_path,
                language="en",
                task="transcribe"
            )
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {
                'text': '',
                'language': None,
                'success': False,
                'error': str(e)
            }
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, format: str = "wav") -> dict:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            result = self.transcribe_audio(temp_path)
            return result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        
        if not text or not text.strip():
            raise ValueError("No text provided for text-to-speech")
        
        try:
            tts = gTTS(text=text, lang=self.tts_lang, slow=False)
            
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".mp3"
                )
                output_path = temp_file.name
                temp_file.close()
            
            tts.save(output_path)
            print(f"Audio saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error during text-to-speech: {e}")
            raise
    
    def speak_answer(self, answer: str) -> str:
        
        return self.text_to_speech(answer)
    
    def process_voice_query(
        self, 
        audio_file_path: str,
        qa_function: callable
    ) -> dict:
        
        transcription = self.transcribe_audio(audio_file_path)
        
        if not transcription['success']:
            return {
                'query': '',
                'answer': 'Error transcribing audio',
                'audio_path': None,
                'error': transcription['error']
            }
        
        query_text = transcription['text']
        
        try:
            answer_result = qa_function(query_text)
            
            if isinstance(answer_result, dict):
                answer_text = answer_result.get('answer', str(answer_result))
            else:
                answer_text = str(answer_result)
            
            audio_path = self.text_to_speech(answer_text)
            
            return {
                'query': query_text,
                'answer': answer_text,
                'audio_path': audio_path,
                'error': None
            }
            
        except Exception as e:
            print(f"Error processing voice query: {e}")
            return {
                'query': query_text,
                'answer': f'Error processing query: {str(e)}',
                'audio_path': None,
                'error': str(e)
            }
    
    def cleanup_audio_file(self, file_path: str):
        """Delete audio file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up audio file: {e}")
