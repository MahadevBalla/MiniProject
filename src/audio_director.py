"""
Real-time audio intent detection module for Autonomous Camera Director.

This module provides real-time audio processing capabilities including:
- Voice Activity Detection (VAD) using Silero VAD
- MFCC and LPC feature extraction
- Direction of Arrival (DOA) estimation for microphone arrays
- Speech classification for lecturer vs audience detection
- Low-latency streaming audio processing

Author: AI Assistant
License: MIT
"""

import queue
import threading
import time
import numpy as np
import sounddevice as sd
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

import librosa
from librosa.feature import mfcc
from spafe.features.lpc import lpc as lpc_feat

# Silero VAD
import torch
from sklearn.linear_model import LogisticRegression

try:
    import pyroomacoustics as pra
    HAVE_PRA = True
    print("pyroomacoustics loaded successfully - DOA features available")
except ImportError:
    HAVE_PRA = False
    print("Warning: pyroomacoustics not available. DOA features disabled.")
except Exception as e:
    HAVE_PRA = False
    print(f"Warning: pyroomacoustics failed to load ({e}). DOA features disabled.")


class AudioDirector:
    """
    Real-time audio processing director for camera focus control.
    
    Features:
    - Low-latency VAD with Silero VAD
    - MFCC and LPC feature extraction on rolling buffer
    - Optional DOA estimation for microphone arrays
    - Speech classification for lecturer vs audience detection
    - Non-blocking audio processing in worker thread
    """
    
    def __init__(self,
                 samplerate=16000,
                 channels=1,
                 frame_ms=25,
                 hop_ms=10,
                 ring_buffer_sec=2.0,
                 device=None,
                 enable_doa=False,
                 callbacks=None):
        """
        Initialize AudioDirector.
        
        Args:
            samplerate (int): Audio sample rate in Hz (default: 16000)
            channels (int): Number of audio channels (default: 1)
            frame_ms (int): Frame size in milliseconds (default: 25)
            hop_ms (int): Hop size in milliseconds (default: 10)
            ring_buffer_sec (float): Ring buffer size in seconds (default: 2.0)
            device (int/str): Audio device ID or name (default: None for default device)
            enable_doa (bool): Enable DOA estimation (requires 4+ channels and pyroomacoustics)
            callbacks (dict): Dictionary of callback functions
        """
        self.sr = samplerate
        self.ch = channels
        self.frame = int(self.sr * frame_ms / 1000)
        self.hop = int(self.sr * hop_ms / 1000)
        
        # Ensure minimum frame size for VAD (Silero needs at least 512 samples at 16kHz)
        min_frame_samples = int(self.sr * 0.032)  # 32ms minimum
        if self.frame < min_frame_samples:
            self.frame = min_frame_samples
            print(f"Adjusted frame size to {self.frame} samples for VAD compatibility")
        self.ring_len = int(self.sr * ring_buffer_sec)
        self.device = device
        self.enable_doa = enable_doa and HAVE_PRA and channels >= 4
        self.cb = callbacks or {}
        
        # Audio processing queue and threading
        self.audio_q = queue.Queue(maxsize=32)
        self.stop_flag = threading.Event()
        self.worker = None
        self.stream = None
        
        # Ring buffer for audio data
        self.ring = np.zeros((self.ring_len, self.ch), dtype=np.float32)
        self.write_pos = 0
        self.frame_idx = 0
        
        # Load Silero VAD model
        self.vad_model, self.get_speech_timestamps, self.read_audio = self._load_silero()
        
        # VAD state management
        self.speaking = False
        self.last_vad_change = 0.0
        self.vad_hang_ms = 250  # Hysteresis to prevent rapid state changes
        self.min_speech_duration_ms = 500  # Minimum speech duration to avoid noise bursts
        
        # Classification model
        self.clf = LogisticRegression(max_iter=200, random_state=42)
        self.clf_ready = False
        self.train_X, self.train_y = [], []
        
        # DOA estimation setup
        self._doa = None
        if self.enable_doa:
            self._setup_doa()
        
        print(f"AudioDirector initialized: {self.sr}Hz, {self.ch}ch, "
              f"frame={self.frame} samples, hop={self.hop} samples")
        if self.enable_doa:
            print("DOA estimation enabled")
    
    def _load_silero(self):
        """Load Silero VAD model for voice activity detection."""
        try:
            # Load Silero VAD model (CPU optimized)
            model, utils = torch.hub.load(
                'snakers4/silero-vad', 
                'silero_vad', 
                trust_repo=True,
                force_reload=False
            )
            get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils
            print("Silero VAD model loaded successfully")
            return model, get_speech_timestamps, read_audio
        except Exception as e:
            print(f"Error loading Silero VAD: {e}")
            raise
    
    def _setup_doa(self):
        """Setup DOA estimation using MUSIC algorithm."""
        if not self.enable_doa:
            return
        
        try:
            # Configure MUSIC algorithm for DOA estimation
            nfft = 512
            self._doa = pra.doa.music.MUSIC(
                M=self.ch,  # Number of microphones
                L=1,        # Number of sources
                fs=self.sr, # Sample rate
                nfft=nfft,  # FFT size
                c=343,      # Speed of sound (m/s)
                num_src=1   # Number of sources to detect
            )
            print("DOA estimation configured")
        except Exception as e:
            print(f"Error setting up DOA: {e}")
            self.enable_doa = False
    
    def start(self):
        """Start audio streaming and processing."""
        try:
            # List available audio devices
            devices = sd.query_devices()
            print(f"Available audio devices: {len(devices)}")
            if self.device is not None:
                print(f"Using device: {devices[self.device]['name']}")
            
            # Initialize audio stream
            self.stream = sd.InputStream(
                samplerate=self.sr,
                channels=self.ch,
                dtype='float32',
                device=self.device,
                blocksize=self.hop,
                callback=self._audio_callback
            )
            
            # Start audio stream
            self.stream.start()
            print("Audio stream started")
            
            # Start processing worker thread
            self.worker = threading.Thread(target=self._process_loop, daemon=True)
            self.worker.start()
            print("Audio processing worker started")
            
        except Exception as e:
            print(f"Error starting AudioDirector: {e}")
            raise
    
    def stop(self):
        """Stop audio streaming and processing."""
        print("Stopping AudioDirector...")
        
        # Signal worker thread to stop
        self.stop_flag.set()
        
        # Wait for worker thread to finish
        if self.worker is not None:
            self.worker.join(timeout=2.0)
            if self.worker.is_alive():
                print("Warning: Worker thread did not stop gracefully")
        
        # Stop and close audio stream
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                print("Audio stream stopped")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
        
        print("AudioDirector stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio input callback function.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            # Handle audio stream status (overruns, underruns, etc.)
            if status.input_underflow:
                print("Warning: Audio input underflow")
            elif status.input_overflow:
                print("Warning: Audio input overflow")
        
        # Add audio data to processing queue
        try:
            self.audio_q.put_nowait(indata.copy())
        except queue.Full:
            # Drop audio data if queue is full to prevent blocking
            pass
    
    def _append_ring(self, block):
        """
        Append audio block to ring buffer.
        
        Args:
            block: Audio data block to append
        """
        n = block.shape[0]
        end = self.write_pos + n
        
        if end <= self.ring_len:
            # Simple case: block fits in remaining space
            self.ring[self.write_pos:end, :] = block
        else:
            # Wrap around case: block spans end of buffer
            first = self.ring_len - self.write_pos
            self.ring[self.write_pos:, :] = block[:first, :]
            self.ring[:end % self.ring_len, :] = block[first:, :]
        
        self.write_pos = end % self.ring_len
    
    def _read_latest(self, n_samples):
        """
        Read the most recent n_samples from ring buffer.
        
        Args:
            n_samples: Number of samples to read
            
        Returns:
            Audio data array of shape (n_samples, channels)
        """
        if n_samples > self.ring_len:
            n_samples = self.ring_len
        
        start = (self.write_pos - n_samples) % self.ring_len
        
        if start + n_samples <= self.ring_len:
            # Simple case: data is contiguous
            return self.ring[start:start+n_samples, :]
        else:
            # Wrap around case: data spans end of buffer
            first = self.ring_len - start
            return np.vstack([
                self.ring[start:, :],
                self.ring[:n_samples-first, :]
            ])
    
    def _vad_decide(self, chunk_mono):
        """
        Perform voice activity detection on audio chunk.
        
        Args:
            chunk_mono: Mono audio data
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        try:
            # Ensure minimum length for Silero VAD (512 samples = 32ms at 16kHz)
            min_length = 512
            if len(chunk_mono) < min_length:
                # Pad with zeros if too short
                chunk_mono = np.pad(chunk_mono, (0, min_length - len(chunk_mono)), mode='constant')
            elif len(chunk_mono) > min_length:
                # Take the last min_length samples
                chunk_mono = chunk_mono[-min_length:]
            
            # Convert to torch tensor (Silero expects 16k mono)
            wav = torch.from_numpy(chunk_mono.copy()).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            
            # Get speech probability
            with torch.no_grad():
                prob = self.vad_model(wav, 16000).item()
            
            return prob
        except Exception as e:
            # Silently handle VAD errors to avoid spam
            return 0.0
    
    def _extract_features(self, mono):
        """
        Extract MFCC and LPC features from mono audio.
        
        Args:
            mono: Mono audio data
            
        Returns:
            Dictionary containing extracted features
        """
        # Ensure minimum length
        if mono.shape[0] < self.frame:
            pad = self.frame - mono.shape[0]
            mono = np.pad(mono, (0, pad), mode='constant')
        
        # Use last exact frame
        mono = mono[-self.frame:]
        
        # Extract MFCC features
        try:
            # Use smaller FFT size for short frames
            n_fft = min(512, len(mono))
            if n_fft < 64:  # Minimum FFT size
                n_fft = 64
            
            mfcc_mat = mfcc(
                y=mono,
                sr=self.sr,
                n_mfcc=13,
                n_fft=n_fft,
                hop_length=len(mono),  # Use full frame
                win_length=len(mono),  # Use full frame
                center=False
            )
            mfcc_vec = mfcc_mat[:, -1]  # Last frame
            
            # Simple delta computation (first difference)
            if mfcc_mat.shape[1] > 1:
                delta_vec = mfcc_mat[:, -1] - mfcc_mat[:, -2]
            else:
                delta_vec = np.zeros_like(mfcc_vec)
                
        except Exception as e:
            # Silently handle MFCC errors
            mfcc_vec = np.zeros(13, dtype=np.float32)
            delta_vec = np.zeros(13, dtype=np.float32)
        
        # Extract LPC features
        try:
            lpc_order = 13
            lpc_coeffs = lpc_feat(
                mono.astype(np.float64),
                fs=self.sr,
                order=lpc_order
            )
            
            # Handle different return types from spafe
            if isinstance(lpc_coeffs, tuple):
                lpc_last = lpc_coeffs[0] if len(lpc_coeffs) > 0 else np.zeros(lpc_order)
            elif lpc_coeffs.ndim > 1:
                lpc_last = lpc_coeffs[-1]
            else:
                lpc_last = lpc_coeffs
                
            # Ensure correct size
            if len(lpc_last) != lpc_order:
                lpc_last = np.zeros(lpc_order, dtype=np.float32)
                
        except Exception as e:
            # Silently handle LPC errors
            lpc_last = np.zeros(13, dtype=np.float32)
        
        features = {
            "mfcc_mean": mfcc_vec.astype(np.float32),
            "mfcc_delta_mean": delta_vec.astype(np.float32),
            "lpc_mean": lpc_last.astype(np.float32),
        }
        
        return features
    
    def _maybe_emit_doa(self, block):
        """
        Perform DOA estimation if enabled.
        
        Args:
            block: Multi-channel audio block
        """
        if not self.enable_doa or self._doa is None:
            return
        
        try:
            # Transpose for pyroomacoustics (channels x samples)
            X = block.T
            
            # Perform DOA estimation
            self._doa.locate_sources(X, freq_range=[300, 3000])
            
            if len(self._doa.azimuth_recon) > 0:
                azimuth = float(np.degrees(self._doa.azimuth_recon[0]))
                
                # Call DOA callback
                if 'on_direction' in self.cb:
                    self.cb['on_direction'](azimuth)
                    
        except Exception as e:
            # Silently handle DOA errors to avoid spam
            pass
    
    def fit_classifier(self, X, y):
        """
        Train the speech classification model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        try:
            self.clf.fit(X, y)
            self.clf_ready = True
            print(f"Classifier trained on {len(X)} samples")
        except Exception as e:
            print(f"Error training classifier: {e}")
    
    def predict_role(self, feats):
        """
        Predict speaker role using trained classifier.
        
        Args:
            feats: Feature dictionary
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.clf_ready:
            return None, 0.0
        
        try:
            # Concatenate features
            x = np.hstack([
                feats['mfcc_mean'],
                feats['mfcc_delta_mean'],
                feats['lpc_mean']
            ]).reshape(1, -1)
            
            # Get prediction probabilities
            prob = self.clf.predict_proba(x)[0]
            idx = np.argmax(prob)
            
            return int(self.clf.classes_[idx]), float(prob[idx])
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def _process_loop(self):
        """Main audio processing loop running in worker thread."""
        vad_prob_smooth = 0.0
        alpha = 0.6  # Smoothing factor for VAD probability
        
        # VAD thresholds with hysteresis (balanced for normal speech)
        speak_thresh_on = 0.4   # Higher threshold to avoid breathing/noise
        speak_thresh_off = 0.3  # Higher threshold for stable detection
        
        print("Audio processing loop started")
        
        while not self.stop_flag.is_set():
            try:
                # Get audio block from queue
                block = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Add to ring buffer
            self._append_ring(block)
            
            # Convert to mono for VAD and feature extraction
            mono = block.mean(axis=1)
            
            # Voice Activity Detection
            vad_prob = self._vad_decide(mono)
            vad_prob_smooth = alpha * vad_prob_smooth + (1 - alpha) * vad_prob
            
            # Calculate RMS for fallback detection
            rms = np.sqrt(np.mean(mono**2))
            
            # Debug: Print VAD info occasionally (less frequent)
            if self.frame_idx % 500 == 0:
                print(f"Frame {self.frame_idx}: RMS={rms:.6f}, VAD_raw={vad_prob:.3f}, VAD_smooth={vad_prob_smooth:.3f}")
            
            # Fallback: Use RMS-based detection if VAD is not working well
            rms_threshold = 0.0005  # Higher threshold to avoid noise/breathing
            if vad_prob_smooth < 0.01:  # If VAD is not detecting anything
                vad_prob_smooth = 1.0 if rms > rms_threshold else 0.0
            
            # State management with hysteresis
            now = time.time()
            
            if not self.speaking and vad_prob_smooth >= speak_thresh_on:
                # Speech started - but only if it's been quiet for a while
                if (now - self.last_vad_change) * 1000 > self.vad_hang_ms:
                    self.speaking = True
                    self.last_vad_change = now
                    if 'on_speech_start' in self.cb:
                        try:
                            self.cb['on_speech_start']()
                        except Exception as e:
                            print(f"Speech start callback error: {e}")
            
            elif (self.speaking and 
                  vad_prob_smooth <= speak_thresh_off and 
                  (now - self.last_vad_change) * 1000 > self.vad_hang_ms):
                # Speech ended (with hang time and minimum duration)
                speech_duration = (now - self.last_vad_change) * 1000
                if speech_duration >= self.min_speech_duration_ms:
                    self.speaking = False
                    self.last_vad_change = now
                    if 'on_speech_end' in self.cb:
                        try:
                            self.cb['on_speech_end']()
                        except Exception as e:
                            print(f"Speech end callback error: {e}")
                else:
                    # Too short, ignore this speech event
                    pass
            
            # Feature extraction
            latest = self._read_latest(self.frame)
            mono_latest = latest.mean(axis=1)
            feats = self._extract_features(mono_latest)
            
            # Emit features
            if 'on_feature' in self.cb:
                try:
                    self.cb['on_feature'](self.frame_idx, feats)
                except Exception as e:
                    print(f"Feature callback error: {e}")
            
            # DOA estimation (if enabled and multi-channel)
            if self.enable_doa and self.ch >= 4:
                self._maybe_emit_doa(block)
            
            self.frame_idx += 1
        
        print("Audio processing loop ended")


def main():
    """Demo main function showing AudioDirector usage."""
    
    def on_speech_start():
        print("[Audio] Speech START")
    
    def on_speech_end():
        print("[Audio] Speech END")
    
    def on_feature(frame_idx, feats):
        # Print feature info every 10 frames to keep console clean
        if frame_idx % 10 == 0:
            fvec = np.hstack([
                feats['mfcc_mean'],
                feats['mfcc_delta_mean'],
                feats['lpc_mean']
            ])
            print(f"[Audio] Frame {frame_idx} feat_dim={fvec.size}")
    
    def on_direction(angle):
        print(f"[Audio] DOA {angle:.1f}°")
    
    # Setup callbacks
    callbacks = {
        'on_speech_start': on_speech_start,
        'on_speech_end': on_speech_end,
        'on_feature': on_feature,
        'on_direction': on_direction,
    }
    
    # Initialize AudioDirector
    director = AudioDirector(
        samplerate=16000,
        channels=1,
        enable_doa=False,  # Set to True if you have a 4+ channel mic array
        callbacks=callbacks
    )
    
    try:
        # Start audio processing
        director.start()
        print("AudioDirector running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Clean shutdown
        director.stop()
        print("Demo finished.")


if __name__ == "__main__":
    main()
