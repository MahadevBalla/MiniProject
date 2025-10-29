import queue
import threading
import time
import warnings

import librosa
import numpy as np
import sounddevice as sd

# Silero VAD
import torch
from librosa.feature import mfcc
from sklearn.linear_model import LogisticRegression
from spafe.features.lpc import lpc as lpc_feat

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

try:
    import pyroomacoustics as pra

    HAVE_PRA = True
except ImportError:
    HAVE_PRA = False


class AudioDirector:
    """Real-time audio processing director for camera focus control."""

    def __init__(
        self,
        samplerate=16000,
        channels=1,
        frame_ms=25,
        hop_ms=10,
        ring_buffer_sec=2.0,
        device=None,
        enable_doa=False,
        callbacks=None,
    ):
        """Initialize AudioDirector."""
        self.sr = samplerate
        self.ch = channels
        self.frame = int(self.sr * frame_ms / 1000)
        self.hop = int(self.sr * hop_ms / 1000)

        # Ensure minimum frame size for VAD
        min_frame_samples = int(self.sr * 0.032)
        if self.frame < min_frame_samples:
            self.frame = min_frame_samples

        self.ring_len = int(self.sr * ring_buffer_sec)
        self.device = device
        self.enable_doa = enable_doa and HAVE_PRA and channels >= 4
        self.cb = callbacks or {}

        # Audio processing queue and threading
        self.audio_q = queue.Queue(maxsize=32)
        self.stop_flag = threading.Event()
        self.worker = None
        self.stream = None

        # Ring buffer
        self.ring = np.zeros((self.ring_len, self.ch), dtype=np.float32)
        self.write_pos = 0
        self.frame_idx = 0

        # Load Silero VAD
        self.vad_model, self.get_speech_timestamps, self.read_audio = (
            self._load_silero()
        )

        # VAD state
        self.speaking = False
        self.last_vad_change = 0.0
        self.vad_hang_ms = 250
        self.min_speech_duration_ms = 500

        # Classification
        self.clf = LogisticRegression(max_iter=200, random_state=42)
        self.clf_ready = False

        print(f"AudioDirector initialized: {self.sr}Hz, {self.ch}ch")

    def _load_silero(self):
        """Load Silero VAD model."""
        try:
            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True, force_reload=False
            )
            (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            ) = utils
            return model, get_speech_timestamps, read_audio
        except Exception as e:
            print(f"Error loading Silero VAD: {e}")
            raise

    def start(self):
        """Start audio streaming and processing."""
        try:
            self.stream = sd.InputStream(
                samplerate=self.sr,
                channels=self.ch,
                dtype="float32",
                device=self.device,
                blocksize=self.hop,
                callback=self._audio_callback,
            )

            self.stream.start()
            self.worker = threading.Thread(target=self._process_loop, daemon=True)
            self.worker.start()
            print("AudioDirector started")

        except Exception as e:
            print(f"Error starting AudioDirector: {e}")
            raise

    def stop(self):
        """Stop audio streaming and processing."""
        self.stop_flag.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            if status.input_overflow:
                pass  # Silently handle
        try:
            self.audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _append_ring(self, block):
        """Append audio block to ring buffer."""
        n = block.shape[0]
        end = self.write_pos + n

        if end <= self.ring_len:
            self.ring[self.write_pos : end, :] = block
        else:
            first = self.ring_len - self.write_pos
            self.ring[self.write_pos :, :] = block[:first, :]
            self.ring[: end % self.ring_len, :] = block[first:, :]

        self.write_pos = end % self.ring_len

    def _read_latest(self, n_samples):
        """Read most recent n_samples from ring buffer."""
        if n_samples > self.ring_len:
            n_samples = self.ring_len

        start = (self.write_pos - n_samples) % self.ring_len

        if start + n_samples <= self.ring_len:
            return self.ring[start : start + n_samples, :]
        else:
            first = self.ring_len - start
            return np.vstack([self.ring[start:, :], self.ring[: n_samples - first, :]])

    def _vad_decide(self, chunk_mono):
        """Perform voice activity detection."""
        try:
            min_length = 512
            if len(chunk_mono) < min_length:
                chunk_mono = np.pad(
                    chunk_mono, (0, min_length - len(chunk_mono)), mode="constant"
                )
            elif len(chunk_mono) > min_length:
                chunk_mono = chunk_mono[-min_length:]

            wav = torch.from_numpy(chunk_mono.copy()).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

            with torch.no_grad():
                prob = self.vad_model(wav, 16000).item()

            return prob
        except Exception:
            return 0.0

    def _extract_features(self, mono):
        """Extract MFCC and LPC features."""
        if mono.shape[0] < self.frame:
            pad = self.frame - mono.shape[0]
            mono = np.pad(mono, (0, pad), mode="constant")

        mono = mono[-self.frame :]

        # MFCC
        try:
            n_fft = min(512, len(mono))
            if n_fft < 64:
                n_fft = 64

            mfcc_mat = mfcc(
                y=mono,
                sr=self.sr,
                n_mfcc=13,
                n_fft=n_fft,
                hop_length=len(mono),
                win_length=len(mono),
                center=False,
            )
            mfcc_vec = mfcc_mat[:, -1]

            if mfcc_mat.shape[1] > 1:
                delta_vec = mfcc_mat[:, -1] - mfcc_mat[:, -2]
            else:
                delta_vec = np.zeros_like(mfcc_vec)

        except Exception:
            mfcc_vec = np.zeros(13, dtype=np.float32)
            delta_vec = np.zeros(13, dtype=np.float32)

        # LPC
        try:
            lpc_order = 13
            lpc_coeffs = lpc_feat(mono.astype(np.float64), fs=self.sr, order=lpc_order)

            if isinstance(lpc_coeffs, tuple):
                lpc_last = lpc_coeffs[0] if len(lpc_coeffs) > 0 else np.zeros(lpc_order)
            elif lpc_coeffs.ndim > 1:
                lpc_last = lpc_coeffs[-1]
            else:
                lpc_last = lpc_coeffs

            if len(lpc_last) != lpc_order:
                lpc_last = np.zeros(lpc_order, dtype=np.float32)

        except Exception:
            lpc_last = np.zeros(13, dtype=np.float32)

        return {
            "mfcc_mean": mfcc_vec.astype(np.float32),
            "mfcc_delta_mean": delta_vec.astype(np.float32),
            "lpc_mean": lpc_last.astype(np.float32),
        }

    def _process_loop(self):
        """Main audio processing loop."""
        vad_prob_smooth = 0.0
        alpha = 0.6

        speak_thresh_on = 0.4
        speak_thresh_off = 0.3

        while not self.stop_flag.is_set():
            try:
                block = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._append_ring(block)
            mono = block.mean(axis=1)

            # VAD
            vad_prob = self._vad_decide(mono)
            vad_prob_smooth = alpha * vad_prob_smooth + (1 - alpha) * vad_prob

            # RMS fallback
            rms = np.sqrt(np.mean(mono**2))
            rms_threshold = 0.0005
            if vad_prob_smooth < 0.01:
                vad_prob_smooth = 1.0 if rms > rms_threshold else 0.0

            now = time.time()

            if not self.speaking and vad_prob_smooth >= speak_thresh_on:
                if (now - self.last_vad_change) * 1000 > self.vad_hang_ms:
                    self.speaking = True
                    self.last_vad_change = now
                    if "on_speech_start" in self.cb:
                        try:
                            self.cb["on_speech_start"]()
                        except Exception:
                            pass

            elif (
                self.speaking
                and vad_prob_smooth <= speak_thresh_off
                and (now - self.last_vad_change) * 1000 > self.vad_hang_ms
            ):
                speech_duration = (now - self.last_vad_change) * 1000
                if speech_duration >= self.min_speech_duration_ms:
                    self.speaking = False
                    self.last_vad_change = now
                    if "on_speech_end" in self.cb:
                        try:
                            self.cb["on_speech_end"]()
                        except Exception:
                            pass

            # Feature extraction
            latest = self._read_latest(self.frame)
            mono_latest = latest.mean(axis=1)
            feats = self._extract_features(mono_latest)

            if "on_feature" in self.cb:
                try:
                    self.cb["on_feature"](self.frame_idx, feats)
                except Exception:
                    pass

            self.frame_idx += 1
