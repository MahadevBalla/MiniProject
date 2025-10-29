# Microphone Troubleshooting Guide

## Issue: "Speech: OFF" - No Speech Detection

Your audio director is showing "Speech: OFF" because the microphone input levels are extremely low (0.000016 RMS). This indicates a microphone configuration issue.

## Quick Fixes

### 1. Check Windows Microphone Settings

1. **Right-click the speaker icon** in the system tray
2. Select **"Open Sound settings"**
3. Click **"Sound Control Panel"** (on the right)
4. Go to **"Recording"** tab
5. **Right-click your microphone** → **"Properties"**
6. Go to **"Levels"** tab
7. **Increase the microphone volume** to 80-100%
8. **Uncheck "Mute"** if it's checked
9. Click **"OK"**

### 2. Check Microphone Privacy Settings

1. Press **Windows + I** to open Settings
2. Go to **"Privacy & Security"** → **"Microphone"**
3. Make sure **"Microphone access"** is **ON**
4. Make sure **"Let apps access your microphone"** is **ON**
5. Scroll down and make sure **"Python"** or **"Python.exe"** is allowed

### 3. Test Your Microphone

1. Open **"Sound Recorder"** or **"Voice Recorder"**
2. Record a short audio clip
3. Play it back - you should hear your voice clearly
4. If you can't hear anything, the microphone hardware may be faulty

### 4. Try Different Microphone Devices

Your system has multiple microphones available:
- Device 0: Microsoft Sound Mapper - Input
- Device 1: Microphone Array (2- Realtek)
- Device 4: Primary Sound Capture Driver
- Device 5: Microphone Array (2- Realtek Audio)

Try running the audio director with a specific device:

```bash
# Test with different devices
python -c "
from src.audio_camera_integration import AudioCameraDirector
director = AudioCameraDirector()
director.audio_director.device = 1  # Try device 1
director.run()
"
```

### 5. Check Audio Driver

1. Press **Windows + X** → **"Device Manager"**
2. Expand **"Audio inputs and outputs"**
3. **Right-click your microphone** → **"Update driver"**
4. Select **"Search automatically for drivers"**

## Testing the Fix

After making changes, test with:

```bash
python test_devices.py
```

You should see RMS values above 0.001 for a working microphone.

## Alternative: Use RMS-Based Detection

If VAD still doesn't work, the system now has a fallback that uses audio level detection instead of AI-based voice detection. This should work even with very quiet microphones.

## Still Not Working?

1. **Try a different microphone** (USB headset, external mic)
2. **Check if other applications** can use the microphone
3. **Restart your computer** after changing settings
4. **Run as Administrator** (right-click Command Prompt → "Run as administrator")

## Expected Results

After fixing the microphone:
- RMS values should be > 0.001 when speaking
- You should see "🎤 SPEECH STARTED!" messages
- The camera system should show "Speech: ON" when you speak
- Bounding boxes should turn green during speech
