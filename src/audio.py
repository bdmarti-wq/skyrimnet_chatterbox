import os


# Set torchaudio backend: Prefer 'sox' for speed/reliability, fallback to 'soundfile'
def set_torchaudio_backend():
    """Configure torchaudio backend with SOX preference (Windows-friendly)."""
    # Set env var BEFORE importing torchaudio
    preferred_backend = 'sox'
    fallback_backend = 'soundfile'

    # Early set to env (torchaudio reads on import)
    os.environ['TORCHAUDIO_BACKEND'] = fallback_backend  # Default fallback

    try:
        import torchaudio  # Temp import to check backends
        available_backends = torchaudio.list_audio_backends()
        if preferred_backend in available_backends:
            os.environ['TORCHAUDIO_BACKEND'] = preferred_backend
            print(f"✓ Using torchaudio backend: {preferred_backend} (faster resample/metadata)")
            return preferred_backend
        else:
            print(f"⚠ SOX not available (install via 'choco install sox'). Using: {fallback_backend}")
            return fallback_backend
    except ImportError:
        print(f"❌ torchaudio not installed—audio ops will fail.")
        return None
    except Exception as e:
        print(f"❌ Backend check failed: {e}. Using fallback: {fallback_backend}")
        return fallback_backend

