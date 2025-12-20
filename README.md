


Windows setup meant for use with SkyrimNet either locally (or on a local secondary PC) install of Zonos.
- Supports NVIDIA Blackwell GPUs; Ampere or below are currently not supported.
- Cache files are stored in the `cache` folder.
- Output files are saved in `output_temp` under timestamped subfolders.

Assumes that Python 3.12 is already installed:
https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

To install other needed files:

`1_Install.bat` 

To run:

`2_Start.bat` 

This should start in a high priority process window.

`2_Start_ML.bat` will start a multilingual version of Chatterbox.

Be sure to set the Language section in the SkyrimNet Zonos tab to match the language you want.

SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}

# Note on values used for voice generation

Currently these values are hardcoded.

- temperature = 0.9
- min_p = 0.07
- top_p = 1.0
- repetition_penalty = 2.0
- cfg_weight = 0.0
- exaggeration = 0.7

Edit `skyrimnet_config.txt` if you would like to use the controls from the SkyrimNet UI.

Be sure to change the values in SkyrimNet UI before loading your save; some value combinations may cause errors.
Randomize Seed should usually be Disabled.

All other SkyrimNet Audio fields are ignored by this integration.


- MIN_P: Modern way to control speech quality (newer method)  

- TOP_P: Classic way to control speech variety (older method) Probably want to keep this at 1.0 (disabled)

- TEMPERATURE: Controls how creative/random the voice sounds
 *Note: This uses the "linear" slider in SkyrimNet UI when set to "api"

- REPETITION_PENALTY: Stops the voice from repeating words too much
  * Note: This uses the 'confidence' slider in SkyrimNet UI when set to "api"

- CFG_WEIGHT: Controls speech pacing and guidance strength
  * Note: This uses the "cfg_scale" slider in SkyrimNet UI when set to "api"

- EXAGGERATION: Controls emotion intensity and expression
  *  Note: This uses the "quadratic" slider in SkyrimNet UI when set to "api"


---
# New features

- Offline voice testing page
  - Test any loaded voice outside the game, adjust generation and post-processing parameters, and save results directly into the audio cache for instant in-game reuse. This is the fastest way to dial in a voice without reloading a save.

- Fuzzy cache for text similarity
  - Reuses cached audio for similar text to avoid regeneration. Example: a cached "Yes sir." can satisfy inputs like "Yes." or "Yes, indeed" when above the fuzzy threshold. This can be 200x+ faster than fresh generation. Sensitivity is configurable and you can exclude specific words (e.g., names) to prevent mismatched returns such as "Yes Bob" for "Yes Fred".

- Built-in post-processing
  - Clean up TTS output with optional EQ, notch filtering, denoising, normalization, gating of trailing artifacts, fades, speaking rate changes, and more. Use this to reduce hiss or ringing, tame harshness, smooth word endings, or match loudness across lines.

- Robust fixes for ghost voices and trailing artifacts
  - Short-text padding mechanism greatly reduces leading/trailing ghost audio common with some voices. Configure a minimum word length threshold and a padding token; the system generates extra text including the token, then trims away the token’s audio. This slightly increases generation time for very short lines, but the aggressive short-line caching offsets it in practice.


<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life across languages. It's also the first open source TTS model to support **emotion exaggeration control** with robust **multilingual zero-shot voice cloning**. Try the english only version now on our [English Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox). Or try the multilingual version on our [Multilingual Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Supported Languages 
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
# Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.




# Acknowledgements
- [Chatterbox for Skyrimnet](https://github.com/langfod/chatterbox)
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Notes on watermarking

Some upstream Chatterbox distributions may include neural watermarking (e.g., Perth). This fork does not require any external watermarking packages and does not embed code that depends on them. If you wish to experiment with watermarking, refer to the upstream project for details.


# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.

---

## Developer notes: single-source modules and refactors

This project now centralizes several cross-cutting concerns into small focused utilities to keep the code DRY, easier to test, and consistent across layers (UI/bridge/pipeline/caches):

- Voice parameter merging: `src/voice_params.py::get_voice_params`
  - Precedence: models defaults < globals (tts/audio) < per-voice overrides < per-request overrides.
  - `config.get_voice_params(...)` delegates to this canonical merger with LRU caching.

- Audio path validation: `src/audio/paths.py`
  - `sanitize_input_path(path)` for lightweight UI/bridge sanity.
  - `validate_user_audio(path, min_dur)` as the single authority for file existence/duration/waveform checks.
  - Used by pipeline phases and caches for consistent behavior.

- Cache key generation: `src/cache_keys.py::generate_audio_cache_key`
  - Single source of truth for audio/fuzzy cache keys and context-generated keys.

- Conditionals validity: `src/generate/cache/conditionals_utils.py`
  - `is_valid_conditionals(obj)` and `is_mock_conditionals(obj)` shared by caches and phases.

- Output formatting for UI/API: `src/output_utils.py::format_output`
  - Standardizes return shape and provides a reusable silence WAV fallback (via `src/audio_fallbacks.py`).

- Config persistence service: `src/config/service.py::save_voice_overrides`
  - Diffs overrides against the effective baseline and writes only changes; creates timestamped backups.

- Post-processing modularization: `src/audio/`
  - Stateless numpy-based transforms in `transforms.py` (trim, gate, filters, rate, fade).
  - Pipeline phase delegates to these helpers to stay thin and testable.



---

# Audio pipeline overview and where settings apply

The generation pipeline runs in phases (see `src/generate/pipeline/coordinator.py`):

1. Inputs Validation: basic checks of text, language, and selected voice.
2. Cache Check: tries full audio cache first; if miss, can consult the fuzzy cache for similar text re-use.
3. Voice Processing: loads/merges voice parameters and conditionals; applies any pre-adjustment.
4. TTS Generation: produces raw waveform from the model using your TTS settings (temperature, cfg_weight, etc.).
5. Post-Processing: optional audio clean-up and formatting (filters, gating, normalization, fades, speaking-rate changes).
6. Output: writes audio to disk, updates caches, and returns the path.

Post-processing runs only if enabled and uses the parameters described below. The Voice Testing page exposes these controls so you can hear changes instantly and then save per-voice overrides.

---

# Per-voice adjustable settings (what they do and why)

You can define global defaults in `config.json` and override them per voice under the `voices` section. The UI Voice Testing page lets you experiment and save overrides. Below is a concise guide to the most important knobs.

TTS generation (globals.tts)
- temperature: Increases randomness and expressiveness as it rises. Lower for consistency; higher for variety or emotional delivery.
- min_p: Modern nucleus-like threshold; filters out very low-probability tokens. Raise slightly to reduce mumbling artifacts; lower to allow more nuanced outputs.
- top_p: Classic nucleus sampling cap. Keep at 1.0 to disable; reduce to limit vocab diversity for steadier delivery.
- repetition_penalty: Discourages repeating words/phrases. Increase if you hear loops; too high can make speech stilted.
- cfg_weight: Guidance strength that also affects pacing. Lower values often slow speech and increase deliberation; higher can make it brisk and more on-prompt.
- exaggeration: Scales expressivity/emotion. Raise for dramatic reads; lower for neutral, broadcast-style delivery.
- max_new_tokens / min_new_tokens: Hard bounds on generation length. Increase max for very long lines; tune min to avoid premature cutoffs.
- max_cache_len / stride_length / generate_token_backend / compile_t3 / warmup_t3 / re_optimize_on_reload: Advanced performance controls for expert users; leave defaults unless optimizing throughput/latency.

Audio post-processing (globals.audio)
- enable_post_processing: Master switch. Turn on to apply the rest of the post-processing chain.
- enable_post_resample: If enabled, resamples output to your target sample rate. Leave off if your downstream expects model SR.
- enable_post_jit_gain: Enables an inline gain/normalization pass optimized for speed.
- enable_post_voice_processing: Allows per-voice post adjustments to take effect.
- enable_pre_adjustment: Apply small pre-normalization before other steps to improve filter behavior.
- eq_gain_db / eq_cutoff_hz: Gentle shelving EQ to tame harshness or brighten dull voices. Positive gain brightens above cutoff; negative softens.
- notch_gain_db / notch_low_hz / notch_high_hz: Band cut to remove ringing/whine between low/high. Use small negative gains (e.g., -6 to -12 dB) to reduce metallic artifacts.
- highpass_cutoff_hz: Removes low hum/rumble. Set around 50–80 Hz for cleaner speech without thinning the voice.
- fade_ms: Applies short fade-in/out to avoid clicks at boundaries. 10–30 ms is typical.
- speaking_rate: Time-stretch without pitch shift. <1.0 slows for gravitas; >1.0 speeds responses for snappiness.
- normalize_method: Peak (default) is fast and transparent; loudness-based options (if enabled) aim for consistent perceived volume.
- gain_max_limit / gain_target_max / max_gain: Prevents over-amplification. Raise cautiously if outputs are too quiet after trimming.
- noise_floor_db / trim_threshold_db: Controls silence detection and trimming. Lower thresholds keep more tails; higher trims more aggressively.
- enable_denoising / enable_denoise_normalize / n_fft_denoise / denoise_median_ksize / denoise_target_band_low/high: Spectral denoise for hiss/whistle bands. Target only the noisy band to avoid lisping.
- enable_audio_padding / base_audio_pad_sec / tiny_audio_pad_multiplier / tiny_threshold_sec: Adds programmable silence padding; tiny clips can get extra padding for better UX.
- trailing_silence_db: Threshold to classify trailing audio as silence when trimming. Adjust if you still hear breath/ghost tails.
- n_fft / hop_length / n_mels: Analysis parameters; keep defaults unless you know what you’re doing.
- ebu_post_gain_db / ebu_true_peak: Optional loudness alignment helpers if using EBU-style pipelines.
- max_short_word_len: Defines what “short” means for padding heuristics and caching.
- max_cache_entries / max_cache_size_mb: Limits for audio cache size and count.
- short_padding_threshold / short_padding_token: If the input text has fewer than this many words, the system appends the token to stabilize synthesis, then trims it away. Use a harmless, easily trimmable token.

Fuzzy cache (globals.fuzzy)
- enable_fuzzy_cache: Enables similarity-based reuse of audio.
- fuzzy_threshold: Similarity needed (0–1). Raise for safer matches; lower to increase hit rate.
- fuzzy_boost_amount / fuzzy_boost_words: Extra weight for filler/vocalized pauses (e.g., “ahh”, “mmm”) to better match short interjections.
- fuzzy_force_skip_words: Words to ignore when matching (e.g., character names) to prevent wrong-person matches.
- fuzzy_index_size: Limit on the fuzzy index entries. Increase for larger corpora; watch memory usage.
- fuzzy_artifact_threshold_hz: Helps detect/avoid returning clips with high-frequency artifact energy.

Per-voice overrides
- Any of the above can be set per voice under `voices.{voice_name}`. Example:
  - Increase `speaking_rate` for a brisk delivery on one character.
  - Enable a narrow `notch_gain_db` cut only for a voice that rings.
  - Raise `short_padding_threshold` and set a custom `short_padding_token` for a ghost-prone voice.

UI Voice Testing page
- The test page lets you pick a voice, type lines, toggle post-processing, tweak thresholds, speed, EQ/notch, and padding. When satisfied, click save to persist overrides back to `config.json` with a timestamped backup.