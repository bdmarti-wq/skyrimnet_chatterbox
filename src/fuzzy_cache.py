# src/fuzzy_cache.py
import hashlib
import json
import re
import threading
import time
from difflib import SequenceMatcher
from queue import Queue
from typing import Optional

from pathlib import Path

from loguru import logger

from config import CONFIG
from .cache import ROOT_DIR, CACHE_AUDIO_DIR  # Import shared paths only

# Globals
FUZZY_QUEUE = Queue(maxsize=0)  # Non-blocking
FUZZY_LOCK = threading.Lock()
FUZZY_AUDIO_DICT = {}  # {stem: {norm_key: entry}}
FUZZY_SAVE_INTERVAL = 5.0
_last_fuzzy_save = 0
MAX_INDEX_SIZE = CONFIG.get_value('fuzzy_index_size', default=1000)  # For fuzzy per-stem
ENABLE_FUZZY = CONFIG.get_value('fuzzy_enable', default=True)  # New: Toggle fuzzy

def normalize_text(text: str) -> str:
    """Normalize text for fuzzy indexing."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    text_hash = hashlib.md5(cleaned.encode()).hexdigest()[:8]
    return f"{text_hash}_{cleaned}"

def string_similarity(s1: str, s2: str, threshold=0.75) -> float:
    """Similarity with RP boosts."""
    s1_clean = re.sub(r'[^\w\s]', '', s1.lower())
    s2_clean = re.sub(r'[^\w\s]', '', s2.lower())
    ratio = SequenceMatcher(None, s1_clean, s2_clean).ratio()
    # Boost (shared config via import if needed)
    boost_words = ['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp', 'oh', 'fuck', 'yes', 'aah']  # Or CONFIG
    boost_amount = 0.15
    if any(word in s1_clean + s2_clean for word in boost_words):
        ratio = min(1.0, ratio + boost_amount)
    return ratio

def try_fuzzy_audio_cache(audio_path: str = None, text_input: str = None, exaggeration: float = 0.5,
                          stem: str = None, threshold: float = None, quiet: bool = False) -> Optional[str]:
    """Fuzzy audio cache: SequenceMatcher on per-stem DB; returns path or None.
    REQUIRES stem (kwarg or from audio_path); detects/ warns on swap (text short like stem).
    Boosts sim for configurable short/moans words; min length tunable."""
    if not text_input or (
            text_input and len(text_input.strip()) < 3):  # Min guard (hardcode 3 if config fails; tunable below)
        if not quiet:
            logger.debug(f"Fuzzy skip: No/invalid text_input ({text_input[:20] if text_input else 'None'})")
        return None

    # Extract/fix stem (required; fallback if swapped)
    if stem is None:
        if stem is None:
            if audio_path:
                full_stem = Path(audio_path).stem.replace('_fixed', '')  # e.g., 'ba_beatricevoice'
                split_stem = full_stem.split('_')
                stem = split_stem[0] if len(split_stem) > 1 else full_stem  # Fallback to full if short
                if len(stem) < 3:  # Still short → warn but use full
                    stem = full_stem
                    logger.warning(f"Fuzzy stem fallback to full '{stem}' from '{audio_path}' (short prefix)")
        else:
            logger.warning(
                "Fuzzy: No audio_path or stem – cannot filter per-voice; using global fallback (inefficient)")
            stem = 'global'  # Fallback: All entries (less precise)
    # Auto-extend short stems (e.g., 'vp' → full from audio_path or cache)
    if len(stem) < 3 and audio_path:
        old_stem = stem
        full_stem = Path(audio_path).stem.replace('_fixed', '').replace('_padded', '').replace('_resampled',
                                                                                               '').replace(
            '_ui_resampled', '')
        match_full = re.match(r'^([a-zA-Z0-9_]+)_?([0-9a-f]{32,})?$', full_stem)
        candidate_stem = match_full.group(1) if match_full else full_stem
        if len(candidate_stem) >= 3 and candidate_stem not in ['global', 'audio_reuse']:
            stem = candidate_stem  # Use full if valid
            logger.debug(f"Fuzzy: Extended short stem '{old_stem}' to '{stem}' from path")

    threshold = threshold or CONFIG.get_value('fuzzy_threshold', default=0.75)

    # Detect swap: If text_input short/looks like stem (e.g., 'dlc1seranavoice'), warn + auto-swap
    min_length = CONFIG.get_value('fuzzy_min_length', default=3)
    if len(text_input.strip()) < 10 and re.match(r'^[a-z0-9_]+(voice|maid|npc)?$',
                                                 text_input.lower()):  # Heuristic: Looks like stem
        logger.warning(
            f"Fuzzy detect: Possible arg swap (text_input='{text_input}' too stem-like) – auto-fixing (use correct: audio_path, text_input=text, stem=voice)")
        text_input, audio_path = audio_path, text_input  # Swap back
        stem = Path(audio_path).stem.split('_')[0] if audio_path else stem  # Re-extract
        if len(text_input.strip()) < min_length:
            if not quiet:
                logger.debug(f"Fuzzy skip after swap: Text too short (<{min_length} chars)")
            return None

    if not quiet:
        per_stem_size = len(FUZZY_AUDIO_DICT.get(stem, {}))
        global_size = sum(len(entries) for entries in FUZZY_AUDIO_DICT.values()) if FUZZY_AUDIO_DICT else 0
        logger.debug(
            f"Trying fuzzy for stem '{stem}', text: '{text_input[:30]}...' | Global DB: {global_size} (this stem: {per_stem_size}) (threshold: {threshold:.2f})")

    # Early exit if no entries for stem
    if stem not in FUZZY_AUDIO_DICT or not FUZZY_AUDIO_DICT[stem]:
        if not quiet:
            logger.debug(
                f"Fuzzy MISS for '{text_input[:30]}...' (stem '{stem}'): No candidates in DB (global: {global_size})")
        return None

    candidates = list(FUZZY_AUDIO_DICT[stem].values())  # List for iteration
    best_match, best_path, best_sim = None, None, 0.0

    clean_text = re.sub(r'[^\w\s]', '', text_input.lower()).strip()  # Normalize text for sim
    if len(clean_text.split()) < min_length // 2:  # Word-based too (e.g., "aah..." → short)
        if not quiet:
            logger.debug(f"Fuzzy MISS early: Normalized text too short ('{clean_text}')")
        return None

    boost_words = CONFIG.get_value('fuzzy_boost_words',
                                   default=['ahh', 'mmm', 'ooh', 'yes'])  # Extended for your RP logs
    boost_amount = CONFIG.get_value('fuzzy_boost_amount', default=0.1)

    for entry in candidates:
        clean_entry = re.sub(r'[^\w\s]', '', entry['orig_text'].lower()).strip()
        raw_ratio = SequenceMatcher(None, clean_text, clean_entry).ratio()

        # Boost: Query + entry sides (additive; cap)
        sim_boost = entry.get('sim_boost', 0.0)  # Stored meta
        if boost_amount > 0 and boost_words:
            has_boost_query = any(word in clean_text for word in boost_words)
            has_boost_entry = any(word in clean_entry for word in boost_words)
            if has_boost_query or has_boost_entry:
                sim_boost += boost_amount * (1 if has_boost_query else 0.5) * (
                    1 if has_boost_entry else 0.5)  # Weighted: Full if both, half if one
        adjusted_sim = min(1.0, raw_ratio + sim_boost)

        if adjusted_sim > best_sim:
            best_sim = adjusted_sim
            best_match = entry['orig_text']
            best_path = entry['wav_path']
            if not quiet and adjusted_sim >= threshold * 0.9:  # Only log near-hits (e.g., 0.675+ for 0.75 thresh)
                logger.debug(
                    f"  Candidate: '{clean_entry[:30]}...' raw={raw_ratio:.3f} +boost={sim_boost:.3f} = {adjusted_sim:.3f}")

    if best_sim >= threshold:
        if not quiet:
            logger.info(
                f"Fuzzy cache HIT: '{text_input[:30]}...' ≈ '{best_match[:30]}...' (sim={best_sim:.3f} >= {threshold:.2f}, boost={boost_amount}) for stem '{stem}' -> {best_path}")
        return best_path
    else:
        if not quiet:
            logger.debug(
                f"Fuzzy MISS for '{text_input[:30]}...' (stem '{stem}'): best sim={best_sim:.3f} < {threshold:.2f} ({len(candidates)} candidates)")
            if best_match:
                logger.debug(f"  Closest: '{best_match[:30]}...' (sim={best_sim:.3f})")
        return None

def _save_fuzzy_audio_cache(save_all: bool = False):
    """Save FUZZY_AUDIO_DICT to JSON (throttled: min 5s or save_all=True; evict old if >1000 total)."""
    global _last_fuzzy_save
    now = time.time()
    if not save_all and (now - _last_fuzzy_save) < FUZZY_SAVE_INTERVAL:
        logger.trace(f"Save skipped: <{FUZZY_SAVE_INTERVAL}s since last (now at {now - _last_fuzzy_save:.1f}s)")
        return
    _last_fuzzy_save = now
    with FUZZY_LOCK:
        if not FUZZY_AUDIO_DICT:
            return
        temp_dict = {}
        total_entries = 0
        for stem, stem_entries in FUZZY_AUDIO_DICT.items():
            if isinstance(stem_entries, dict) and stem_entries:
                temp_dict[stem] = {}
                for norm_key, entry in stem_entries.items():
                    if 'wav_path' in entry and Path(entry['wav_path']).exists():
                        abs_path = Path(entry['wav_path'])
                        rel_path = abs_path.relative_to(ROOT_DIR)
                        save_entry = entry.copy()
                        save_entry['wav_path'] = str(rel_path)
                        temp_dict[stem][norm_key] = save_entry
                        total_entries += 1
                    else:
                        logger.trace(f"Save fuzzy: Skipping invalid for {stem}:{norm_key}")
                if not temp_dict[stem]:
                    del temp_dict[stem]
        # Global eviction if too big (e.g., >1000 total)
        if total_entries > MAX_INDEX_SIZE * 2:  # Over 2000: Prune oldest per-stem
            for stem in list(temp_dict):
                while len(temp_dict[stem]) > MAX_INDEX_SIZE:
                    oldest_key = min(temp_dict[stem], key=lambda k: temp_dict[stem][k].get('time_indexed', 0))
                    del temp_dict[stem][oldest_key]
                    logger.debug(f"Pruned old entry in {stem}: {oldest_key}")
        if total_entries == 0:
            return
        fuzzy_json = CACHE_AUDIO_DIR / "fuzzy_audio_cache.json"
        try:
            with open(fuzzy_json, 'w') as f:
                json.dump(temp_dict, f, indent=2)
            if save_all:
                logger.info(f"Full fuzzy save: {total_entries} entries across {len(temp_dict)} stems")
            else:
                logger.trace(f"Throttled fuzzy save: {total_entries} entries")  # TRACE: Less spam
        except Exception as e:
            logger.error(f"Save fuzzy cache failed: {e}")

# Load (called from init)
def load_fuzzy_cache():
    fuzzy_json = CACHE_AUDIO_DIR / "fuzzy_audio_cache.json"
    global FUZZY_AUDIO_DICT
    if fuzzy_json.exists():
        try:
            with open(fuzzy_json, 'r') as f:
                data = json.load(f)
            FUZZY_AUDIO_DICT = {}
            for stem, stem_entries in data.items():
                FUZZY_AUDIO_DICT[stem] = {}
                for norm_key, entry in stem_entries.items():
                    if 'wav_path' in entry:
                        rel_path = entry['wav_path']
                        full_path = ROOT_DIR / rel_path
                        if full_path.exists():
                            entry['wav_path'] = str(full_path)
                        else:
                            continue  # Skip invalid
                    FUZZY_AUDIO_DICT[stem][norm_key] = entry
            logger.info(f"Loaded fuzzy cache: {sum(len(entries) for entries in FUZZY_AUDIO_DICT.values())} entries across {len(FUZZY_AUDIO_DICT)} stems")
        except Exception as e:
            logger.warning(f"Load fuzzy cache failed: {e}")
            FUZZY_AUDIO_DICT = {}

# Export clear (for clear_cache)
def clear_fuzzy_cache():
    global FUZZY_AUDIO_DICT
    with FUZZY_LOCK:
        FUZZY_AUDIO_DICT.clear()
        logger.debug("Cleared fuzzy cache")


def _background_index_worker():
    global _fuzzy_save_counter
    while True:
        try:
            text, wav_path, voice_stem = FUZZY_QUEUE.get(timeout=1)
            orig_text = text
            min_length = CONFIG.get_value('fuzzy_min_length', default=3)
            if len(orig_text.strip()) < min_length:  # Optional: Skip indexing very short (e.g., "a" noise)
                logger.debug(f"Skipped indexing short text (<{min_length}): {orig_text[:10]}...")
                continue
            norm_key = normalize_text(text)
            boost_words = CONFIG.get_value('fuzzy_boost_words', default=['ahh', 'mmm', 'ooh', 'throbb', 'moan', 'gasp'])
            boost_amount = CONFIG.get_value('fuzzy_boost_amount', default=0.1)
            clean_text = re.sub(r'[^\w\s]', '', orig_text.lower())
            sim_boost = 0.0
            if boost_amount > 0 and boost_words:
                for word in boost_words:
                    if word in clean_text:
                        sim_boost = boost_amount
                        break  # Meta: Potential boost for this entry
            with FUZZY_LOCK:
                # Ensure per-stem sub-dict
                if voice_stem not in FUZZY_AUDIO_DICT:
                    FUZZY_AUDIO_DICT[voice_stem] = {}
                stem_dict = FUZZY_AUDIO_DICT[voice_stem]
                if norm_key in stem_dict:  # Dedup per-stem/norm_key
                    logger.debug(f"Skipped dup fuzzy index: {norm_key[:30]} ({voice_stem})")
                else:
                    if len(stem_dict) >= MAX_INDEX_SIZE:  # Per-stem cap
                        stem_dict.pop(next(iter(stem_dict)))  # Evict oldest in stem
                        logger.debug(f"Fuzzy per-stem full ({voice_stem}) – evicted oldest")
                    time_indexed = time.time()
                    stem_dict[norm_key] = {
                        'wav_path': wav_path,
                        'orig_text': orig_text,
                        'stem': voice_stem,
                        'sim_boost': sim_boost,  # Meta for future
                        'time_indexed': time_indexed
                    }
                    _fuzzy_save_counter += 1
                    logger.debug(f"Indexed fuzzy audio: {norm_key[:30]} -> {wav_path} (stem: {voice_stem}, boost={sim_boost})")
                # Throttle saves: Every 10 adds, queue >20, or 30s idle
                if (_fuzzy_save_counter >= 10 or FUZZY_QUEUE.qsize() > 20):
                    _save_fuzzy_audio_cache(save_all=False)  # Incremental
                    _fuzzy_save_counter = 0
            _save_fuzzy_audio_cache()  # Persist (add if not exists; thread-safe)
        except:
            # Idle timeout: Save if dirty (e.g., >30s no queue, but entries > prev)
            if FUZZY_QUEUE.empty() and FUZZY_AUDIO_DICT:  # Periodic full save
                time_since_last = time.time() - _last_fuzzy_save
                if time_since_last > 30:  # >30s idle → full safe save
                    _save_fuzzy_audio_cache(save_all=True)
            pass  # Continue loop


# Start worker (call in main.py init)
threading.Thread(target=_background_index_worker, daemon=True, name="FuzzyIndexer").start()