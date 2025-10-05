# skyrimnet_chatterbox.py (Imports and generate shell)
import functools
import warnings

from config import DEVICE, DTYPE, _USE_API_MODE, load_skyrimnet_config, get_config_value, ENABLE_MEMORY_CACHE, \
    ENABLE_DISK_CACHE, MODEL, MULTILINGUAL
# Lazy import inside generate (avoids global Gradio scan/inference)
from src.audio_utils import set_torchaudio_backend
from src.fuzzy_cache import load_fuzzy_cache

backend = set_torchaudio_backend()

import gradio as gr
from argparse import ArgumentParser
import torch
from src.cache import (
    init_conditional_memory_cache, clear_cache_files, clear_output_directories
)
from loguru import logger

import warnings
warnings.filterwarnings('ignore', message=r'.*torchaudio._backend.utils.info.*')
warnings.filterwarnings('ignore', message=r'.*deprecated.*torchaudio.*')

def load_model():
    global MODEL, MULTILINGUAL
    if MODEL is None:
        if MULTILINGUAL:
            logger.info("Loading Multilingual Model")
            from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
        else:
            logger.info("Loading English Model")
            from src.chatterbox.tts import ChatterboxTTS as Chatterbox
        MODEL = Chatterbox.from_pretrained(DEVICE)
        MODEL.t3.to(dtype=DTYPE)
        MODEL.conds.t3.to(dtype=DTYPE)
        torch.cuda.empty_cache()
    return MODEL


def generate(model, text, language_id="en", audio_prompt_path=None, exaggeration=0.5, temperature=0.8, seed_num=0,
             cfgw=0, min_p=0.05, top_p=1.0, repetition_penalty=1.2, cache_uuid=0):
    """
    UI shell: Keeps server-compatible sig. Minimal setup, delegates to generate_audio.
    """
    if model is None:
        model = load_model()

    if not text:
        logger.warning("No text provided – returning empty")
        return ""  # Or dummy path

    if seed_num != 0:
        from src.generate_audio import set_seed  # Lazy
        set_seed(int(seed_num))

    # Lazy import + call (no global exposure)
    from src.generate_audio import generate_audio
    result = generate_audio(
        model, text, audio_prompt_path,  # None as-is (no "" coercion)
        float(exaggeration), int(cache_uuid),
        float(temperature), float(cfgw), float(min_p), float(top_p), float(repetition_penalty),
        language_id, int(seed_num),
        ENABLE_MEMORY_CACHE, ENABLE_DISK_CACHE
    )
    return result



### SkyrimNet Zonos Emulated
@functools.cache
def cpp_uuid_to_seed(uuid_64: int) -> int:
    """
    Convert a 64-bit UUID to a valid PyTorch seed (0 to 2^32 - 1).
    Uses hash() for better distribution across the seed space.
    """
    return abs(hash(uuid_64)) % (2 ** 32)


def generate_audio_ui(
        model_choice=None,
        text="On that first day from Saturalia, My missus gave for me, A big bowl of moon sugar!",
        language="en",
        speaker_audio=None,
        prefix_audio=None,
        e1=None,
        e2=None,
        e3=None,
        e4=None,
        e5=None,
        e6=None,
        e7=None,
        e8=None,
        vq_single=None,
        fmax=None,
        pitch_std=None,
        speaking_rate=None,
        dnsmos_ovrl=None,
        speaker_noised: bool = None,
        cfg_scale=0.3,
        top_p=1.0,
        top_k=None,
        min_p=0.5,
        linear=None,
        confidence=None,
        quadratic=None,
        uuid=-1,
        randomize_seed: bool = False,
        unconditional_keys: list = None
):
    """Generate audio using configurable parameter system"""

    if _USE_API_MODE:
        defaults, modes = {}, {}
    else:
        defaults, modes, flags = load_skyrimnet_config()

    api_temperature = linear if linear is not None else None
    api_min_p = min_p if min_p is not None else None
    api_top_p = top_p if top_p is not None else None
    api_repetition_penalty = confidence if confidence is not None else None
    api_cfg_weight = cfg_scale if cfg_scale is not None else None
    api_exaggeration = quadratic if quadratic is not None else None

    final_temperature = get_config_value('temperature', api_temperature, defaults, modes, _USE_API_MODE)
    final_min_p = get_config_value('min_p', api_min_p, defaults, modes, _USE_API_MODE)
    final_top_p = get_config_value('top_p', api_top_p, defaults, modes, _USE_API_MODE)
    final_repetition_penalty = get_config_value('repetition_penalty', api_repetition_penalty, defaults, modes,
                                                _USE_API_MODE)
    final_cfg_weight = get_config_value('cfg_weight', api_cfg_weight, defaults, modes, _USE_API_MODE)
    final_exaggeration = get_config_value('exaggeration', api_exaggeration, defaults, modes, _USE_API_MODE)

    logger.debug(
        f"Final parameters - temp: {final_temperature}, min_p: {final_min_p}, top_p: {final_top_p}, rep_penalty: {final_repetition_penalty}, cfg_weight: {final_cfg_weight}, exaggeration: {final_exaggeration}")

    # Important - use server supplied uuid for consistent seed
    seed_num = cpp_uuid_to_seed(uuid)

    # Lazy import + call (via kwargs for sig safety; model=MODEL from global)
    from src.generate_audio import generate_audio
    result = generate_audio(
        model=MODEL,
        text=text,
        audio_prompt_path=speaker_audio,  # None ok
        seed_num=seed_num,
        cache_uuid=uuid,
        exaggeration=final_exaggeration,
        temperature=final_temperature,
        cfgw=final_cfg_weight,
        min_p=final_min_p,
        top_p=final_top_p,
        repetition_penalty=final_repetition_penalty,
        language_id=language  # kwarg-safe
    )
    return result, uuid  # Matches outputs



with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize",
                lines=5,
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File",
                               value=None)

            exaggeration = gr.Slider(0.25, 2, step=.05,
                                     label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.55)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.0)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables",
                                  value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8",
                                  value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=2.0)
            language_id = gr.Dropdown([
                "ar",
                "da",
                "de",
                "el",
                "en",
                "es",
                "fi",
                "fr",
                "he",
                "hi",
                "it",
                "ja",
                "ko",
                "ms",
                "nl",
                "no",
                "pl",
                "pt",
                "ru",
                "sv",
                "sw",
                "tr",
                "zh"], value="en", multiselect=False, label="Language", info="Language only for multilanguage model")
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio", type="filepath", autoplay=True)

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state, text, language_id, ref_wav, exaggeration, temp, seed_num,
            cfg_weight, min_p, top_p, repetition_penalty,
        ],
        outputs=audio_output,
    )

    model_choice = gr.Textbox(visible=False)
    language = gr.Textbox(visible=False)
    speaker_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File",
                             value=None, visible=False)
    prefix_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None,
                            visible=False)
    emotion1 = gr.Number(visible=False)
    emotion2 = gr.Number(visible=False)
    emotion3 = gr.Number(visible=False)
    emotion4 = gr.Number(visible=False)
    emotion5 = gr.Number(visible=False)
    emotion6 = gr.Number(visible=False)
    emotion7 = gr.Number(visible=False)
    emotion8 = gr.Number(visible=False)
    vq_single = gr.Number(visible=False)
    fmax = gr.Number(visible=False)
    pitch_std = gr.Number(visible=False)
    speaking_rate = gr.Number(visible=False)
    dnsmos = gr.Number(visible=False)
    speaker_noised_checkbox = gr.Checkbox(visible=False)
    cfg_scale = gr.Number(visible=False)
    top_p = gr.Number(visible=False)
    min_k = gr.Number(visible=False)
    min_p = gr.Number(visible=False)
    linear = gr.Number(visible=False)
    confidence = gr.Number(visible=False)
    quadratic = gr.Number(visible=False)
    randomize_seed_toggle = gr.Checkbox(visible=False)
    unconditional_keys = gr.Textbox(visible=False)
    hidden_btn = gr.Button(visible=False)
    hidden_btn.click(
        fn=generate_audio_ui,
        api_name="generate_audio",
        inputs=[
            model_choice, text, language, speaker_audio, prefix_audio,
            emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8,
            vq_single, fmax, pitch_std, speaking_rate, dnsmos,
            speaker_noised_checkbox, cfg_scale, top_p, min_k, min_p,
            linear, confidence, quadratic, seed_num,
            randomize_seed_toggle, unconditional_keys,
        ],
        outputs=[audio_output, seed_num],
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true',
                        help="Create a EXTERNAL facing public link using Gradio's servers")
    parser.add_argument("--server", type=str, default='0.0.0.0', help="Server address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=False, default=7860,
                        help="Port to run the server on (default: 7860)")
    parser.add_argument("--inbrowser", action='store_true', help="Open the UI in a new browser window")
    parser.add_argument("--multilingual", action='store_true', default=False,
                        help="Use the multilingual model (requires more VRAM)")
    parser.add_argument("--clearoutput", action='store_true',
                        help="Remove all folders in audio output directory and exit")
    parser.add_argument("--clearcache", action='store_true', help="Remove all .pt cache files and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Handle cleanup arguments that exit immediately
    if args.clearoutput:
        logger.info("Clearing output directories...")
        count = clear_output_directories()
        logger.info(f"Cleared {count} output directories. Exiting.")
        exit(0)

    if args.clearcache:
        logger.info("Clearing cache files...")
        count = clear_cache_files()
        logger.info(f"Cleared {count} cache files. Exiting.")
        exit(0)

    MULTILINGUAL = args.multilingual

    # Load configuration at startup
    logger.info("Loading SkyrimNet configuration...")
    load_skyrimnet_config()

    model = load_model()
    init_conditional_memory_cache(model, DEVICE, DTYPE, quiet=False, pre_validate_voices=False)  # Quiet for prod
    load_fuzzy_cache()

    demo.queue(
        max_size=12,
        default_concurrency_limit=2,
    ).launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser
    )