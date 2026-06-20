"""
Project Name: AI-TRI-MODEL-CHAT
Author: Niroj Kumar Sahoo (nirojkumarsahoo55@gmail.com)
Copyright (c) 2026 Niroj Kumar Sahoo
License: MIT License 
Source: https://github.com/nirojhub/AI-Tri-Model-Chat
Description: Defines the conversation state and logic for managing the tri-model chat discussion, 
including message construction, model invocation, and optional text-to-speech synthesis. 
"""

from dataclasses import dataclass, field
import io
import re
import wave

import numpy as np
from kokoro import KPipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import build_lmstudio_model, build_ollama_model, build_msty_model

BREVITY_INSTRUCTION = (
    "Keep every reply to at most 2 sentences. "
    "Be concise and do not use bullet points or numbered lists."
)

TTS_SAMPLE_RATE = 24000  # Kokoro outputs at 24kHz
VOICE_PRESETS = ["af_sky", "am_adam", "bm_george"]

print("📥 Loading Kokoro TTS (82M)...")
tts_pipeline = KPipeline(lang_code='a')  # 'a' for American English

@dataclass
class ModelConfig:
    name: str
    model: str
    base_url: str
    system_prompt: str
    temperature: float


@dataclass
class ConversationState:
    model_a: ModelConfig
    model_b: ModelConfig
    model_c: ModelConfig
    seed_message: str
    max_turns: int
    turn_index: int = 0
    messages: list[dict] = field(default_factory=list)
    running: bool = False
    last_error: str | None = None
    voice_enabled: bool = False


def _get_input_text(state: ConversationState) -> str:
    if state.turn_index == 0:
        return state.seed_message
    
    # Get the last message
    last_message = state.messages[-1]
    
    # If we have at least 2 messages, also get the previous one for context
    if len(state.messages) >= 2:
        previous_message = state.messages[-2]
        combined_context = (
            f"Previous response from {previous_message['speaker']}:\n"
            f"{previous_message['content']}\n\n"
            f"Latest response from {last_message['speaker']}:\n"
            f"{last_message['content']}\n\n"
            f"Please consider both responses above and provide your reply. "
            f"Keep your conversation around \"{state.seed_message}\" and avoid diverging into unrelated topics."
        )
        return combined_context
    
    return last_message["content"]


def _get_active_config(state: ConversationState) -> ModelConfig:
    turn = state.turn_index % 3
    if turn == 0:
        return state.model_a
    if turn == 1:
        return state.model_b
    return state.model_c


def _build_llm(state: ConversationState,config: ModelConfig):
    if state.turn_index % 3 == 0:
        return build_ollama_model(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )
    if state.turn_index % 3 == 1:
        return build_lmstudio_model(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )
    if state.turn_index % 3 == 2:
        return build_msty_model(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
        )


def _build_messages(config: ModelConfig, input_text: str) -> list:
    system_parts = []
    if config.system_prompt.strip():
        system_parts.append(config.system_prompt.strip())
    system_parts.append(BREVITY_INSTRUCTION)
    messages = [SystemMessage(content="\n\n".join(system_parts))]
    messages.append(HumanMessage(content=input_text))
    return messages


def _get_voice_preset(turn: int) -> str:
    return VOICE_PRESETS[turn % len(VOICE_PRESETS)]


def synthesize_speech(text: str, turn: int) -> bytes | None:
    if not text or not text.strip():
        return None

    voice_preset = _get_voice_preset(turn)
    audio_chunks = []
    for _, _, audio in tts_pipeline(text, voice=voice_preset, speed=1.0, split_pattern=r'\n+'):
        audio_chunks.append(np.asarray(audio))

    if not audio_chunks:
        return None

    combined_audio = np.concatenate(audio_chunks)
    if combined_audio.dtype.kind == "f":
        combined_audio = np.clip(combined_audio, -1.0, 1.0)
        combined_audio = (combined_audio * 32767).astype(np.int16)
    elif combined_audio.dtype != np.int16:
        combined_audio = combined_audio.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TTS_SAMPLE_RATE)
        wav_file.writeframes(combined_audio.tobytes())

    return buffer.getvalue()


def run_single_turn(state: ConversationState) -> ConversationState:
    if state.turn_index >= state.max_turns:
        state.running = False
        return state

    config = _get_active_config(state)
    input_text = _get_input_text(state)

    try:
        llm = _build_llm(state, config)
        input_messages = _build_messages(config, input_text)
        response = llm.invoke(input_messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)

        # This removes the entire think block and any remaining tags.        
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
        content = content.strip()

        audio_bytes = None
        if state.voice_enabled:
            try:
                audio_bytes = synthesize_speech(content, state.turn_index)
            except Exception as exc:
                print(f"TTS generation failed for turn {state.turn_index}: {exc}")
                audio_bytes = None
                state.last_audio_end_time = 0
        else:
            audio_bytes = None
            state.last_audio_end_time = 0

        state.messages.append(
            {
                "speaker": config.name,
                "content": content,
                "turn": state.turn_index,
                "audio": audio_bytes,
            }
        )
        state.turn_index += 1
        state.last_error = None

        if state.turn_index > state.max_turns:
            state.running = False

    except Exception as exc:
        state.last_error = f"{config.name} failed: {exc}"
        state.running = False

    return state


def get_active_speaker_name(state: ConversationState) -> str:
    return _get_active_config(state).name
