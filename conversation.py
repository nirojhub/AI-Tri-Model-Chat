from dataclasses import dataclass, field
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import build_lmstudio_model, build_ollama_model, build_msty_model

BREVITY_INSTRUCTION = (
    "Keep every reply to at most 2 sentences. "
    "Be concise and do not use bullet points or numbered lists."
)


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


def _get_input_text(state: ConversationState) -> str:
    if state.turn_index == 0:
        return state.seed_message
    return state.messages[-1]["content"]


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


def run_single_turn(state: ConversationState) -> ConversationState:
    if state.turn_index >= state.max_turns:
        state.running = False
        return state

    config = _get_active_config(state)
    input_text = _get_input_text(state)

    try:
        llm = _build_llm(state, config)
        response = llm.invoke(_build_messages(config, input_text))
        content = response.content if isinstance(response, AIMessage) else str(response.content)

        # If this is the third model in the conversation (model_c), remove any
        # internal "<think>...</think>" sections before showing the reply.
        # This removes the entire think block and any remaining tags.
        if config is state.model_c:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
            content = content.strip()

        state.messages.append(
            {
                "speaker": config.name,
                "content": content,
                "turn": state.turn_index,
            }
        )
        state.turn_index += 1
        state.last_error = None

        if state.turn_index >= state.max_turns:
            state.running = False

    except Exception as exc:
        state.last_error = f"{config.name} failed: {exc}"
        state.running = False

    return state


def get_active_speaker_name(state: ConversationState) -> str:
    return _get_active_config(state).name
