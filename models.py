"""
Project Name: AI-TRI-MODEL-CHAT
Author: Niroj Kumar Sahoo (nirojkumarsahoo55@gmail.com)
Copyright (c) 2026 Niroj Kumar Sahoo
License: MIT License 
Source: https://github.com/nirojhub/AI-Tri-Model-Chat
Description: Defines the model configurations and helper functions to build language models 
             for the tri-model chat application. 
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def build_ollama_model(model: str, base_url: str, temperature: float, top_p: float = 0.95, frequency_penalty: float = 0.4) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
    )


def build_lmstudio_model(model: str, base_url: str, temperature: float, top_p: float = 0.95, frequency_penalty: float = 0.4) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="lm-studio",
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
    )


def build_msty_model(model: str, base_url: str, temperature: float, top_p: float = 0.95, frequency_penalty: float = 0.4) -> ChatOpenAI:
    return ChatOllama(
        model=model,
        base_url=base_url,
        api_key="deepseek_r1",
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
    )
