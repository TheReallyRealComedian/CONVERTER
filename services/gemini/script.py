# services/gemini/script.py
"""LLM script generation: raw text → structured podcast dialogue.

Calls ``gemini-2.5-flash`` to transform source material into a speaker-tagged
script and parses the result into the dialogue list-of-dicts shape that the
TTS step consumes. Stateless aside from the supplied ``client``.
"""
import logging
import time

from google.genai import types

from services.gemini.dialogue import parse_dialogue
from services.gemini.prompts import (
    STYLE_DIRECTIVES,
    build_multi_speaker_prompt,
    build_single_speaker_prompt,
    calculate_tag_guidance,
)

logger = logging.getLogger(__name__)


_LANGUAGE_NAMES = {
    'en': 'English',
    'de': 'German',
    'es': 'Spanish',
    'fr': 'French',
}

_LENGTH_INFO = {
    'short': ('300-500 words', 'short (2-3 minute)'),
    'medium': ('800-1200 words', 'medium-length (5-7 minute)'),
    'long': ('1500-2500 words', 'long (10-15 minute)'),
    'very-long': ('3000-5000 words', 'very long (20-30 minute)'),
}


def format_dialogue_with_llm(client, raw_text, num_speakers=2, speaker_descriptions=None,
                              language='en', narration_style='conversational',
                              script_length='medium', custom_prompt=None):
    """
    Format raw text into structured narrative using Gemini LLM with style directives.

    Args:
        client: A configured ``google.genai.Client`` instance.
        raw_text: Raw text to format
        num_speakers: Number of speakers (1-4)
        speaker_descriptions: List of dicts with name, voice, personality
        language: Language code
        narration_style: Delivery style (documentary, conversational, academic, satirical, dramatic)
        script_length: 'short', 'medium', 'long', 'very-long'
        custom_prompt: Optional custom system prompt

    Returns:
        dict: {'dialogue': [...], 'raw_formatted_text': '...'}
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("No text provided")

    if num_speakers < 1 or num_speakers > 4:
        raise ValueError("Number of speakers must be between 1 and 4")

    speaker_descriptions = speaker_descriptions or []

    # Build speakers info string
    speakers_info = ""
    for i, desc in enumerate(speaker_descriptions[:num_speakers], 1):
        name = desc.get('name', f'Speaker{i}')
        voice = desc.get('voice', 'Kore')
        personality = desc.get('personality', 'neutral')
        speakers_info += f"- {name} (Voice: {voice}, Personality: {personality})\n"

    language_name = _LANGUAGE_NAMES.get(language, 'English')
    target_length, target_length_desc = _LENGTH_INFO.get(script_length, _LENGTH_INFO['medium'])

    # Calculate dynamic tag guidance
    tag_info = calculate_tag_guidance(raw_text, narration_style)

    # Get the style directive
    style_directive = STYLE_DIRECTIVES.get(narration_style, STYLE_DIRECTIVES['conversational'])

    # Build prompt based on num_speakers
    if custom_prompt:
        prompt = custom_prompt
    else:
        if num_speakers == 1:
            prompt = build_single_speaker_prompt(
                style_directive, language_name, target_length, target_length_desc,
                speakers_info, raw_text, tag_info, narration_style
            )
        else:
            prompt = build_multi_speaker_prompt(
                style_directive, num_speakers, language_name, target_length,
                target_length_desc, speakers_info, raw_text, tag_info, narration_style
            )

    logger.info(f"=== LLM SCRIPT GENERATION START ===")
    logger.info(f"Model: gemini-2.5-flash")
    logger.info(f"Narration style: {narration_style}")
    logger.info(f"Tag guidance: {tag_info['tag_range']} tags ({tag_info['chars_per_tag']} chars/tag)")
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info(f"Script config: length={script_length}, speakers={num_speakers}")
    logger.info(f"Input text: {len(raw_text)} chars, ~{tag_info['spoken_minutes']} min spoken")

    # Call Gemini with timing
    llm_start_time = time.time()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=16384
            )
        )

        llm_elapsed = time.time() - llm_start_time
        logger.info(f"=== LLM SCRIPT GENERATION SUCCESS ===")
        logger.info(f"LLM response time: {llm_elapsed:.2f}s")

        # Log response metadata if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            logger.info(f"Usage: prompt_tokens={getattr(um, 'prompt_token_count', 'N/A')}, "
                       f"output_tokens={getattr(um, 'candidates_token_count', 'N/A')}, "
                       f"total={getattr(um, 'total_token_count', 'N/A')}")

        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                logger.info(f"Finish reason: {candidate.finish_reason}")

    except Exception as e:
        llm_elapsed = time.time() - llm_start_time
        logger.error(f"=== LLM SCRIPT GENERATION FAILED ===")
        logger.error(f"LLM failed after: {llm_elapsed:.2f}s")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
        raise

    if not response.text:
        logger.error("LLM returned empty response")
        raise ValueError("LLM did not generate any output")

    formatted_dialogue = response.text.strip()
    logger.info(f"LLM output: {len(formatted_dialogue)} characters")

    # Check if LLM refused or gave invalid output
    if len(formatted_dialogue) < 100:
        logger.error(f"LLM gave suspiciously short response: '{formatted_dialogue}'")
        raise ValueError(f"LLM generated invalid output (too short): {formatted_dialogue}")

    # Parse dialogue
    dialogue_lines = parse_dialogue(formatted_dialogue)

    if not dialogue_lines:
        raise ValueError("Could not parse dialogue from LLM output")

    logger.info(f"Parsed {len(dialogue_lines)} dialogue lines")

    return {
        'dialogue': dialogue_lines,
        'raw_formatted_text': formatted_dialogue
    }
