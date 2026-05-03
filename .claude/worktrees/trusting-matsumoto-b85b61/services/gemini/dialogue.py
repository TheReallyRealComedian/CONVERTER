# services/gemini/dialogue.py
"""Dialogue text parsing, filtering and splitting helpers.

Pure functions operating on the dialogue list-of-dicts shape produced by the
script-generation step. No SDK calls, no class state. Imported by both
``services.gemini.script`` (uses ``parse_dialogue``) and
``services.gemini.tts`` (uses ``filter_metadata_lines`` /
``split_long_dialogue_turns`` before TTS synthesis).
"""
import logging
import re

logger = logging.getLogger(__name__)


# Patterns that indicate metadata / captions / non-speakable content.
_METADATA_PATTERNS = [
    r'^foto:',
    r'^quelle:',
    r'^source:',
    r'^bild:',
    r'\.de\s*$',
    r'\.com\s*$',
    r'^http',
    r'wahlrecht\.de',
    r'eigene berechnung',
    r'^\s*IMAGO\s*$',
    r'^\s*Getty\s*$',
    r'^\s*dpa\s*$',
]


def parse_dialogue(formatted_text):
    """Parse formatted dialogue text into structured list."""
    dialogue_lines = []
    for line in formatted_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('**'):
            continue

        if ':' in line:
            parts = line.split(':', 1)
            speaker_part = parts[0].strip()
            text_part = parts[1].strip() if len(parts) > 1 else ""

            style = ""
            speaker = speaker_part
            if '[' in speaker_part and ']' in speaker_part:
                speaker = speaker_part.split('[')[0].strip()
                style = speaker_part.split('[')[1].split(']')[0].strip()

            if text_part:
                dialogue_lines.append({
                    'speaker': speaker,
                    'style': style,
                    'text': text_part
                })

    return dialogue_lines


def filter_metadata_lines(dialogue_lines):
    """Remove metadata, captions, and non-speakable content."""
    filtered = []

    for line in dialogue_lines:
        text = line['text'].strip()

        # Skip empty/tiny lines (reduced from 15 to preserve short dialogue)
        if len(text) < 2:
            logger.info(f"Skipping empty/tiny line: {text}")
            continue

        # Check for metadata patterns
        is_metadata = False
        for pattern in _METADATA_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Filtering metadata: {text[:50]}...")
                is_metadata = True
                break

        if not is_metadata:
            filtered.append(line)

    logger.info(f"Filtered {len(dialogue_lines) - len(filtered)} metadata lines")
    return filtered


def split_long_dialogue_turns(dialogue_lines, max_words=50):
    """
    Split overly long dialogue turns into shorter chunks for TTS.
    Gemini TTS works better with shorter, natural turns (20-50 words).
    """

    def split_text_into_chunks(text, max_words):
        """Split text at sentence boundaries, keeping under max_words."""
        # Split into sentences (handles German/English punctuation)
        sentences = re.split(r'([.!?…]+\s+)', text)

        chunks = []
        current_chunk = ""
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            words = len(sentence.split())

            # If adding this would exceed limit, start new chunk
            if current_words > 0 and current_words + words > max_words:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_words = words
            else:
                current_chunk += sentence
                current_words += words

            i += 1

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]  # Fallback to original if splitting fails

    split_dialogue = []

    for turn in dialogue_lines:
        text = turn['text'].strip()
        word_count = len(text.split())

        if word_count <= max_words:
            split_dialogue.append(turn)
        else:
            logger.info(f"Splitting long turn ({word_count} words) for {turn['speaker']}")
            chunks = split_text_into_chunks(text, max_words)

            for i, chunk in enumerate(chunks):
                split_dialogue.append({
                    'speaker': turn['speaker'],
                    'style': turn['style'] if i == 0 else '',  # Style only on first chunk
                    'text': chunk
                })

    logger.info(f"Split {len(dialogue_lines)} turns into {len(split_dialogue)} turns")
    return split_dialogue
