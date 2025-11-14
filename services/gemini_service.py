# services/gemini_service.py
import logging
import tempfile
import os
import wave
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
    
    def format_dialogue_with_llm(self, raw_text, num_speakers=2, speaker_descriptions=None,
                                  language='en', tone='professional and informative',
                                  script_length='medium', custom_prompt=None):
        """
        Format raw text into structured dialogue using Gemini LLM.
        
        Args:
            raw_text: Raw text to format
            num_speakers: Number of speakers (1-4)
            speaker_descriptions: List of dicts with name, voice, personality
            language: Language code
            tone: Overall tone/style
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
        
        # Language mapping
        language_name = {
            'en': 'English',
            'de': 'German',
            'es': 'Spanish',
            'fr': 'French'
        }.get(language, 'English')
        
        # Length info
        length_info = {
            'short': ('300-500 words', 'short (2-3 minute)'),
            'medium': ('800-1200 words', 'medium-length (5-7 minute)'),
            'long': ('1500-2500 words', 'long (10-15 minute)'),
            'very-long': ('3000-5000 words', 'very long (20-30 minute)')
        }
        target_length, target_length_desc = length_info.get(script_length, length_info['medium'])
        
        # Build prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            if num_speakers == 1:
                prompt = self._build_single_speaker_prompt(
                    language_name, tone, target_length, target_length_desc, speakers_info, raw_text
                )
            else:
                prompt = self._build_multi_speaker_prompt(
                    num_speakers, language_name, tone, target_length, target_length_desc, speakers_info, raw_text
                )
        
        # Replace placeholders
        prompt = prompt.replace('{num_speakers}', str(num_speakers))
        prompt = prompt.replace('{language_name}', language_name)
        prompt = prompt.replace('{tone}', tone)
        prompt = prompt.replace('{target_length}', target_length)
        prompt = prompt.replace('{target_length_desc}', target_length_desc)
        prompt = prompt.replace('{speakers_info}', speakers_info)
        prompt = prompt.replace('{raw_text}', raw_text)
        
        logger.info(f"Generating dialogue with script_length={script_length}, num_speakers={num_speakers}")
        
        # Call Gemini
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192
            )
        )
        
        if not response.text:
            raise ValueError("LLM did not generate any output")
        
        formatted_dialogue = response.text.strip()
        # DEBUG: Log what the LLM actually returned
        logger.info(f"LLM raw response:\n{formatted_dialogue}")
        logger.info(f"LLM response length: {len(formatted_dialogue)} characters")

        # Check if LLM refused or gave invalid output
        if len(formatted_dialogue) < 100:
            logger.error(f"LLM gave suspiciously short response: '{formatted_dialogue}'")
            raise ValueError(f"LLM generated invalid output (too short): {formatted_dialogue}")
                
        # Parse dialogue
        dialogue_lines = self._parse_dialogue(formatted_dialogue)
        
        if not dialogue_lines:
            raise ValueError("Could not parse dialogue from LLM output")
        
        logger.info(f"Parsed {len(dialogue_lines)} dialogue lines")
        
        return {
            'dialogue': dialogue_lines,
            'raw_formatted_text': formatted_dialogue
        }
    
    def _build_single_speaker_prompt(self, language_name, tone, target_length, target_length_desc, speakers_info, raw_text):
        speaker_name = speakers_info.split('\n')[0].split('(')[0].replace('-', '').strip() if speakers_info else 'Narrator'
        
        return f"""Convert this text into a natural audio narration in {language_name}.

        **Speaker:** {speaker_name}
        **Tone:** {tone}
        **Target:** {target_length} ({target_length_desc})

        **Source Text:**
        {raw_text}

        **CRITICAL FORMAT - Each line MUST follow this EXACT pattern:**
        {speaker_name} [style]: Short natural sentence.
        {speaker_name} [style]: Another short sentence.

        **Rules:**
        1. Each line is ONE short thought (15-40 words maximum)
        2. ALWAYS start with "{speaker_name} [style]:" 
        3. Filter out captions, sources, metadata
        4. Use styles like [warmly], [thoughtfully], [with emphasis]
        5. Natural, flowing speech - NO fragments or quotes
        6. Expand to {target_length} with context and examples

        **Example:**
        {speaker_name} [warmly]: Today I want to tell you about a fascinating political development.
        {speaker_name} [thoughtfully]: To understand why this matters, we need some background.
        {speaker_name} [with emphasis]: This represents a significant shift in the political landscape.

        Now convert the source text. Remember: SHORT lines, EXACT format!"""
    

    def _build_multi_speaker_prompt(self, num_speakers, language_name, tone, target_length, target_length_desc, speakers_info, raw_text):
        return f"""You are a podcast script formatter. Convert the following raw text into a structured dialogue script.

**Context:**
- Number of speakers: {num_speakers}
- Language: {language_name}
- Overall tone: {tone}
- Target length: {target_length}
- Speakers:
{speakers_info}

**Raw Text to Convert:**
{raw_text}

**Instructions:**
1. Create a natural, engaging {target_length_desc} podcast script from the raw text
2. Split the content naturally into dialogue turns between the {num_speakers} speaker(s)
3. Each turn should be 1-4 sentences for natural conversational flow
4. Add appropriate style hints in square brackets (e.g., enthusiastically, calmly, thoughtfully)
5. Make it conversational, engaging, and maintain the {tone} tone
6. Use the exact speaker names provided above
7. IMPORTANT: Generate enough content to match the target length of {target_length}. Don't be brief - expand on topics naturally with details, examples, and explanations.
8. Include transitions, questions, clarifications, and natural back-and-forth between speakers
9. Add depth by exploring different angles, asking follow-up questions, and providing context

**Output Format (one line per dialogue turn):**
SpeakerName [style]: Text of what they say

**Example:**
Anna [enthusiastically]: Welcome to our show!
Max [professionally]: Today we discuss artificial intelligence and how it's transforming every aspect of our daily lives.
Anna [curiously]: That's fascinating! Can you break down what exactly makes AI so revolutionary for our listeners who might not be familiar with the technical details?
Max [thoughtfully]: Great question. Let me start with the basics and then we'll dive deeper into some specific examples...

Now convert the raw text above into this format. Remember to generate {target_length_desc} content - don't rush, take time to explore the topic thoroughly!"""
    
    def _parse_dialogue(self, formatted_text):
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
    
    def generate_podcast(self, dialogue, language='en'):
        """
        Generate multi-speaker podcast audio using Gemini TTS.
        
        Args:
            dialogue: List of dicts with 'speaker', 'style', 'text'
            language: Language code
        
        Returns:
            str: Path to temporary WAV file
        """
        if not dialogue or len(dialogue) == 0:
            raise ValueError("No dialogue provided")
        
        # Build speaker voice configs
        speaker_voice_configs = []
        seen_speakers = set()
        dialogue_lines = []
                
        for turn in dialogue:
            speaker = turn.get('speaker', 'Kore')
            text = turn.get('text', '').strip()
            style = turn.get('style', '').strip()
            
            if not text:
                continue
            
            if speaker not in seen_speakers:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker
                            )
                        )
                    )
                )
                seen_speakers.add(speaker)
            
            dialogue_lines.append({
                'speaker': speaker,
                'style': style,
                'text': text
            })

        # CRITICAL: Filter and split before TTS
        logger.info(f"Raw dialogue: {len(dialogue_lines)} turns")

        # Remove metadata/captions
        dialogue_lines = self._filter_metadata_lines(dialogue_lines)

        # Split long turns (50 words max per turn)
        dialogue_lines = self._split_long_dialogue_turns(dialogue_lines, max_words=50)

        if not dialogue_lines:
            raise ValueError("No valid dialogue lines after filtering")

        logger.info(f"Final dialogue: {len(dialogue_lines)} turns")

        # Format for TTS
        formatted_lines = []
        for turn in dialogue_lines:
            if turn['style']:
                formatted_lines.append(f"{turn['speaker']}: [{turn['style']}] {turn['text']}")
            else:
                formatted_lines.append(f"{turn['speaker']}: {turn['text']}")

        full_dialogue = "\n".join(formatted_lines)
        unique_speakers = list(seen_speakers)
        
        if len(unique_speakers) < 1:
            raise ValueError("No speakers found. Please configure at least 1 speaker.")
        elif len(unique_speakers) > 4:
            raise ValueError("Gemini TTS supports maximum 4 speakers. Please reduce to 1-4 speakers.")
        
        logger.info(f"Gemini-TTS dialogue:\n{full_dialogue}")
        logger.info(f"Speakers: {unique_speakers}")
        
        # Generate audio
        if len(unique_speakers) == 1:
            logger.info("Using single-speaker mode")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=full_dialogue,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker_voice_configs[0].speaker
                            )
                        )
                    )
                )
            )
        else:
            logger.info("Using multi-speaker mode")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=full_dialogue,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=speaker_voice_configs
                        )
                    )
                )
            )
        
        logger.info("Gemini-TTS response received")
        
        if not response or not response.candidates:
            raise ValueError("Invalid response from Gemini-TTS")
        
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
        
        if not audio_data:
            raise ValueError("No audio data in response")
        
        logger.info(f"Found audio: {len(audio_data)} bytes, type: {mime_type}")
        
        # Convert to WAV
        temp_audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        with wave.open(temp_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        
        logger.info(f"Audio converted and saved to: {temp_audio_path}")
        
        return temp_audio_path
    
    def _split_long_dialogue_turns(self, dialogue_lines, max_words=50):
        """
        Split overly long dialogue turns into shorter chunks for TTS.
        Gemini TTS works better with shorter, natural turns (20-50 words).
        """
        import re
        
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
    
    def _filter_metadata_lines(self, dialogue_lines):
        """Remove metadata, captions, and non-speakable content."""
        import re
        
        # Patterns that indicate metadata/captions
        metadata_patterns = [
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
        
        filtered = []
        
        for line in dialogue_lines:
            text = line['text'].strip()
            
            # Skip very short lines
            if len(text) < 15:
                logger.info(f"Skipping short line: {text}")
                continue
            
            # Check for metadata patterns
            is_metadata = False
            for pattern in metadata_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.info(f"Filtering metadata: {text[:50]}...")
                    is_metadata = True
                    break
            
            if not is_metadata:
                filtered.append(line)
        
        logger.info(f"Filtered {len(dialogue_lines) - len(filtered)} metadata lines")
        return filtered