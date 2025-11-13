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
        
        logger.info(f"LLM formatted dialogue length: {len(formatted_dialogue)} characters")
        
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
        return f"""You are a narrator/audiobook reader. Convert the following raw text into a natural, engaging narration.

**Context:**
- Single speaker (monologue)
- Language: {language_name}
- Overall tone: {tone}
- Target length: {target_length}
- Speaker:
{speakers_info}

**Raw Text to Convert:**
{raw_text}

**Instructions:**
1. Create a natural, engaging {target_length_desc} narration from the raw text
2. Format the text naturally for speaking, with appropriate pauses and emphasis
3. Add style hints in square brackets where appropriate (e.g., [thoughtfully], [enthusiastically], [calmly])
4. Maintain the {tone} tone throughout
5. Use the exact speaker name provided above
6. IMPORTANT: Generate enough content to match the target length of {target_length}. Expand on topics naturally with details and examples.
7. Make it engaging and natural to listen to

**Output Format (one line per section):**
SpeakerName [optional-style]: Text to be spoken

**Example:**
Narrator [warmly]: Welcome to today's story about artificial intelligence and how it's transforming our world.
Narrator [thoughtfully]: Let me start by explaining what AI really means...

Now convert the raw text above into this format for a single narrator. Remember to generate {target_length_desc} content!"""
    
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
            
            if style:
                dialogue_lines.append(f"{speaker}: [{style}] {text}")
            else:
                dialogue_lines.append(f"{speaker}: {text}")
        
        full_dialogue = "\n".join(dialogue_lines)
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