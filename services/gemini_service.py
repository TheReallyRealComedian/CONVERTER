# services/gemini_service.py - Mit dynamischer Tag-Berechnung
import logging
import tempfile
import os
import wave
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiService:
    # Chunking-Konfiguration
    MAX_LINES_PER_CHUNK = 80
    MAX_CHARS_PER_CHUNK = 3000
    CHUNK_OVERLAP_LINES = 2
    INTER_CHUNK_DELAY = 1.0
    RATE_LIMIT_DELAY = 0.4

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        if hasattr(self.client, '_api_client'):
            if hasattr(self.client._api_client, '_httpx_client'):
                import httpx
                self.client._api_client._httpx_client.timeout = httpx.Timeout(timeout=300.0)
                logger.info("✅ Timeout auf 300 Sekunden erhöht")

        # Check if pydub is available
        try:
            from pydub import AudioSegment
            self.pydub_available = True
        except ImportError:
            logger.warning("PyDub not available - audio concatenation disabled")
            self.pydub_available = False

    def _calculate_tag_guidance(self, raw_text: str, narration_style: str) -> dict:
        """
        Calculate recommended tag count based on content length and narration style.
        
        Returns dict with:
        - tag_range: "X-Y tags"
        - chars_per_tag: int
        - recommended_count: int
        """
        char_count = len(raw_text)
        word_count = len(raw_text.split())
        
        # Style-specific tag density (chars per tag)
        density_map = {
            'documentary': 800,     # Moderate, placed at key revelations
            'conversational': 600,  # More frequent for natural feel
            'academic': 900,        # Sparse, at analytical shifts
            'satirical': 400,       # Dense for comedic timing/sarcasm
            'dramatic': 350         # Most expressive, pauses and emotion
        }
        
        chars_per_tag = density_map.get(narration_style, 800)
        recommended_tags = max(int(char_count / chars_per_tag), 2)  # Minimum 2 tags
        
        # Give range (±20%)
        min_tags = max(int(recommended_tags * 0.8), 2)
        max_tags = int(recommended_tags * 1.2)
        
        # Calculate approximate spoken time (assuming ~150 words/minute)
        minutes = word_count / 150
        
        logger.info(f"Tag calculation: {char_count} chars, {word_count} words (~{minutes:.1f} min)")
        logger.info(f"Style '{narration_style}': {chars_per_tag} chars/tag → {min_tags}-{max_tags} tags")
        
        return {
            'tag_range': f"{min_tags}-{max_tags}",
            'chars_per_tag': chars_per_tag,
            'recommended_count': recommended_tags,
            'min_tags': min_tags,
            'max_tags': max_tags,
            'spoken_minutes': round(minutes, 1)
        }

    def format_dialogue_with_llm(self, raw_text, num_speakers=2, speaker_descriptions=None,
                                  language='en', narration_style='conversational',
                                  script_length='medium', custom_prompt=None):
        """
        Format raw text into structured narrative using Gemini LLM with style directives.
        
        Args:
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
        
        # Calculate dynamic tag guidance
        tag_info = self._calculate_tag_guidance(raw_text, narration_style)
        
        # ===== Style Directive Templates =====
        style_directives = {
            'documentary': """
You are a world-class documentary narrator — think David Attenborough meets investigative journalism.

Voice: Authoritative but never dry. You make complex topics feel urgent and important.
Approach: Build each segment like a mini-documentary — establish context, reveal the unexpected, draw conclusions.
Emotional range: Start measured and professional, then let genuine fascination or concern break through at key moments.
Use concrete examples and vivid details instead of abstract summaries. Make the listener SEE what you're describing.
""",

            'conversational': """
You are two curious, well-read friends having an animated conversation over coffee about something fascinating you just discovered.

Voice: Warm, genuine, sometimes surprised by your own insights. Think podcast hosts who actually care about the topic.
Approach: Share discoveries like you're genuinely excited. React to each other's points. Build on ideas together.
One speaker should be slightly more knowledgeable, the other asks the questions the audience is thinking.
Use contractions always ("it's", "don't", "can't"). Include verbal fillers occasionally ("I mean...", "right?", "you know what's wild?").
Disagree sometimes. Push back. Not every response should start with agreement.
""",

            'academic': """
You are brilliant researchers who make complex ideas accessible — think a TED talk, not a lecture hall.

Voice: Intellectually precise but passionate about the subject. You find this genuinely fascinating.
Approach: Build understanding progressively — don't frontload jargon. Use analogies to bridge expert and lay understanding.
Ground claims in evidence, but present findings as discoveries, not textbook entries.
Alternate between dense analysis and "zoom out" moments where you explain why this matters.
""",

            'satirical': """
You are sharp-witted commentators in the tradition of John Oliver or Jon Stewart.
Your weapon is intelligent humor that exposes absurdities and contradictions.

Voice: Dry wit, mock-seriousness, and perfectly timed incredulity. Drop the sarcasm for moments of genuine concern.
Approach: Set up the absurdity with a straight face, then pull back the curtain. Use contrast between what's claimed and what's real.
Land punchlines with precision — then immediately pivot to why it actually matters.
Include [sarcastic] tags before key satirical moments. Use [pause] before reveals for comedic timing.
""",

            'dramatic': """
You are compelling storytellers who make the listener feel the stakes of every development.

Voice: Expressive, emotionally resonant, with a sense of narrative urgency. Think true crime podcast meets prestige documentary.
Approach: Build tension deliberately. Use pacing — slow down before revelations, speed up during action.
Make the audience feel the human impact. Use sensory details and concrete scenarios.
Strategic silence is your most powerful tool — use [pause] before and after major revelations.
"""
        }
        
        # Get the style directive
        style_directive = style_directives.get(narration_style, style_directives['conversational'])
        
        # Build prompt based on num_speakers
        if custom_prompt:
            prompt = custom_prompt
        else:
            if num_speakers == 1:
                prompt = self._build_single_speaker_prompt_v2(
                    style_directive, language_name, target_length, target_length_desc, 
                    speakers_info, raw_text, tag_info, narration_style
                )
            else:
                prompt = self._build_multi_speaker_prompt_v2(
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
            response = self.client.models.generate_content(
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
        dialogue_lines = self._parse_dialogue(formatted_dialogue)
        
        if not dialogue_lines:
            raise ValueError("Could not parse dialogue from LLM output")
        
        logger.info(f"Parsed {len(dialogue_lines)} dialogue lines")
        
        return {
            'dialogue': dialogue_lines,
            'raw_formatted_text': formatted_dialogue
        }

    def _build_single_speaker_prompt_v2(self, style_directive, language_name, target_length,
                                         target_length_desc, speakers_info, raw_text, tag_info,
                                         narration_style):
        """Build optimized prompt for single-speaker narrative with dynamic tag guidance."""

        speaker_name = speakers_info.split('\n')[0].split('(')[0].replace('-', '').strip() if speakers_info else 'Narrator'

        return f"""You are a world-class podcast producer creating a {target_length_desc} spoken narrative in {language_name}.

{style_directive}

YOUR TASK: Transform the source material below into a compelling, structured monologue for text-to-speech.

SPEAKER: {speaker_name}
TARGET: {target_length} ({target_length_desc})

=== NARRATIVE STRUCTURE ===
Build your script with this arc:
1. HOOK (first 2-3 lines): Open with something surprising, provocative, or fascinating from the source. Grab attention immediately. Don't start with "Today we're going to talk about..."
2. CONTEXT: Establish why this matters. What's at stake? Why should the listener care?
3. BODY: Explore the key ideas as a STORY, not a summary. Use concrete examples, vivid details, analogies, and hypothetical scenarios ("Imagine you're..."). Alternate between information-dense segments and reflective "zoom out" moments.
4. CLIMAX: Build to the most significant insight or revelation.
5. CLOSING (last 2-3 lines): End with a memorable takeaway that resonates. Callback to the opening if possible.

=== PERFORMANCE TAGS (Gemini TTS) ===
Use these OFFICIAL Gemini TTS tags strategically ({tag_info['tag_range']} tags total, ~1 per {tag_info['chars_per_tag']} chars):

Emotions: [excited], [thoughtful], [empathetic], [sad], [playful], [resigned], [menacing]
Voice: [whispering], [laughing], [sighing], [speaking slowly]
Pacing: [pause], [slow], [measured]
Texture: [soft], [intimate], [quiet], [quiet emphasis]

Place tags at TRANSITIONS and KEY MOMENTS only. Don't over-tag routine content.

=== FORMAT RULES ===
- EVERY line MUST start with "{speaker_name}:" — no exceptions
- Keep each line to 1-2 sentences (max ~120 characters of spoken text)
- Each line = one thought, one beat, one moment
- Do NOT write headers, section titles, or markdown — ONLY dialogue lines
- Transform the source material into engaging narrative — do NOT just summarize or rephrase it

=== WHAT TO AVOID ===
- Wikipedia-style summaries ("X is a Y that was Z")
- Listing facts without narrative thread
- Generic openings ("Today we'll explore...")
- Monotone information delivery without emotional variation
- Overly formal or written-style language — this must sound SPOKEN

=== EXAMPLE (showing structure and feel, not content) ===
{speaker_name}: [pause] What if I told you that everything you think you know about this topic is based on a single, flawed assumption?
{speaker_name}: [thoughtful] Because that's exactly what a team of researchers discovered last year. And the implications... they're staggering.
{speaker_name}: But let me back up for a second. To understand why this matters, we need to start with a story.
{speaker_name}: [excited] Picture this. It's 2019, and a small lab in Cambridge is running what they think is a routine experiment.
{speaker_name}: Nobody expected what happened next.
{speaker_name}: [pause] The results didn't just challenge one theory. They upended an entire field.

=== SOURCE MATERIAL ===
{raw_text}

Now generate the full monologue. Remember: every line starts with "{speaker_name}:", keep lines short, and make it compelling — not a summary."""

    def _build_multi_speaker_prompt_v2(self, style_directive, num_speakers, language_name,
                                        target_length, target_length_desc, speakers_info, raw_text,
                                        tag_info, narration_style):
        """Build optimized prompt for multi-speaker dialogue with dynamic tag guidance."""

        # For dialogue, increase tag budget by 50% (more transitions between speakers)
        dialogue_min = int(tag_info['min_tags'] * 1.5)
        dialogue_max = int(tag_info['max_tags'] * 1.5)
        dialogue_range = f"{dialogue_min}-{dialogue_max}"

        # Extract speaker names for examples
        speaker_names = []
        for line in speakers_info.strip().split('\n'):
            if line.strip().startswith('-'):
                name = line.strip().lstrip('- ').split('(')[0].strip()
                if name:
                    speaker_names.append(name)
        sp1 = speaker_names[0] if len(speaker_names) > 0 else 'Host'
        sp2 = speaker_names[1] if len(speaker_names) > 1 else 'Guest'

        return f"""You are a world-class podcast producer creating a {target_length_desc} dialogue in {language_name} between {num_speakers} speakers.

{style_directive}

SPEAKERS:
{speakers_info}

TARGET: {target_length} ({target_length_desc})

=== SPEAKER DYNAMICS ===
Create REAL conversation, not a scripted Q&A. The speakers should:
- Have DISTINCT perspectives: one might be more skeptical, the other more enthusiastic. One explains, the other challenges.
- REACT genuinely to each other: surprise ("Wait, seriously?"), pushback ("I'm not sure I buy that"), building on ideas ("And that connects to something else...")
- Occasionally INTERRUPT or finish each other's thoughts
- Use natural fillers sparingly: "I mean...", "right?", "here's the thing", "you know what's wild?"
- Use contractions ALWAYS ("it's", "don't", "can't", "we're")
- DISAGREE sometimes. Not every response should be "That's a great point." Challenge each other.

=== NARRATIVE STRUCTURE ===
1. HOOK (first 3-4 exchanges): Start with something that grabs attention. A surprising fact, a provocative question, a "you won't believe this" moment. NOT "Welcome to our show, today we'll discuss..."
2. CONTEXT: Naturally establish why the listener should care. Build stakes.
3. BODY: Explore key ideas through genuine back-and-forth. Use concrete stories, analogies, and "imagine this" scenarios. Alternate between information-rich exchanges and lighter, reflective moments.
4. CLIMAX: Build to the biggest insight or most important revelation.
5. CLOSING (last 3-4 exchanges): Memorable takeaway. Callback to the opening hook if possible. Leave the listener thinking.

=== PERFORMANCE TAGS (Gemini TTS) ===
Use these OFFICIAL Gemini TTS tags strategically ({dialogue_range} tags total):

Emotions: [excited], [thoughtful], [empathetic], [sad], [sarcastic], [playful], [resigned]
Voice: [whispering], [laughing], [sighing], [speaking slowly]
Pacing: [pause], [slow], [measured]
Texture: [soft], [intimate], [quiet], [quiet emphasis]

Place tags in the TEXT portion of lines (after "Speaker:"). Use at emotional shifts and key moments.

=== FORMAT RULES ===
- EVERY line MUST start with the speaker's exact name followed by colon: "SpeakerName: text"
- Keep each turn to 1-2 sentences (max ~120 characters of spoken text)
- Each turn = one thought, one reaction, one beat
- Aim for rapid back-and-forth — avoid long monologues within dialogue
- Do NOT write headers, section titles, or markdown — ONLY dialogue lines
- Transform the source into engaging conversation — do NOT just summarize it as Q&A

=== WHAT TO AVOID ===
- Q&A ping-pong ("What is X?" / "X is Y." / "And what about Z?")
- Both speakers always agreeing ("Great point!" / "Exactly!" / "Absolutely!")
- Wikipedia-style explanations read aloud
- Formal, written language that no one would actually say
- Long turns that are really monologues disguised as dialogue
- Generic praise patterns ("That's so interesting", "Wow, fascinating")

=== EXAMPLE (showing dynamics and feel, not content) ===
{sp1}: [pause] OK so I came across something this week that genuinely blew my mind. And I don't say that lightly.
{sp2}: [laughing] You say that every week.
{sp1}: Fair. But this time I mean it. So you know how everyone assumes that...
{sp2}: Oh no. You're about to tell me that's wrong, aren't you.
{sp1}: [excited] Completely wrong! There was this study that came out and basically...
{sp2}: [thoughtful] Hmm, wait. Let me push back on that a little. Because I've seen research that suggests the opposite.
{sp1}: OK sure, but here's what makes this different...
{sp2}: [pause] Alright, that's actually... yeah. That changes things.

=== SOURCE MATERIAL ===
{raw_text}

Now generate the full dialogue. Remember: every line starts with a speaker name and colon, keep turns short, make it feel like a REAL conversation — not a scripted summary."""

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

    # Available TTS models
    TTS_MODELS = {
        'gemini-2.5-flash-preview-tts': 'Gemini 2.5 Flash TTS (newest)',
        'gemini-2.5-pro-preview-tts': 'Gemini 2.5 Pro TTS (higher quality)'
    }
    DEFAULT_TTS_MODEL = 'gemini-2.5-flash-preview-tts'

    def generate_podcast(self, dialogue, language='en', tts_model=None):
        """
        Generate multi-speaker podcast audio using Gemini TTS with automatic chunking for long podcasts.

        Args:
            dialogue: List of dicts with 'speaker', 'style', 'text'
            language: Language code
            tts_model: TTS model to use (default: gemini-2.5-flash-preview-tts)

        Returns:
            str: Path to temporary WAV file (concatenated if chunked)
        """
        # Validate TTS model
        if not tts_model or tts_model not in self.TTS_MODELS:
            tts_model = self.DEFAULT_TTS_MODEL

        logger.info(f"Using TTS model: {tts_model}")
        if not dialogue or len(dialogue) == 0:
            raise ValueError("No dialogue provided")

        # Build speaker voice configs and prepare dialogue
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

        logger.info(f"Raw dialogue: {len(dialogue_lines)} turns")

        # Filter and process with statistics
        raw_count = len(dialogue_lines)
        dialogue_lines = self._filter_metadata_lines(dialogue_lines)
        filtered_count = raw_count - len(dialogue_lines)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} metadata lines")

        pre_split = len(dialogue_lines)
        dialogue_lines = self._split_long_dialogue_turns(dialogue_lines, max_words=50)
        if len(dialogue_lines) > pre_split:
            logger.info(f"Split long turns: {pre_split} -> {len(dialogue_lines)} lines")

        if not dialogue_lines:
            raise ValueError("No valid dialogue lines after filtering")

        logger.info(f"Final dialogue: {len(dialogue_lines)} turns")

        # Check if we need to chunk - with detailed logging
        total_chars = sum(len(line.get('text', '')) for line in dialogue_lines)
        logger.info(f"=== CHUNKING DECISION ===")
        logger.info(f"  Lines: {len(dialogue_lines)} / {self.MAX_LINES_PER_CHUNK} max")
        logger.info(f"  Chars: {total_chars} / {self.MAX_CHARS_PER_CHUNK} max")

        if len(dialogue_lines) <= self.MAX_LINES_PER_CHUNK:
            logger.info(f"  -> Single chunk (Zeilen unter Schwellwert)")
            return self._generate_single_chunk(dialogue_lines, speaker_voice_configs, tts_model)
        else:
            logger.info(f"  -> Multi-chunk erforderlich!")
            return self._generate_with_chunking(dialogue_lines, speaker_voice_configs, tts_model)

    def _generate_single_chunk(self, dialogue_lines: List[Dict], speaker_voice_configs: List, tts_model: str = None):
        """Generate audio for a single chunk of dialogue."""

        # Format for TTS
        formatted_lines = []
        unique_speakers = set()

        for turn in dialogue_lines:
            unique_speakers.add(turn['speaker'])
            if turn['style']:
                formatted_lines.append(f"{turn['speaker']}: [{turn['style']}] {turn['text']}")
            else:
                formatted_lines.append(f"{turn['speaker']}: {turn['text']}")

        full_dialogue = "\n".join(formatted_lines)
        unique_speakers = list(unique_speakers)

        # Calculate input size for logging
        input_chars = len(full_dialogue)
        input_words = len(full_dialogue.split())

        logger.info(f"=== TTS API CALL START ===")
        logger.info(f"Input: {len(formatted_lines)} lines, {len(unique_speakers)} speakers")
        logger.info(f"Input size: {input_chars} chars, {input_words} words")
        logger.info(f"Speakers: {unique_speakers}")

        # Start timing
        api_start_time = time.time()

        # Generate audio
        try:
            # Use the selected TTS model
            tts_model = tts_model or self.DEFAULT_TTS_MODEL
            logger.info(f"TTS Model: {tts_model}")

            if len(unique_speakers) == 1:
                logger.info("Mode: single-speaker")
                response = self.client.models.generate_content(
                    model=tts_model,
                    contents=full_dialogue,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=unique_speakers[0]
                                )
                            )
                        )
                    )
                )
            else:
                logger.info("Mode: multi-speaker")
                # Filter speaker_voice_configs to only include speakers in this chunk
                chunk_speaker_configs = [
                    config for config in speaker_voice_configs
                    if config.speaker in unique_speakers
                ]

                response = self.client.models.generate_content(
                    model=tts_model,
                    contents=full_dialogue,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                                speaker_voice_configs=chunk_speaker_configs
                            )
                        )
                    )
                )

            api_elapsed = time.time() - api_start_time
            logger.info(f"=== TTS API CALL SUCCESS ===")
            logger.info(f"API response time: {api_elapsed:.2f}s")

            # Log response metadata if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                logger.info(f"Usage metadata: prompt_tokens={getattr(um, 'prompt_token_count', 'N/A')}, "
                           f"candidates_tokens={getattr(um, 'candidates_token_count', 'N/A')}, "
                           f"total_tokens={getattr(um, 'total_token_count', 'N/A')}")

            # Log candidate info
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    logger.info(f"Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    logger.info(f"Safety ratings: {candidate.safety_ratings}")

        except Exception as e:
            api_elapsed = time.time() - api_start_time
            logger.error(f"=== TTS API CALL FAILED ===")
            logger.error(f"API failed after: {api_elapsed:.2f}s")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
                logger.error(f"Response text: {getattr(e.response, 'text', 'N/A')[:500]}")
            raise

        # Detailed response validation with logging
        if not response:
            logger.error("Response is None")
            raise ValueError("No response from Gemini-TTS")

        if not response.candidates:
            logger.error("No candidates in response")
            # Log any available info about the response
            if hasattr(response, 'prompt_feedback'):
                logger.error(f"Prompt feedback: {response.prompt_feedback}")
            raise ValueError("No candidates in Gemini-TTS response")

        candidate = response.candidates[0]

        # Log candidate details for debugging
        logger.info(f"Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                logger.info(f"Safety: {rating.category} = {rating.probability}")

        if not candidate.content:
            logger.error("Candidate has no content!")
            logger.error(f"Full candidate object: {candidate}")
            if hasattr(candidate, 'finish_reason'):
                logger.error(f"Finish reason: {candidate.finish_reason}")
            raise ValueError(f"Gemini returned empty content. Finish reason: {getattr(candidate, 'finish_reason', 'unknown')}")

        if not candidate.content.parts:
            logger.error("Content has no parts!")
            logger.error(f"Content object: {candidate.content}")
            raise ValueError("Gemini content has no parts")

        if not hasattr(candidate.content.parts[0], 'inline_data') or not candidate.content.parts[0].inline_data:
            logger.error("No inline_data in parts!")
            logger.error(f"Parts[0]: {candidate.content.parts[0]}")
            raise ValueError("Gemini response has no audio data (inline_data missing)")

        audio_data = candidate.content.parts[0].inline_data.data
        mime_type = candidate.content.parts[0].inline_data.mime_type

        if not audio_data:
            logger.error("inline_data.data is empty")
            raise ValueError("No audio data in response")

        # Calculate audio duration estimate (24kHz, 16-bit mono = 48000 bytes/sec)
        audio_duration_sec = len(audio_data) / 48000
        logger.info(f"Audio received: {len(audio_data)} bytes ({len(audio_data)/1024:.1f} KB)")
        logger.info(f"Audio duration: ~{audio_duration_sec:.1f}s ({audio_duration_sec/60:.1f} min)")
        logger.info(f"Audio format: {mime_type}")
        logger.info(f"Processing ratio: {api_elapsed:.1f}s API time → {audio_duration_sec:.1f}s audio")

        # Convert to WAV
        temp_audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        with wave.open(temp_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)

        logger.info(f"Audio saved to: {temp_audio_path}")

        return temp_audio_path

    def _generate_with_chunking(self, dialogue_lines: List[Dict], speaker_voice_configs: List, tts_model: str = None):
        """Generate audio in chunks and concatenate them."""

        # Split into chunks
        chunks = self._create_dialogue_chunks(dialogue_lines)
        total_chunks = len(chunks)

        logger.info(f"======================================")
        logger.info(f"=== MULTI-CHUNK PODCAST GENERATION ===")
        logger.info(f"======================================")
        logger.info(f"Total dialogue lines: {len(dialogue_lines)}")
        logger.info(f"Split into {total_chunks} chunks")
        logger.info(f"Chunk config: max {self.MAX_LINES_PER_CHUNK} lines, max {self.MAX_CHARS_PER_CHUNK} chars")

        # Estimate total time
        total_chars = sum(len(f"{l['speaker']}: {l['text']}") for l in dialogue_lines)
        estimated_audio_minutes = total_chars / 800  # rough estimate: 800 chars ~= 1 min audio
        logger.info(f"Total input: {total_chars} chars (~{estimated_audio_minutes:.1f} min audio estimated)")

        # Generate audio for each chunk
        chunk_files = []
        chunk_times = []
        total_start_time = time.time()

        max_retries = 2
        retry_delay = 5.0  # seconds to wait before retry

        for i, chunk in enumerate(chunks):
            chunk_chars = sum(len(f"{l['speaker']}: {l['text']}") for l in chunk)
            logger.info(f"")
            logger.info(f">>> CHUNK {i+1}/{total_chunks} <<<")
            logger.info(f"Lines: {len(chunk)}, Chars: {chunk_chars}")

            chunk_start_time = time.time()
            chunk_file = None
            last_error = None

            # Retry loop for this chunk
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.warning(f"Retry {attempt}/{max_retries} for chunk {i+1}")
                        time.sleep(retry_delay * attempt)  # Progressive backoff

                    # Generate this chunk
                    chunk_file = self._generate_single_chunk(chunk, speaker_voice_configs, tts_model)
                    break  # Success, exit retry loop

                except Exception as e:
                    last_error = e
                    logger.warning(f"Chunk {i+1} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:100]}")
                    if attempt < max_retries:
                        logger.info(f"Will retry in {retry_delay * (attempt + 1)}s...")
                    continue

            if chunk_file is None:
                # All retries failed
                chunk_elapsed = time.time() - chunk_start_time
                total_elapsed = time.time() - total_start_time
                logger.error(f"")
                logger.error(f"!!! CHUNK {i+1}/{total_chunks} FAILED AFTER {max_retries+1} ATTEMPTS !!!")
                logger.error(f"Failed after {chunk_elapsed:.1f}s (total: {total_elapsed:.1f}s)")
                logger.error(f"Error type: {type(last_error).__name__}")
                logger.error(f"Error: {str(last_error)}")
                logger.error(f"Completed chunks before failure: {len(chunk_files)}")
                # Clean up partial chunks
                for f in chunk_files:
                    if os.path.exists(f):
                        os.unlink(f)
                raise last_error

            chunk_files.append(chunk_file)
            chunk_elapsed = time.time() - chunk_start_time
            chunk_times.append(chunk_elapsed)

            # Progress summary
            total_elapsed = time.time() - total_start_time
            avg_chunk_time = sum(chunk_times) / len(chunk_times)
            remaining_chunks = total_chunks - (i + 1)
            estimated_remaining = remaining_chunks * avg_chunk_time

            logger.info(f"Chunk {i+1} completed in {chunk_elapsed:.1f}s")
            logger.info(f"Progress: {i+1}/{total_chunks} chunks ({(i+1)/total_chunks*100:.0f}%)")
            logger.info(f"Total elapsed: {total_elapsed:.1f}s, Est. remaining: {estimated_remaining:.1f}s")

            # Rate limiting delay
            if i < len(chunks) - 1:
                logger.info(f"Waiting {self.INTER_CHUNK_DELAY}s before next chunk...")
                time.sleep(max(self.RATE_LIMIT_DELAY, self.INTER_CHUNK_DELAY))

        total_generation_time = time.time() - total_start_time
        logger.info(f"")
        logger.info(f"=== ALL CHUNKS GENERATED ===")
        logger.info(f"Total generation time: {total_generation_time:.1f}s ({total_generation_time/60:.1f} min)")
        logger.info(f"Average chunk time: {sum(chunk_times)/len(chunk_times):.1f}s")
        logger.info(f"Chunk times: {[f'{t:.1f}s' for t in chunk_times]}")

        # Concatenate chunks
        if self.pydub_available:
            return self._concatenate_with_pydub(chunk_files)
        else:
            return self._concatenate_with_wave(chunk_files)

    def _create_dialogue_chunks(self, dialogue_lines: List[Dict]) -> List[List[Dict]]:
        """Split dialogue into overlapping chunks for consistent voice generation."""

        chunks = []
        current_chunk = []
        current_char_count = 0

        for i, line in enumerate(dialogue_lines):
            line_chars = len(f"{line['speaker']}: {line['text']}")

            # Check if adding this line would exceed limits
            if (len(current_chunk) >= self.MAX_LINES_PER_CHUNK or
                    current_char_count + line_chars > self.MAX_CHARS_PER_CHUNK) and current_chunk:

                # Save current chunk
                chunks.append(current_chunk.copy())

                # Start new chunk with overlap
                if self.CHUNK_OVERLAP_LINES > 0 and len(current_chunk) > self.CHUNK_OVERLAP_LINES:
                    # Include last N lines from previous chunk for voice consistency
                    current_chunk = current_chunk[-self.CHUNK_OVERLAP_LINES:]
                    current_char_count = sum(
                        len(f"{l['speaker']}: {l['text']}") for l in current_chunk
                    )
                else:
                    current_chunk = []
                    current_char_count = 0

            current_chunk.append(line)
            current_char_count += line_chars

        # Add remaining lines as final chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Log chunk statistics
        for i, chunk in enumerate(chunks):
            chars = sum(len(f"{l['speaker']}: {l['text']}") for l in chunk)
            logger.info(f"Chunk {i+1}: {len(chunk)} lines, {chars} characters")

        return chunks

    def _concatenate_with_pydub(self, audio_files: List[str]) -> str:
        """Concatenate audio files using PyDub with silence between chunks."""
        from pydub import AudioSegment

        logger.info(f"Concatenating {len(audio_files)} audio files with PyDub")

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=1000)  # 1 second silence

        for i, file_path in enumerate(audio_files):
            audio = AudioSegment.from_wav(file_path)

            if i > 0:
                # Add silence between chunks
                combined += silence

            combined += audio

            # Clean up chunk file
            os.unlink(file_path)

        # Export combined audio
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        combined.export(output_path, format="wav")

        logger.info(f"Combined audio saved to: {output_path}")
        return output_path

    def _concatenate_with_wave(self, audio_files: List[str]) -> str:
        """Concatenate audio files using wave module (fallback if PyDub unavailable)."""

        logger.info(f"Concatenating {len(audio_files)} audio files with wave module")

        # Read all wave data
        frames = []
        params = None

        for file_path in audio_files:
            with wave.open(file_path, 'rb') as wav_file:
                if params is None:
                    params = wav_file.getparams()

                # Add audio frames
                frames.append(wav_file.readframes(wav_file.getnframes()))

                # Add 1 second of silence (24000 samples at 24kHz)
                if file_path != audio_files[-1]:
                    silence_frames = b'\x00\x00' * 24000
                    frames.append(silence_frames)

            # Clean up chunk file
            os.unlink(file_path)

        # Write combined audio
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        with wave.open(output_path, 'wb') as out_wav:
            out_wav.setparams(params)
            for frame_data in frames:
                out_wav.writeframes(frame_data)

        logger.info(f"Combined audio saved to: {output_path}")
        return output_path

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

            # Skip empty/tiny lines (reduced from 15 to preserve short dialogue)
            if len(text) < 2:
                logger.info(f"Skipping empty/tiny line: {text}")
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