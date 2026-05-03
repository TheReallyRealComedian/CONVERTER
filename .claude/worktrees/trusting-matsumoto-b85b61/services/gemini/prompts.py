# services/gemini/prompts.py
"""Prompt templates and tag-guidance calculation for Gemini script generation.

Pure functions / module-level constants. No external SDK calls, no I/O,
no class state. Imported by ``services.gemini.script`` (caller of
``format_dialogue_with_llm``).
"""
import logging

logger = logging.getLogger(__name__)


# Style-specific tag density (chars per tag) — used by ``calculate_tag_guidance``.
_TAG_DENSITY_MAP = {
    'documentary': 800,     # Moderate, placed at key revelations
    'conversational': 600,  # More frequent for natural feel
    'academic': 900,        # Sparse, at analytical shifts
    'satirical': 400,       # Dense for comedic timing/sarcasm
    'dramatic': 350         # Most expressive, pauses and emotion
}


# Style directive templates — selected by narration style and embedded into
# both the single- and multi-speaker prompt bodies.
STYLE_DIRECTIVES = {
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


def calculate_tag_guidance(raw_text: str, narration_style: str) -> dict:
    """
    Calculate recommended tag count based on content length and narration style.

    Returns dict with:
    - tag_range: "X-Y tags"
    - chars_per_tag: int
    - recommended_count: int
    """
    char_count = len(raw_text)
    word_count = len(raw_text.split())

    chars_per_tag = _TAG_DENSITY_MAP.get(narration_style, 800)
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


def build_single_speaker_prompt(style_directive, language_name, target_length,
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


def build_multi_speaker_prompt(style_directive, num_speakers, language_name,
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
