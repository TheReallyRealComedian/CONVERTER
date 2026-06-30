"""Canonical Gemini-TTS voice catalog (reference data).

The named Gemini TTS voices, grouped by perceived register, each with a short
character description. Relocated here from the (now-removed) alt-podcast routes
in NARR-5: the alt-podcast flow that exposed it via ``/api/get-gemini-voices``
is gone, but the catalog itself is small, canonical, and worth keeping as a
reference — the faithful-narration skill picks voice ids from this set, and a
future read endpoint could surface it. No live code consumer today; pure data.
"""

GEMINI_VOICES = {
    "male": [
        {"name": "Kore", "description": "Firm and authoritative"},
        {"name": "Charon", "description": "Informative and clear"},
        {"name": "Fenrir", "description": "Excitable and energetic"},
        {"name": "Orus", "description": "Firm and steady"},
        {"name": "Puck", "description": "Upbeat and cheerful"},
        {"name": "Enceladus", "description": "Breathy and soft"},
        {"name": "Iapetus", "description": "Clear and precise"},
        {"name": "Algenib", "description": "Gravelly and deep"},
        {"name": "Achernar", "description": "Soft and gentle"},
        {"name": "Algieba", "description": "Smooth and polished"},
        {"name": "Gacrux", "description": "Mature and experienced"},
        {"name": "Alnilam", "description": "Firm and direct"},
        {"name": "Rasalgethi", "description": "Informative and educational"},
        {"name": "Sadaltager", "description": "Knowledgeable and wise"},
        {"name": "Zubenelgenubi", "description": "Casual and relaxed"},
    ],
    "female": [
        {"name": "Zephyr", "description": "Bright and lively"},
        {"name": "Leda", "description": "Youthful and fresh"},
        {"name": "Laomedeia", "description": "Upbeat and positive"},
        {"name": "Aoede", "description": "Breezy and light"},
        {"name": "Callirrhoe", "description": "Easy-going and friendly"},
        {"name": "Autonoe", "description": "Bright and clear"},
        {"name": "Erinome", "description": "Clear and articulate"},
        {"name": "Umbriel", "description": "Easy-going and calm"},
        {"name": "Despina", "description": "Smooth and flowing"},
        {"name": "Pulcherrima", "description": "Forward and confident"},
        {"name": "Vindemiatrix", "description": "Gentle and warm"},
    ],
    "neutral": [
        {"name": "Kore", "description": "Firm (can be male or female)"},
        {"name": "Achird", "description": "Friendly and approachable"},
        {"name": "Schedar", "description": "Even and balanced"},
        {"name": "Sadachbia", "description": "Lively and animated"},
        {"name": "Sulafat", "description": "Warm and inviting"},
    ],
}
