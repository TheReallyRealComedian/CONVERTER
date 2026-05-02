# services/gemini/__init__.py
#
# Stage 3 decomposition target. ``GeminiService`` (the public class) will be
# re-exported here once the submodules are populated. For now this is an
# empty package alongside the legacy ``services/gemini_service.py``; existing
# imports (``from services import GeminiService``) continue to resolve via
# ``services/__init__.py`` against the legacy module.
