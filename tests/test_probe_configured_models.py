"""OKTOBER P3 — the model probe reads the CONFIGURED names (never a literal
list) and is loud when a model stops answering. The real calls are not made
here (they cost money and need credentials); what is pinned is the contract:
names come from the modules that call the models, a dead model yields exit 1
with its name and error in the output, a setup problem yields exit 2."""
import importlib.util
import io
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'probe_configured_models.py'


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location('probe_configured_models', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_names_come_from_the_configuration_not_a_literal_list(probe, monkeypatch):
    import services.narration_render as narration_render
    import services.pdf_cloud as pdf_cloud

    names = [entry[1] for entry in probe.configured_models()]
    assert names == [narration_render.DEFAULT_NARRATION_MODEL, pdf_cloud.DEFAULT_CLOUD_PDF_MODEL]

    # A changed configuration must change what gets probed — no copy of the names.
    monkeypatch.setattr(narration_render, 'DEFAULT_NARRATION_MODEL', 'gemini-9.9-flash-tts')
    monkeypatch.setattr(pdf_cloud, 'DEFAULT_CLOUD_PDF_MODEL', 'gemini-9.9-flash')
    assert [entry[1] for entry in probe.configured_models()] == ['gemini-9.9-flash-tts', 'gemini-9.9-flash']


def test_source_label_says_whether_the_name_is_an_env_override(probe, monkeypatch):
    monkeypatch.delenv('NARRATION_TTS_MODEL', raising=False)
    assert probe._source('NARRATION_TTS_MODEL') == 'Code-Default'
    monkeypatch.setenv('NARRATION_TTS_MODEL', 'gemini-3.1-flash-tts-preview')
    assert probe._source('NARRATION_TTS_MODEL') == 'env NARRATION_TTS_MODEL'


def test_a_dead_model_is_loud_and_names_itself(probe):
    def answers(name, timeout):
        return 'WAV 1 B'

    def dead(name, timeout):
        raise RuntimeError('404 NOT_FOUND: model not found')

    out = io.StringIO()
    code = probe.run([('A', 'model-alive', 'Code-Default', answers),
                      ('B', 'model-dead', 'env X', dead)], timeout=1, out=out)
    text = out.getvalue()
    assert code == 1
    assert 'FAIL' in text and 'model-dead' in text and '404 NOT_FOUND' in text
    assert 'OK' in text and 'model-alive' in text
    assert '1 von 2' in text


def test_all_answering_is_exit_zero(probe):
    out = io.StringIO()
    code = probe.run([('A', 'model-alive', 'Code-Default', lambda n, t: 'Beleg')], timeout=1, out=out)
    assert code == 0
    assert 'Alle 1' in out.getvalue()


def test_setup_problems_are_distinguished_from_dead_models(probe):
    def no_credentials(name, timeout):
        raise probe.ProbeSetupError('GEMINI_API_KEY nicht gesetzt')

    out = io.StringIO()
    code = probe.run([('A', 'model-x', 'Code-Default', no_credentials)], timeout=1, out=out)
    assert code == 2
    assert 'SETUP' in out.getvalue() and 'GEMINI_API_KEY' in out.getvalue()

    # a FAIL next to a SETUP problem still dominates (a dead model is the louder fact)
    out = io.StringIO()
    code = probe.run([('A', 'model-x', 'Code-Default', no_credentials),
                      ('B', 'model-dead', 'Code-Default', lambda n, t: (_ for _ in ()).throw(RuntimeError('404')))],
                     timeout=1, out=out)
    assert code == 1
