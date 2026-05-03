/* Audio converter page: live transcription, file transcription, podcast generator. */
document.addEventListener('DOMContentLoaded', function() {

    const PageData = window.PageData || {};
    const deepgramAvailable = !!PageData.deepgramApiKeySet;
    const geminiAvailable = !!PageData.geminiApiKeySet;

    // ==========================================================
    // ===== ALERT CONTAINERS (P4) =====
    // ==========================================================
    const liveAlertContainer = document.getElementById('live-alert-container');
    const fileAlertContainer = document.getElementById('file-alert-container');
    const podcastAlertContainer = document.getElementById('podcast-alert-container');

    // ==========================================================
    // ===== TAB SWITCHING =====
    // ==========================================================
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const languageSelector = document.getElementById('transcription-language-selector');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            tabButtons.forEach(b => {
                b.classList.remove('tab-active');
                b.classList.add('tab-inactive');
            });
            this.classList.add('tab-active');
            this.classList.remove('tab-inactive');

            const target = this.dataset.tab;
            tabPanes.forEach(p => p.classList.add('hidden'));
            document.getElementById(target + '-pane').classList.remove('hidden');

            if (target === 'live' || target === 'file') {
                languageSelector.style.display = 'block';
            } else {
                languageSelector.style.display = 'none';
            }
        });
    });

    // ==========================================================
    // ===== LANGUAGE SELECTION =====
    // ==========================================================
    const languageButtons = document.querySelectorAll('.language-btn');
    let selectedLanguage = 'en';

    languageButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.disabled) return;
            languageButtons.forEach(btn => {
                btn.classList.remove('lang-active');
                btn.classList.add('lang-inactive');
            });
            this.classList.add('lang-active');
            this.classList.remove('lang-inactive');
            selectedLanguage = this.dataset.lang;
        });
    });

    function lockLanguageButtons(locked) {
        languageButtons.forEach(btn => {
            btn.disabled = locked;
            const baseLabel = btn.dataset.langLabel || btn.textContent.trim();
            if (locked) {
                btn.title = 'Während der Aufnahme gesperrt';
                btn.setAttribute('aria-label', baseLabel + ' (während der Aufnahme gesperrt)');
            } else {
                btn.removeAttribute('title');
                btn.setAttribute('aria-label', baseLabel);
            }
        });
    }

    // ==========================================================
    // ===== SAVE-BUTTON LIFECYCLE HELPER (P6) =====
    // ==========================================================
    function resetSaveBtn(btn) {
        if (!btn) return;
        btn.disabled = false;
        btn.textContent = 'In Library speichern';
        btn.classList.remove('saved');
    }

    // ==========================================================
    // ===== LIVE TRANSCRIPTION LOGIC =====
    // ==========================================================
    const micButton = document.getElementById('mic-button');
    const transcriptOutput = document.getElementById('live-transcript-output');
    const liveRecordingHint = document.getElementById('live-recording-hint');

    let mediaRecorder;
    let socket;
    let isRecording = false;
    let baseText = '';
    let keepAliveInterval;

    const livePlaceholder = 'Live-Transkription erscheint hier …';
    const liveRecordingPlaceholder = 'Transkription läuft — bearbeitbar nach Stop';

    function setMicLoading(loading) {
        if (!micButton) return;
        if (loading) {
            micButton.classList.add('mic-loading');
            micButton.disabled = true;
            micButton.title = 'Verbindung wird aufgebaut …';
            micButton.setAttribute('aria-label', 'Verbindung zur Transkription wird aufgebaut');
        } else {
            micButton.classList.remove('mic-loading');
            micButton.disabled = false;
        }
    }

    function setMicRecording(recording) {
        if (!micButton) return;
        if (recording) {
            micButton.classList.add('recording');
            micButton.title = 'Aufnahme stoppen';
            micButton.setAttribute('aria-label', 'Aufnahme stoppen');
            micButton.setAttribute('aria-pressed', 'true');
        } else {
            micButton.classList.remove('recording');
            micButton.title = 'Aufnahme starten';
            micButton.setAttribute('aria-label', 'Aufnahme starten');
            micButton.setAttribute('aria-pressed', 'false');
        }
    }

    function setLiveTextareaReadonly(readonly) {
        if (!transcriptOutput) return;
        if (readonly) {
            transcriptOutput.readOnly = true;
            transcriptOutput.placeholder = liveRecordingPlaceholder;
            transcriptOutput.title = 'Während der Aufnahme schreibgeschützt';
            transcriptOutput.classList.add('is-readonly');
            if (liveRecordingHint) liveRecordingHint.classList.remove('hidden');
        } else {
            transcriptOutput.readOnly = false;
            transcriptOutput.placeholder = livePlaceholder;
            transcriptOutput.removeAttribute('title');
            transcriptOutput.classList.remove('is-readonly');
            if (liveRecordingHint) liveRecordingHint.classList.add('hidden');
        }
    }

    async function getDeepgramToken() {
        const response = await fetch('/api/get-deepgram-token');
        if (!response.ok) {
            const errorData = await safeJSON(response);
            throw new Error(errorData.error || 'Token-Abruf fehlgeschlagen');
        }
        const data = await safeJSON(response);
        return data.deepgram_token;
    }

    function teardownAudioPipeline() {
        clearInterval(keepAliveInterval);
        if (mediaRecorder) {
            if (mediaRecorder.processor) {
                mediaRecorder.processor.disconnect();
                mediaRecorder.processor.onaudioprocess = null;
            }
            if (mediaRecorder.source) mediaRecorder.source.disconnect();
            if (mediaRecorder.audioContext) mediaRecorder.audioContext.close();
            if (mediaRecorder.stream) mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        mediaRecorder = null;
    }

    const connectToDeepgram = async () => {
        setMicLoading(true);

        let deepgramToken;
        try {
            deepgramToken = await getDeepgramToken();
        } catch (error) {
            console.error('Failed to get token:', error);
            setMicLoading(false);
            showAlert(liveAlertContainer, 'danger',
                'API-Token konnte nicht abgerufen werden. Server-Konfiguration prüfen.');
            return;
        }

        const params = new URLSearchParams({
            model: 'nova-3',
            language: selectedLanguage,
            encoding: 'linear16',
            channels: '1',
            sample_rate: '16000',
            interim_results: 'true',
            smart_format: 'true',
            punctuate: 'true',
            numerals: 'true',
            utterances: 'true'
        });

        const deepgramUrl = `wss://api.deepgram.com/v1/listen?${params.toString()}`;
        socket = new WebSocket(deepgramUrl, ['token', deepgramToken]);

        socket.onopen = async () => {
            baseText = transcriptOutput.value;

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: { channelCount: 1, sampleRate: 16000 }
                });
            } catch (error) {
                console.error('getUserMedia failed:', error);
                setMicLoading(false);
                if (socket && socket.readyState === WebSocket.OPEN) socket.close();
                socket = null;

                let msg;
                if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    msg = 'Mikrofon-Zugriff blockiert. Erlaube den Zugriff in den Browser-Site-Einstellungen und versuche es erneut.';
                } else if (error.name === 'NotFoundError') {
                    msg = 'Kein Mikrofon gefunden. Schließe ein Aufnahmegerät an und versuche es erneut.';
                } else {
                    msg = 'Mikrofon konnte nicht gestartet werden. Browser-Berechtigung und angeschlossenes Gerät prüfen.';
                }
                showAlert(liveAlertContainer, 'danger', msg);
                return;
            }

            const audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmData = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    socket.send(pcmData.buffer);
                }
            };

            mediaRecorder = { processor, source, stream, audioContext };

            isRecording = true;
            setMicLoading(false);
            setMicRecording(true);
            setLiveTextareaReadonly(true);
            lockLanguageButtons(true);

            // Reset save-live button to neutral state for the new recording
            const saveLiveBtnEl = document.getElementById('save-live-btn');
            if (saveLiveBtnEl) {
                saveLiveBtnEl.classList.add('hidden');
                resetSaveBtn(saveLiveBtnEl);
            }

            keepAliveInterval = setInterval(() => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ type: 'KeepAlive' }));
                }
            }, 5000);
        };

        socket.onmessage = (message) => {
            const received = JSON.parse(message.data);
            const transcript = received.channel?.alternatives[0]?.transcript;
            if (transcript) {
                if (received.is_final) {
                    transcriptOutput.value = baseText + transcript + ' ';
                    baseText = transcriptOutput.value;
                } else {
                    transcriptOutput.value = baseText + transcript;
                }
            }
        };

        socket.onclose = () => {
            if (isRecording) stopRecording(false);
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            setMicLoading(false);
            showAlert(liveAlertContainer, 'danger',
                'Verbindung zur Transkription fehlgeschlagen. Netzwerk und API-Konfiguration prüfen.');
        };
    };

    const stopRecording = (shouldCloseSocket = true) => {
        if (!isRecording && shouldCloseSocket) return;

        teardownAudioPipeline();

        if (socket && socket.readyState === WebSocket.OPEN && shouldCloseSocket) {
            socket.close();
        }

        if (shouldCloseSocket) socket = null;
        isRecording = false;

        setMicRecording(false);
        setLiveTextareaReadonly(false);
        lockLanguageButtons(false);

        transcriptOutput.focus();

        const saveLiveBtn = document.getElementById('save-live-btn');
        if (saveLiveBtn && transcriptOutput.value.trim()) {
            saveLiveBtn.classList.remove('hidden');
            resetSaveBtn(saveLiveBtn);
        }
    };

    if (micButton) {
        micButton.addEventListener('click', () => {
            if (micButton.disabled) return;
            if (isRecording) stopRecording();
            else connectToDeepgram();
        });
    }

    // ==========================================================
    // ===== FILE UPLOAD TRANSCRIPTION LOGIC (P1, P2) =====
    // ==========================================================
    const uploadForm = document.getElementById('audio-upload-form');
    const fileInput = document.getElementById('file-upload-input');
    const fileUploadText = document.getElementById('file-upload-text');
    const fileDropZone = document.getElementById('file-drop-zone');
    const resultContainer = document.getElementById('transcription-result-container');
    const transcribeBtn = document.getElementById('transcribe-file-btn');

    const fileDefaultText = 'Audio-Datei hier ablegen oder klicken zum Auswählen';
    const fileResultPlaceholder = 'Transkription erscheint hier nach der Verarbeitung.';
    let fileWarningTimer = null;

    function clearFileInvalidState() {
        if (fileDropZone) fileDropZone.classList.remove('c-drop-zone--invalid');
    }

    function clearFileWarningState() {
        if (fileDropZone) fileDropZone.classList.remove('c-drop-zone--warning');
        if (fileWarningTimer) {
            clearTimeout(fileWarningTimer);
            fileWarningTimer = null;
        }
    }

    function resetFileResultArea() {
        if (resultContainer) {
            resultContainer.innerHTML = '';
            const placeholder = document.createElement('span');
            placeholder.className = 'text-neo-faint';
            placeholder.dataset.placeholder = 'true';
            placeholder.textContent = fileResultPlaceholder;
            resultContainer.appendChild(placeholder);
        }
        const saveBtn = document.getElementById('save-transcription-btn');
        if (saveBtn) {
            saveBtn.classList.add('hidden');
            resetSaveBtn(saveBtn);
            saveBtn._transcriptionData = null;
        }
    }

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileUploadText.textContent = fileInput.files[0].name;
                clearFileInvalidState();
                if (fileAlertContainer) fileAlertContainer.innerHTML = '';
            } else {
                fileUploadText.textContent = fileDefaultText;
            }
        });
    }

    if (fileDropZone) {
        fileDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            // Best-effort MIME-detection during dragover. Browser security often
            // hides the file's type until drop, so we only flip into warning
            // when the type is present and clearly not in our accept list.
            let unsupported = false;
            const items = e.dataTransfer && e.dataTransfer.items;
            if (items && items.length === 1 && items[0].kind === 'file') {
                const t = (items[0].type || '').toLowerCase();
                if (t && t !== 'application/octet-stream') {
                    unsupported = !t.startsWith('audio/');
                }
            }
            if (unsupported) {
                fileDropZone.classList.add('c-drop-zone--warning');
                fileDropZone.classList.remove('drop-zone-active');
            } else {
                clearFileWarningState();
                fileDropZone.classList.add('drop-zone-active');
            }
        });

        fileDropZone.addEventListener('dragleave', () => {
            fileDropZone.classList.remove('drop-zone-active');
            clearFileWarningState();
        });

        fileDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            fileDropZone.classList.remove('drop-zone-active');
            clearFileWarningState();
            if (!e.dataTransfer.files.length) return;
            const file = e.dataTransfer.files[0];
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            fileUploadText.textContent = file.name;
            clearFileInvalidState();
            if (fileAlertContainer) fileAlertContainer.innerHTML = '';
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            if (!fileInput.files[0]) {
                showAlert(fileAlertContainer, 'danger',
                    'Bitte zuerst eine Audio-Datei auswählen oder per Drag & Drop hineinziehen.');
                if (fileDropZone) {
                    fileDropZone.classList.add('c-drop-zone--invalid');
                    setTimeout(() => clearFileInvalidState(), 2000);
                    fileDropZone.focus();
                }
                return;
            }

            const formData = new FormData();
            formData.append('audio_file', fileInput.files[0]);
            formData.append('language', selectedLanguage);

            transcribeBtn.disabled = true;
            transcribeBtn.textContent = 'Wird umgewandelt …';
            resultContainer.innerHTML = '';
            const processingSpan = document.createElement('span');
            processingSpan.className = 'text-neo-faint';
            processingSpan.textContent = 'Audio-Datei wird verarbeitet …';
            resultContainer.appendChild(processingSpan);

            const saveBtn = document.getElementById('save-transcription-btn');
            if (saveBtn) {
                saveBtn.classList.add('hidden');
                resetSaveBtn(saveBtn);
            }

            try {
                const response = await fetch('/transcribe-audio-file', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const result = await safeJSON(response);
                    throw new Error(result.error || `Transkription fehlgeschlagen (${response.status})`);
                }

                const result = await safeJSON(response);
                resultContainer.textContent = result.transcript || 'Es wurde kein Transkript zurückgegeben.';

                if (saveBtn) {
                    saveBtn.classList.remove('hidden');
                    resetSaveBtn(saveBtn);
                    saveBtn._transcriptionData = {
                        content: result.transcript,
                        filename: fileInput.files[0].name,
                        mimetype: fileInput.files[0].type,
                        size: fileInput.files[0].size,
                        metadata: result.metadata || {}
                    };
                }

            } catch (error) {
                console.error('Transcription error:', error);
                resetFileResultArea();
                showAlert(fileAlertContainer, 'danger',
                    'Transkription fehlgeschlagen. Datei prüfen und erneut versuchen.');
            } finally {
                transcribeBtn.disabled = !deepgramAvailable;
                transcribeBtn.textContent = 'Datei umwandeln';
            }
        });
    }

    // ==========================================================
    // ===== COPY & CLEAR BUTTONS (P11) =====
    // ==========================================================
    const clearLiveBtn = document.getElementById('clear-live-btn');
    const clearFileBtn = document.getElementById('clear-file-btn');
    const copyLiveBtn = document.getElementById('copy-live-btn');
    const copyFileBtn = document.getElementById('copy-file-btn');

    if (clearLiveBtn) {
        clearLiveBtn.addEventListener('click', () => {
            transcriptOutput.value = '';
            baseText = '';
            const saveLiveBtn = document.getElementById('save-live-btn');
            if (saveLiveBtn) {
                saveLiveBtn.classList.add('hidden');
                resetSaveBtn(saveLiveBtn);
            }
        });
    }

    if (clearFileBtn) {
        clearFileBtn.addEventListener('click', () => {
            resetFileResultArea();
            if (fileInput) fileInput.value = '';
            if (fileUploadText) fileUploadText.textContent = fileDefaultText;
            if (fileAlertContainer) fileAlertContainer.innerHTML = '';
        });
    }

    function setCopySuccessLabel(button) {
        const label = button.querySelector('.copy-btn__label');
        if (!label) return;
        const original = label.textContent;
        label.textContent = '✓ Kopiert';
        setTimeout(() => { label.textContent = original; }, 2000);
    }

    async function copyToClipboard(text, button, alertContainer) {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                await fallbackCopyText(text);
            }
            setCopySuccessLabel(button);
        } catch (err) {
            console.error('Failed to copy:', err);
            showAlert(alertContainer, 'danger', 'Kopieren in die Zwischenablage fehlgeschlagen.');
        }
    }

    if (copyLiveBtn) {
        copyLiveBtn.addEventListener('click', () => {
            const text = transcriptOutput.value;
            if (!text || !text.trim()) {
                showAlert(liveAlertContainer, 'warning', 'Es gibt nichts zu kopieren.');
                return;
            }
            copyToClipboard(text, copyLiveBtn, liveAlertContainer);
        });
    }

    if (copyFileBtn) {
        copyFileBtn.addEventListener('click', () => {
            // Sentinel-based empty-detection: placeholder span carries
            // data-placeholder="true"; presence means "no real content".
            const placeholder = resultContainer.querySelector('[data-placeholder="true"]');
            const text = resultContainer.textContent;
            if (placeholder || !text || !text.trim()) {
                showAlert(fileAlertContainer, 'warning', 'Es gibt nichts zu kopieren.');
                return;
            }
            copyToClipboard(text, copyFileBtn, fileAlertContainer);
        });
    }

    // ==========================================================
    // ===== PODCAST GENERATOR =====
    // ==========================================================

    const podcastLanguageSelect = document.getElementById('podcast-language');
    const podcastRawText = document.getElementById('podcast-raw-text');
    const podcastScript = document.getElementById('podcast-script');
    const generateScriptBtn = document.getElementById('generate-script-btn');
    const generatePodcastBtn = document.getElementById('generate-podcast-btn');
    const podcastResultContainer = document.getElementById('podcast-result-container');
    const podcastAudio = document.getElementById('podcast-audio');
    const podcastAudioSource = document.getElementById('podcast-audio-source');
    const downloadPodcastBtn = document.getElementById('download-podcast-btn');

    const promptEditorToggle = document.getElementById('prompt-editor-toggle');
    const promptEditorContent = document.getElementById('prompt-editor-content');
    const promptToggleIcon = document.getElementById('prompt-toggle-icon');
    const customPromptEditor = document.getElementById('custom-prompt-editor');
    const resetPromptBtn = document.getElementById('reset-prompt-btn');

    if (promptEditorToggle) {
        promptEditorToggle.addEventListener('click', function() {
            const isExpanded = promptEditorContent.classList.contains('expanded');
            if (isExpanded) {
                promptEditorContent.classList.remove('expanded');
                promptToggleIcon.innerHTML = '&#9660;';
            } else {
                promptEditorContent.classList.add('expanded');
                promptToggleIcon.innerHTML = '&#9650;';
            }
        });
    }

    if (resetPromptBtn) {
        resetPromptBtn.addEventListener('click', function() {
            customPromptEditor.value = '';
        });
    }

    if (generateScriptBtn) {
        generateScriptBtn.addEventListener('click', async () => {
            const rawText = podcastRawText.value.trim();

            if (!rawText) {
                showAlert(podcastAlertContainer, 'warning',
                    'Bitte erst Quelltext im Feld oben eintragen.');
                return;
            }

            const mode = document.querySelector('input[name="podcast-mode"]:checked').value;
            const numSpeakers = mode === 'monolog' ? 1 : 2;

            const speakerDescriptions = numSpeakers === 1
                ? [{ name: 'Kate', voice: 'Zephyr', personality: 'warm and professional' }]
                : [
                    { name: 'Kate', voice: 'Zephyr', personality: 'warm and enthusiastic' },
                    { name: 'Max', voice: 'Charon', personality: 'professional and informative' }
                  ];

            const labelEl = generateScriptBtn.querySelector('.generate-script-btn__label');
            generateScriptBtn.disabled = true;
            if (labelEl) labelEl.textContent = 'Generiert …';

            try {
                const customPrompt = customPromptEditor.value.trim() || null;

                const response = await fetch('/format-dialogue-with-llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        raw_text: rawText,
                        num_speakers: numSpeakers,
                        speaker_descriptions: speakerDescriptions,
                        language: podcastLanguageSelect.value,
                        narration_style: document.getElementById('narration-style').value,
                        script_length: 'medium',
                        custom_prompt: customPrompt
                    })
                });

                if (!response.ok) {
                    const errorData = await safeJSON(response);
                    throw new Error(errorData.error || `HTTP-Fehler! Status: ${response.status}`);
                }

                const data = await safeJSON(response);
                podcastScript.value = data.raw_formatted_text;

            } catch (error) {
                console.error('Script generation error:', error);
                showAlert(podcastAlertContainer, 'danger',
                    'Skript-Generierung fehlgeschlagen. Bitte erneut versuchen.');
            } finally {
                generateScriptBtn.disabled = !geminiAvailable;
                if (labelEl) labelEl.textContent = 'Skript aus Quelltext generieren';
            }
        });
    }

    if (generatePodcastBtn) {
        generatePodcastBtn.addEventListener('click', async () => {
            const scriptText = podcastScript.value.trim();

            if (!scriptText) {
                showAlert(podcastAlertContainer, 'warning',
                    'Bitte erst ein Skript eintragen oder generieren lassen.');
                return;
            }

            const mode = document.querySelector('input[name="podcast-mode"]:checked').value;

            const dialogue = [];
            const voiceMap = { 'Kate': 'Zephyr', 'Max': 'Charon' };

            for (const line of scriptText.split('\n')) {
                const trimmedLine = line.trim();
                if (!trimmedLine || trimmedLine.startsWith('#') || trimmedLine.startsWith('**')) continue;

                if (trimmedLine.includes(':')) {
                    const parts = trimmedLine.split(':', 1);
                    const speakerPart = parts[0].trim();
                    const textPart = trimmedLine.substring(parts[0].length + 1).trim();

                    let speaker = speakerPart;
                    let style = '';

                    if (speakerPart.includes('[') && speakerPart.includes(']')) {
                        speaker = speakerPart.split('[')[0].trim();
                        style = speakerPart.split('[')[1].split(']')[0].trim();
                    }

                    const voice = voiceMap[speaker] || 'Kore';

                    if (textPart) {
                        dialogue.push({ speaker: voice, style: style, text: textPart });
                    }
                }
            }

            if (dialogue.length === 0) {
                showAlert(podcastAlertContainer, 'danger',
                    'Skript konnte nicht gelesen werden. Format prüfen: Sprecher [stil]: Text.');
                return;
            }

            generatePodcastBtn.disabled = true;
            generatePodcastBtn.textContent = 'Generiert …';
            podcastResultContainer.classList.add('hidden');

            try {
                const ttsModelSelect = document.getElementById('tts-model');
                const startResponse = await fetch('/generate-gemini-podcast', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dialogue: dialogue,
                        language: podcastLanguageSelect.value,
                        tts_model: ttsModelSelect.value
                    })
                });

                if (!startResponse.ok) {
                    const errorData = await safeJSON(startResponse);
                    throw new Error(errorData.error || `HTTP-Fehler! Status: ${startResponse.status}`);
                }

                const { job_id } = await safeJSON(startResponse);
                generatePodcastBtn.textContent = 'Generiert …';

                let status = 'pending';
                let pollCount = 0;
                while (status === 'pending' || status === 'processing') {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    pollCount++;
                    generatePodcastBtn.textContent = `Generiert … (${pollCount * 2} s)`;

                    const statusResponse = await fetch(`/podcast-status/${job_id}`);
                    const statusData = await safeJSON(statusResponse);
                    status = statusData.status;

                    if (status === 'failed') {
                        throw new Error(statusData.error || 'Generierung fehlgeschlagen');
                    }
                }

                const downloadResponse = await fetch(`/podcast-download/${job_id}`);
                if (!downloadResponse.ok) {
                    throw new Error('Podcast-Download fehlgeschlagen');
                }

                const audioBlob = await downloadResponse.blob();
                const audioUrl = URL.createObjectURL(audioBlob);

                podcastAudioSource.src = audioUrl;
                podcastAudio.load();

                downloadPodcastBtn.href = audioUrl;

                podcastResultContainer.classList.remove('hidden');
                podcastResultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } catch (error) {
                console.error('Podcast generation error:', error);
                showAlert(podcastAlertContainer, 'danger',
                    'Podcast-Generierung fehlgeschlagen. Bitte erneut versuchen.');
            } finally {
                generatePodcastBtn.disabled = !geminiAvailable;
                generatePodcastBtn.textContent = 'Podcast generieren';
            }
        });
    }

    // ==========================================================
    // ===== SAVE TO LIBRARY (P6 lifecycle) =====
    // ==========================================================
    const saveTranscriptionBtn = document.getElementById('save-transcription-btn');
    if (saveTranscriptionBtn) {
        saveTranscriptionBtn.addEventListener('click', async function() {
            const btn = this;
            const data = btn._transcriptionData;
            if (!data) return;

            btn.disabled = true;
            btn.textContent = 'Speichert …';

            try {
                const stem = data.filename.replace(/\.[^.]+$/, '');
                const response = await fetch('/api/conversions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        conversion_type: 'audio_transcription',
                        title: stem,
                        content: data.content,
                        source_filename: data.filename,
                        source_mimetype: data.mimetype,
                        source_size_bytes: data.size,
                        metadata: data.metadata
                    })
                });
                if (response.ok) {
                    btn.textContent = '✓ Gespeichert';
                    btn.classList.add('saved');
                    return;
                }

                let serverError = null;
                try {
                    const errData = await safeJSON(response);
                    serverError = errData && errData.error;
                } catch (_) { /* fall back to generic message */ }

                resetSaveBtn(btn);
                const msg = serverError
                    ? 'Speichern in die Library fehlgeschlagen: ' + serverError + '.'
                    : 'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.';
                showAlert(fileAlertContainer, 'danger', msg);
            } catch (_err) {
                resetSaveBtn(btn);
                showAlert(fileAlertContainer, 'danger',
                    'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.');
            }
        });
    }

    const saveLiveBtn = document.getElementById('save-live-btn');
    if (saveLiveBtn) {
        saveLiveBtn.addEventListener('click', async function() {
            const btn = this;
            const content = document.getElementById('live-transcript-output').value.trim();
            if (!content) {
                showAlert(liveAlertContainer, 'warning',
                    'Es gibt nichts zu speichern. Erst eine Live-Transkription aufnehmen.');
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Speichert …';

            try {
                const dateStr = new Intl.DateTimeFormat('de-DE', {
                    dateStyle: 'short',
                    timeStyle: 'short'
                }).format(new Date());
                const title = `Live-Transkription ${dateStr}`;
                const response = await fetch('/api/conversions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        conversion_type: 'audio_transcription',
                        title: title,
                        content: content,
                        source_filename: 'live-recording',
                        metadata: { source: 'live_transcription' }
                    })
                });
                if (response.ok) {
                    btn.textContent = '✓ Gespeichert';
                    btn.classList.add('saved');
                    return;
                }

                let serverError = null;
                try {
                    const errData = await safeJSON(response);
                    serverError = errData && errData.error;
                } catch (_) { /* fall back to generic message */ }

                resetSaveBtn(btn);
                const msg = serverError
                    ? 'Speichern in die Library fehlgeschlagen: ' + serverError + '.'
                    : 'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.';
                showAlert(liveAlertContainer, 'danger', msg);
            } catch (_err) {
                resetSaveBtn(btn);
                showAlert(liveAlertContainer, 'danger',
                    'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.');
            }
        });
    }

});
