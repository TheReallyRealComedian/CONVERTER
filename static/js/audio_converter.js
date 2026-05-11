/* Audio converter page: live transcription, file transcription, podcast generator. */
document.addEventListener('DOMContentLoaded', function() {

    const PageData = window.PageData || {};
    const deepgramAvailable = !!PageData.deepgramApiKeySet;
    const geminiAvailable = !!PageData.geminiApiKeySet;
    const acceptedAudioExtensions = PageData.acceptedAudioExtensions || [];
    const maxAudioFileSizeMb = Number(PageData.maxAudioFileSizeMb) || 500;
    const maxRawTextChars = Number(PageData.maxRawTextChars) || 50000;
    const acceptedAudioExtensionsLabel = 'MP3, WAV, M4A, OGG, FLAC, WEBM';
    const PODCAST_ACTIVE_JOB_STORAGE_KEY = 'podcast.activeJobId';

    function getFileExtension(filename) {
        const m = /\.([^.\\/]+)$/.exec(filename || '');
        return m ? m[1].toLowerCase() : '';
    }

    function isAcceptedAudioFilename(filename) {
        if (!acceptedAudioExtensions.length) return true;
        return acceptedAudioExtensions.includes(getFileExtension(filename));
    }

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
            // F-4.3 P12: service-gated tabs carry aria-disabled="true" in markup
            // (Deepgram / Gemini key missing). Without this guard the click still
            // activated the pane and exposed a non-functional sub-UI.
            if (this.getAttribute('aria-disabled') === 'true') {
                showToast('Service nicht konfiguriert', { level: 'info' });
                return;
            }
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
            transcriptOutput.setAttribute('aria-busy', 'true');
            if (liveRecordingHint) liveRecordingHint.classList.remove('hidden');
        } else {
            transcriptOutput.readOnly = false;
            transcriptOutput.placeholder = livePlaceholder;
            transcriptOutput.removeAttribute('title');
            transcriptOutput.classList.remove('is-readonly');
            transcriptOutput.setAttribute('aria-busy', 'false');
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

        // Acquire the mic before opening the WebSocket so the browser
        // permission prompt fires before any network handshake. Denial leaves
        // no socket dangling.
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000 }
            });
        } catch (error) {
            console.error('getUserMedia failed:', error);
            setMicLoading(false);

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

        // If the user revokes mic access mid-recording (browser tab indicator,
        // hardware switch), tear down the live session.
        const audioTrack = stream.getAudioTracks()[0];
        if (audioTrack) {
            audioTrack.addEventListener('ended', () => {
                if (isRecording) {
                    showAlert(liveAlertContainer, 'warning',
                        'Mikrofon-Verbindung beendet. Aufnahme wurde gestoppt.');
                    stopRecording();
                }
            });
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

        socket.onopen = () => {
            baseText = transcriptOutput.value;

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
            if (isRecording) {
                stopRecording(false);
            } else if (stream) {
                // WS closed before onopen — release acquired mic.
                stream.getTracks().forEach(t => t.stop());
            }
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            setMicLoading(false);
            if (!isRecording && stream) {
                stream.getTracks().forEach(t => t.stop());
            }
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
                const f = fileInput.files[0];
                fileUploadText.textContent = `${f.name} (${formatFileSize(f.size)})`;
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
            if (!isAcceptedAudioFilename(file.name)) {
                showAlert(fileAlertContainer, 'danger',
                    'Dieses Dateiformat wird nicht unterstützt. Erlaubt: '
                    + acceptedAudioExtensionsLabel + '.');
                fileDropZone.classList.add('c-drop-zone--warning');
                if (fileWarningTimer) clearTimeout(fileWarningTimer);
                fileWarningTimer = setTimeout(() => clearFileWarningState(), 2000);
                return;
            }
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            fileUploadText.textContent = `${file.name} (${formatFileSize(file.size)})`;
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

            const selectedFile = fileInput.files[0];
            if (!isAcceptedAudioFilename(selectedFile.name)) {
                showAlert(fileAlertContainer, 'danger',
                    'Dieses Dateiformat wird nicht unterstützt. Erlaubt: '
                    + acceptedAudioExtensionsLabel + '.');
                return;
            }
            if (selectedFile.size > maxAudioFileSizeMb * 1024 * 1024) {
                showAlert(fileAlertContainer, 'danger',
                    'Datei ist zu groß. Maximum: ' + maxAudioFileSizeMb + ' MB.');
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
            if (!confirmIfLong(transcriptOutput.value,
                'Live-Transkription wirklich leeren? Der Text geht verloren.')) {
                return;
            }
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
            const placeholder = resultContainer && resultContainer.querySelector('[data-placeholder="true"]');
            const currentText = placeholder ? '' : (resultContainer ? resultContainer.textContent : '');
            if (!confirmIfLong(currentText,
                'Transkriptions-Ergebnis wirklich leeren? Der Text geht verloren.')) {
                return;
            }
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
    const podcastRawTextCounter = document.getElementById('podcast-raw-text-counter');
    const podcastScript = document.getElementById('podcast-script');
    const podcastScriptStatusHint = document.getElementById('podcast-script-status-hint');
    const generateScriptBtn = document.getElementById('generate-script-btn');
    const generatePodcastBtn = document.getElementById('generate-podcast-btn');
    const cancelPodcastBtn = document.getElementById('cancel-podcast-btn');
    const podcastResultContainer = document.getElementById('podcast-result-container');
    const podcastAudio = document.getElementById('podcast-audio');
    const podcastAudioSource = document.getElementById('podcast-audio-source');
    const downloadPodcastBtn = document.getElementById('download-podcast-btn');
    const podcastStageIndicator = document.getElementById('podcast-stage-indicator');
    const podcastStageText = document.getElementById('podcast-stage-text');
    const podcastStageCounter = document.getElementById('podcast-stage-counter');
    const podcastStageBar = document.getElementById('podcast-stage-bar');
    const podcastStageBarFill = document.getElementById('podcast-stage-bar-fill');

    // Cancel-state machine: idle → confirm-pending → cancelling.
    // confirm-pending is a 5 s window where a second click confirms; the btn
    // returns to idle on timeout or other interaction. cancelling is a
    // post-roundtrip terminal-pending state — polling-loop sees the worker
    // status change and resolves the UI then.
    const CANCEL_BTN_LABEL_IDLE = 'Abbrechen';
    const CANCEL_BTN_LABEL_CONFIRM = 'Ja, abbrechen';
    const CANCEL_BTN_LABEL_CANCELLING = 'Wird abgebrochen …';
    const CANCEL_CONFIRM_TIMEOUT_MS = 5000;
    // F-4.3 P5: queue-stuck diagnostic banner after this much time without a
    // worker pickup — single-user single-worker setup means a stuck queue is
    // almost always "worker container is down".
    const QUEUE_STUCK_MS = 30000;
    // F-4.3 P8: defensive ceilings for the polling loop. RETRY_MAX is a server-
    // 5xx retry budget; TIMEOUT_MS is the absolute "something is very wrong"
    // ceiling that surfaces a banner with a manual cancel option.
    const POLLING_INTERVAL_MS = 2000;
    const POLLING_RETRY_MAX = 3;
    const POLLING_TIMEOUT_MS = 10 * 60 * 1000;
    const POLLING_KNOWN_STATES = new Set([
        'queued', 'started', 'processing', 'finished', 'completed',
        'failed', 'cancelled', 'canceled', 'stopped', 'deferred', 'scheduled',
    ]);
    let podcastJobId = null;
    let podcastCancelState = 'idle';
    let podcastCancelConfirmTimer = null;
    // F-4.3 P6 / BT4: track the most recent blob URL so we can revoke the
    // previous one on each fresh generate or page-unload.
    let lastPodcastBlobUrl = null;

    // F-4.3 P11: lock the script textarea while the worker is generating.
    function setPodcastScriptReadonly(readonly) {
        if (!podcastScript) return;
        if (readonly) {
            podcastScript.readOnly = true;
            podcastScript.classList.add('is-readonly');
            podcastScript.title = 'Während der Generierung schreibgeschützt';
            podcastScript.setAttribute('aria-busy', 'true');
            if (podcastScriptStatusHint) podcastScriptStatusHint.classList.remove('hidden');
        } else {
            podcastScript.readOnly = false;
            podcastScript.classList.remove('is-readonly');
            podcastScript.removeAttribute('title');
            podcastScript.setAttribute('aria-busy', 'false');
            if (podcastScriptStatusHint) podcastScriptStatusHint.classList.add('hidden');
        }
    }

    // F-4.3 P3: render the stage sub-caption + (optional) chunk-progress bar.
    // ``meta`` is whatever /podcast-status returned for the started state —
    // may include {stage, chunk_current, chunk_total}.
    function setPodcastStage(meta) {
        if (!podcastStageIndicator) return;
        if (!meta || !meta.stage) {
            hidePodcastStage();
            return;
        }
        podcastStageIndicator.classList.remove('hidden');
        const labels = {
            filtering: 'Skript wird gefiltert …',
            chunking: 'Wird aufgeteilt …',
            tts_chunk: 'Wird gesprochen …',
            concatenating: 'Audio wird zusammengefügt …',
            finalizing: 'Datei wird abgeschlossen …',
        };
        podcastStageText.textContent = labels[meta.stage] || meta.stage;
        if (meta.stage === 'tts_chunk' && meta.chunk_total) {
            podcastStageCounter.textContent = ' (Chunk ' + meta.chunk_current
                + '/' + meta.chunk_total + ')';
            podcastStageBar.classList.remove('hidden');
            const pct = Math.max(0, Math.min(100,
                Math.round((meta.chunk_current / meta.chunk_total) * 100)));
            podcastStageBarFill.style.width = pct + '%';
        } else {
            podcastStageCounter.textContent = '';
            podcastStageBar.classList.add('hidden');
        }
    }
    function hidePodcastStage() {
        if (!podcastStageIndicator) return;
        podcastStageIndicator.classList.add('hidden');
        podcastStageText.textContent = '';
        podcastStageCounter.textContent = '';
        podcastStageBar.classList.add('hidden');
        podcastStageBarFill.style.width = '0%';
    }
    function setPodcastStageQueueWaiting() {
        if (!podcastStageIndicator) return;
        podcastStageIndicator.classList.remove('hidden');
        podcastStageText.textContent = 'Wartet auf Worker …';
        podcastStageCounter.textContent = '';
        podcastStageBar.classList.add('hidden');
    }

    // F-4.3 P7: unified loading state for the two generate buttons.
    function setBtnLoading(btn, loading) {
        if (!btn) return;
        if (loading) {
            btn.classList.add('is-loading');
            btn.setAttribute('aria-busy', 'true');
        } else {
            btn.classList.remove('is-loading');
            btn.setAttribute('aria-busy', 'false');
        }
    }

    // F-4.3 P4: localStorage helpers for browser-reload recovery.
    function rememberActiveJob(jobId) {
        try { localStorage.setItem(PODCAST_ACTIVE_JOB_STORAGE_KEY, jobId); }
        catch (_) { /* private mode / storage full — non-fatal */ }
    }
    function forgetActiveJob() {
        try { localStorage.removeItem(PODCAST_ACTIVE_JOB_STORAGE_KEY); }
        catch (_) { /* non-fatal */ }
    }
    function readActiveJob() {
        try { return localStorage.getItem(PODCAST_ACTIVE_JOB_STORAGE_KEY); }
        catch (_) { return null; }
    }

    // F-4.3 P9: live char-counter on the source-text textarea. Greys at
    // ≤80%, warns at 80–100%, errors past 100%.
    function updateRawTextCounter() {
        if (!podcastRawText || !podcastRawTextCounter) return;
        const len = podcastRawText.value.length;
        podcastRawTextCounter.textContent = len + '/' + maxRawTextChars + ' Zeichen';
        podcastRawTextCounter.classList.remove('is-warning', 'is-over');
        if (len > maxRawTextChars) {
            podcastRawTextCounter.classList.add('is-over');
        } else if (len > maxRawTextChars * 0.8) {
            podcastRawTextCounter.classList.add('is-warning');
        }
    }
    if (podcastRawText) {
        podcastRawText.addEventListener('input', updateRawTextCounter);
        updateRawTextCounter();
    }

    // F-4.3 P10: pre-parse the script on blur and warn in-place if no
    // dialogue lines were recognised (most common reason: no `:` separator).
    function preParseScriptForHint() {
        if (!podcastScript) return;
        const text = podcastScript.value;
        if (!text.trim()) return;
        let dialogueLines = 0;
        let firstBadLine = null;
        const lines = text.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('**')) continue;
            if (trimmed.includes(':')) {
                dialogueLines++;
            } else if (firstBadLine === null) {
                firstBadLine = i + 1;
            }
        }
        if (dialogueLines === 0) {
            const msg = firstBadLine !== null
                ? 'Skript konnte nicht gelesen werden — Zeile ' + firstBadLine
                  + ': kein Doppelpunkt-Trenner gefunden.'
                : 'Skript konnte nicht gelesen werden. Format prüfen: Sprecher [stil]: Text.';
            showAlert(podcastAlertContainer, 'warning', msg, { autoDismissMs: 8000 });
        }
    }
    if (podcastScript) {
        podcastScript.addEventListener('blur', preParseScriptForHint);
    }

    function setPodcastGenerating(isGenerating) {
        if (isGenerating) {
            if (generatePodcastBtn) generatePodcastBtn.classList.add('hidden');
            if (cancelPodcastBtn) {
                cancelPodcastBtn.classList.remove('hidden');
                resetPodcastCancelBtn();
            }
            // F-4.3 P11: freeze script during generation (worker has its own snapshot).
            setPodcastScriptReadonly(true);
        } else {
            if (cancelPodcastBtn) cancelPodcastBtn.classList.add('hidden');
            if (generatePodcastBtn) {
                generatePodcastBtn.classList.remove('hidden');
                generatePodcastBtn.disabled = !geminiAvailable;
                setBtnLoading(generatePodcastBtn, false);
            }
            resetPodcastCancelBtn();
            setPodcastScriptReadonly(false);
            hidePodcastStage();
        }
    }

    function resetPodcastCancelBtn() {
        if (!cancelPodcastBtn) return;
        if (podcastCancelConfirmTimer) {
            clearTimeout(podcastCancelConfirmTimer);
            podcastCancelConfirmTimer = null;
        }
        podcastCancelState = 'idle';
        cancelPodcastBtn.disabled = false;
        cancelPodcastBtn.textContent = CANCEL_BTN_LABEL_IDLE;
        cancelPodcastBtn.removeAttribute('aria-pressed');
    }

    if (cancelPodcastBtn) {
        cancelPodcastBtn.addEventListener('click', async () => {
            if (podcastCancelState === 'cancelling') return;

            if (podcastCancelState === 'idle') {
                podcastCancelState = 'confirm-pending';
                cancelPodcastBtn.textContent = CANCEL_BTN_LABEL_CONFIRM;
                cancelPodcastBtn.setAttribute('aria-pressed', 'true');
                showAlert(podcastAlertContainer, 'warning',
                    'Generierung wirklich abbrechen? TTS-Token sind teilweise schon verbraucht.',
                    { autoDismissMs: CANCEL_CONFIRM_TIMEOUT_MS });
                podcastCancelConfirmTimer = setTimeout(() => {
                    if (podcastCancelState === 'confirm-pending') resetPodcastCancelBtn();
                }, CANCEL_CONFIRM_TIMEOUT_MS);
                return;
            }

            // confirm-pending → cancelling
            if (podcastCancelConfirmTimer) {
                clearTimeout(podcastCancelConfirmTimer);
                podcastCancelConfirmTimer = null;
            }
            podcastCancelState = 'cancelling';
            cancelPodcastBtn.textContent = CANCEL_BTN_LABEL_CANCELLING;
            cancelPodcastBtn.disabled = true;

            if (!podcastJobId) {
                // Nothing in flight (race with finish). Just reset UI.
                resetPodcastCancelBtn();
                return;
            }

            try {
                const r = await fetch(`/podcast-cancel/${podcastJobId}`, { method: 'POST' });
                if (!r.ok && r.status !== 202) {
                    const errData = await safeJSON(r);
                    throw new Error(errData.error || `Cancel fehlgeschlagen (${r.status})`);
                }
            } catch (err) {
                console.error('Cancel error:', err);
                showAlert(podcastAlertContainer, 'danger',
                    'Abbruch konnte nicht ausgeführt werden. Bitte erneut versuchen.');
                resetPodcastCancelBtn();
            }
            // Polling-Loop reads worker-confirmed terminal status and finalises UI.
        });
    }

    if (downloadPodcastBtn) {
        downloadPodcastBtn.addEventListener('click', () => {
            if (!downloadPodcastBtn.getAttribute('href')) return;
            // F-4.3 P6: replace the always-green toast (which lied when the
            // browser silently blocked the download) with an info-banner that
            // names the most likely block cause.
            showAlert(podcastAlertContainer, 'info',
                'Download gestartet. Falls nichts passiert, Pop-Up-Blocker prüfen.',
                { autoDismissMs: 5000 });
        });
    }

    // F-4.3 BT4: revoke the audio blob URL on page-hide so the browser does
    // not retain WAV bytes after the user leaves the tab. The active URL is
    // also revoked at the start of every fresh generation (see generate-btn).
    window.addEventListener('pagehide', () => {
        if (lastPodcastBlobUrl) {
            try { URL.revokeObjectURL(lastPodcastBlobUrl); } catch (_) { /* noop */ }
            lastPodcastBlobUrl = null;
        }
    });

    const podcastAudioError = document.getElementById('podcast-audio-error');
    if (podcastAudio) {
        podcastAudio.addEventListener('error', () => {
            if (!podcastAudioError) return;
            showAlert(podcastAudioError, 'info',
                'Audio nicht mehr verfügbar — bitte erneut generieren.',
                { closable: false, autoDismissMs: null });
            podcastAudioError.classList.remove('hidden');
        });
        podcastAudio.addEventListener('loadeddata', () => {
            if (!podcastAudioError) return;
            podcastAudioError.innerHTML = '';
            podcastAudioError.classList.add('hidden');
        });
    }

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
                promptEditorToggle.setAttribute('aria-expanded', 'false');
            } else {
                promptEditorContent.classList.add('expanded');
                promptToggleIcon.innerHTML = '&#9650;';
                promptEditorToggle.setAttribute('aria-expanded', 'true');
            }
        });
    }

    if (resetPromptBtn) {
        resetPromptBtn.addEventListener('click', function() {
            if (!confirmIfLong(customPromptEditor.value,
                'Eigenen Prompt wirklich auf Standard zurücksetzen? Dein Text geht verloren.')) {
                return;
            }
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
            // F-4.3 P7: spin the existing icon (F16) + add aria-busy via the
            // same .is-loading toggle now used for both generate buttons.
            setBtnLoading(generateScriptBtn, true);

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
                setBtnLoading(generateScriptBtn, false);
            }
        });
    }

    // F-4.3 P8: defensive polling-loop. Awaits the next status snapshot; on
    // 5xx tries up to POLLING_RETRY_MAX before surfacing a banner; on 404
    // (job-result-TTL expired) surfaces a dedicated banner; on unknown
    // status surfaces a warning. Returns a status-object to the caller.
    async function pollPodcastStatus(jobId) {
        const startTs = Date.now();
        let consecutive5xx = 0;
        let queueStuckBannerShown = false;

        while (true) {
            await new Promise(resolve => setTimeout(resolve, POLLING_INTERVAL_MS));

            // Hard ceiling — surface a banner so the user has an exit ramp.
            if (Date.now() - startTs > POLLING_TIMEOUT_MS) {
                showAlert(podcastAlertContainer, 'info',
                    'Generierung dauert länger als erwartet — Worker-Logs prüfen.',
                    { autoDismissMs: null });
                return { kind: 'timeout' };
            }

            let response;
            try {
                response = await fetch(`/podcast-status/${jobId}`);
            } catch (networkErr) {
                consecutive5xx++;
                if (consecutive5xx > POLLING_RETRY_MAX) {
                    showAlert(podcastAlertContainer, 'danger',
                        'Status-Server reagiert nicht. Verbindung prüfen oder Container neu starten.');
                    return { kind: 'network_error', error: networkErr };
                }
                continue;
            }

            // 404 → job-result TTL expired (or never existed). Distinct from
            // a transport error.
            if (response.status === 404) {
                showAlert(podcastAlertContainer, 'danger',
                    'Job-Status nicht mehr verfügbar. Bitte neu generieren.');
                return { kind: 'gone' };
            }
            if (response.status >= 500) {
                consecutive5xx++;
                if (consecutive5xx > POLLING_RETRY_MAX) {
                    showAlert(podcastAlertContainer, 'danger',
                        'Status-Server reagiert nicht. Verbindung prüfen oder Container neu starten.');
                    return { kind: 'server_error' };
                }
                continue;
            }
            if (!response.ok) {
                showAlert(podcastAlertContainer, 'danger',
                    `Status-Abruf fehlgeschlagen (HTTP ${response.status}).`);
                return { kind: 'http_error' };
            }
            consecutive5xx = 0;

            const data = await safeJSON(response);
            const status = data.status;

            if (!POLLING_KNOWN_STATES.has(status)) {
                showAlert(podcastAlertContainer, 'warning',
                    'Unbekannter Job-Status — bitte erneut versuchen.');
                return { kind: 'unknown_status', status };
            }

            // F-4.3 P5: differentiated visibility for queued vs. started.
            if (status === 'queued') {
                setPodcastStageQueueWaiting();
                if (!queueStuckBannerShown && Date.now() - startTs > QUEUE_STUCK_MS) {
                    queueStuckBannerShown = true;
                    showAlert(podcastAlertContainer, 'info',
                        'Worker reagiert nicht. Container-Status prüfen: docker ps.');
                }
                continue;
            }
            // F-4.3 P8: explicit deferred/scheduled handling — user sees that
            // RQ is waiting on a condition, not stuck.
            if (status === 'deferred' || status === 'scheduled') {
                setPodcastStageQueueWaiting();
                continue;
            }
            if (status === 'started' || status === 'processing') {
                setPodcastStage(data);
                continue;
            }
            // Terminal states.
            if (status === 'completed' || status === 'finished') {
                return { kind: 'completed', data };
            }
            if (status === 'failed') {
                return { kind: 'failed', error: data.error || 'Generierung fehlgeschlagen' };
            }
            if (status === 'cancelled' || status === 'canceled' || status === 'stopped') {
                return { kind: 'cancelled' };
            }
        }
    }

    // F-4.3 P3 / P4 / P5 / P6 / P8: shared post-status pipeline used by both
    // the fresh-start path and the browser-reload re-attach path.
    async function runPodcastJobUntilDone(jobId, { onCompletedDownload = true } = {}) {
        podcastJobId = jobId;
        rememberActiveJob(jobId);
        setPodcastGenerating(true);
        podcastResultContainer.classList.add('hidden');

        try {
            const result = await pollPodcastStatus(jobId);

            if (result.kind === 'cancelled') {
                showAlert(podcastAlertContainer, 'warning', 'Generierung abgebrochen.');
                return;
            }
            if (result.kind === 'failed') {
                showAlert(podcastAlertContainer, 'danger',
                    'Podcast-Generierung fehlgeschlagen. Bitte erneut versuchen.');
                return;
            }
            if (result.kind !== 'completed') {
                // gone / timeout / network_error / server_error / unknown — banner
                // already shown by the poller.
                return;
            }
            if (!onCompletedDownload) {
                // Re-attach found a finished job — surface the result without
                // an extra banner; the user wants the file.
            }

            const downloadResponse = await fetch(`/podcast-download/${jobId}`);
            if (!downloadResponse.ok) {
                if (downloadResponse.status === 404) {
                    showAlert(podcastAlertContainer, 'danger',
                        'Podcast-Datei nicht mehr verfügbar — bitte erneut generieren.');
                } else {
                    showAlert(podcastAlertContainer, 'danger',
                        'Podcast-Download fehlgeschlagen.');
                }
                return;
            }

            const audioBlob = await downloadResponse.blob();
            // F-4.3 BT4: revoke the previous URL before grabbing a new one
            // so successive generations do not retain the old WAV bytes.
            if (lastPodcastBlobUrl) {
                try { URL.revokeObjectURL(lastPodcastBlobUrl); } catch (_) { /* noop */ }
            }
            const audioUrl = URL.createObjectURL(audioBlob);
            lastPodcastBlobUrl = audioUrl;

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
            setPodcastGenerating(false);
            podcastJobId = null;
            forgetActiveJob();
        }
    }

    if (generatePodcastBtn) {
        generatePodcastBtn.addEventListener('click', async () => {
            const scriptText = podcastScript.value.trim();

            if (!scriptText) {
                showAlert(podcastAlertContainer, 'warning',
                    'Bitte erst ein Skript eintragen oder generieren lassen.');
                return;
            }

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
                await runPodcastJobUntilDone(job_id);
            } catch (error) {
                console.error('Podcast enqueue error:', error);
                showAlert(podcastAlertContainer, 'danger',
                    'Podcast-Generierung konnte nicht gestartet werden. Bitte erneut versuchen.');
                setPodcastGenerating(false);
                podcastJobId = null;
                forgetActiveJob();
            }
        });
    }

    // F-4.3 P4: Browser-reload-recovery. If the previous tab was polling an
    // active job, the job_id is in localStorage. Probe /podcast-status once
    // and either re-attach the polling loop or show a final-state banner.
    (async function reattachActiveJobOnLoad() {
        const jobId = readActiveJob();
        if (!jobId) return;
        let probe;
        try {
            probe = await fetch(`/podcast-status/${jobId}`);
        } catch (_) {
            forgetActiveJob();
            return;
        }
        if (!probe.ok) {
            forgetActiveJob();
            return;
        }
        let data;
        try { data = await safeJSON(probe); }
        catch (_) { forgetActiveJob(); return; }

        const status = data.status;
        if (status === 'queued' || status === 'started' || status === 'processing'
            || status === 'deferred' || status === 'scheduled') {
            showAlert(podcastAlertContainer, 'info',
                'Laufende Generierung wiederhergestellt.', { autoDismissMs: 4000 });
            // Render the in-flight state immediately (don't wait the 2s tick).
            setPodcastGenerating(true);
            if (status === 'queued' || status === 'deferred' || status === 'scheduled') {
                setPodcastStageQueueWaiting();
            } else if (data.stage) {
                setPodcastStage(data);
            }
            runPodcastJobUntilDone(jobId);
        } else if (status === 'failed') {
            showAlert(podcastAlertContainer, 'danger',
                'Vorherige Generierung ist fehlgeschlagen. Bitte neu starten.');
            forgetActiveJob();
        } else if (status === 'cancelled' || status === 'canceled' || status === 'stopped') {
            showAlert(podcastAlertContainer, 'warning',
                'Vorherige Generierung wurde abgebrochen.');
            forgetActiveJob();
        } else if (status === 'completed' || status === 'finished') {
            showAlert(podcastAlertContainer, 'info',
                'Vorherige Generierung ist fertig — Datei zum Download bereit.',
                { autoDismissMs: 4000 });
            forgetActiveJob();
            // The result file may already be cleaned up by the prior
            // download — don't auto-fetch; the user can hit "Erneut
            // generieren". (Simpler than chasing the TTL race.)
        } else {
            forgetActiveJob();
        }
    })();

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
