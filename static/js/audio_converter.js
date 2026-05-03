/* Audio converter page: live transcription, file transcription, podcast generator. */
document.addEventListener('DOMContentLoaded', function() {

    // ==========================================================
    // ===== CUSTOM TAB SWITCHING =====
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
            languageButtons.forEach(btn => {
                btn.classList.remove('lang-active');
                btn.classList.add('lang-inactive');
            });
            this.classList.add('lang-active');
            this.classList.remove('lang-inactive');
            selectedLanguage = this.dataset.lang;
        });
    });

    // ==========================================================
    // ===== LIVE TRANSCRIPTION LOGIC =====
    // ==========================================================
    const micButton = document.getElementById('mic-button');
    const transcriptOutput = document.getElementById('live-transcript-output');

    let mediaRecorder;
    let socket;
    let isRecording = false;
    let baseText = '';
    let keepAliveInterval;

    async function getDeepgramToken() {
        try {
            const response = await fetch('/api/get-deepgram-token');
            if (!response.ok) {
                const errorData = await safeJSON(response);
                throw new Error(errorData.error || 'Failed to get token');
            }
            const data = await safeJSON(response);
            return data.deepgram_token;
        } catch (error) {
            console.error("Failed to get token:", error);
            alert("Error getting API token. Check server configuration.");
            throw error;
        }
    }

    const connectToDeepgram = async () => {
        let deepgramToken;
        try {
            deepgramToken = await getDeepgramToken();
        } catch (error) {
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

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000
                }
            });

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
            micButton.classList.add('recording');
            micButton.title = "Click to Stop Recording";

            languageButtons.forEach(btn => btn.disabled = true);

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

        socket.onclose = (event) => {
            if (isRecording) stopRecording(false);
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            alert("Connection failed. Check your network and API configuration.");
        };
    };

    const stopRecording = (shouldCloseSocket = true) => {
        if (!isRecording && shouldCloseSocket) return;

        clearInterval(keepAliveInterval);

        if (mediaRecorder) {
            if (mediaRecorder.processor) {
                mediaRecorder.processor.disconnect();
                mediaRecorder.processor.onaudioprocess = null;
            }
            if (mediaRecorder.source) {
                mediaRecorder.source.disconnect();
            }
            if (mediaRecorder.audioContext) {
                mediaRecorder.audioContext.close();
            }
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }

        if (socket && socket.readyState === WebSocket.OPEN && shouldCloseSocket) {
            socket.close();
        }

        mediaRecorder = null;
        if (shouldCloseSocket) socket = null;
        isRecording = false;

        micButton.classList.remove('recording');
        micButton.title = "Click to Start Recording";

        languageButtons.forEach(btn => btn.disabled = false);

        transcriptOutput.focus();

        const saveLiveBtn = document.getElementById('save-live-btn');
        if (saveLiveBtn && transcriptOutput.value.trim()) {
            saveLiveBtn.classList.remove('hidden');
            saveLiveBtn.textContent = 'Save to Library';
            saveLiveBtn.disabled = false;
            saveLiveBtn.classList.remove('saved');
        }
    };

    micButton.addEventListener('click', () => {
        if (isRecording) stopRecording();
        else connectToDeepgram();
    });

    // ==========================================================
    // ===== FILE UPLOAD TRANSCRIPTION LOGIC =====
    // ==========================================================
    const uploadForm = document.getElementById('audio-upload-form');
    const fileInput = document.getElementById('file-upload-input');
    const fileUploadText = document.getElementById('file-upload-text');
    const resultContainer = document.getElementById('transcription-result-container');
    const transcribeBtn = document.getElementById('transcribe-file-btn');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileUploadText.textContent = fileInput.files[0].name;
        } else {
            fileUploadText.textContent = 'Click to select an audio file or drag and drop';
        }
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        if (!fileInput.files[0]) {
            alert("Please select an audio file first.");
            return;
        }

        const formData = new FormData();
        formData.append('audio_file', fileInput.files[0]);
        formData.append('language', selectedLanguage);

        transcribeBtn.disabled = true;
        transcribeBtn.textContent = 'Transcribing...';
        resultContainer.innerHTML = '<span class="text-neo-faint">Processing your audio file...</span>';

        try {
            const response = await fetch('/transcribe-audio-file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const result = await safeJSON(response);
                throw new Error(result.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await safeJSON(response);

            resultContainer.textContent = result.transcript || 'No transcript was returned.';

            const saveBtn = document.getElementById('save-transcription-btn');
            saveBtn.classList.remove('hidden');
            saveBtn.textContent = 'Save to Library';
            saveBtn.disabled = false;
            saveBtn._transcriptionData = {
                content: result.transcript,
                filename: fileInput.files[0].name,
                mimetype: fileInput.files[0].type,
                size: fileInput.files[0].size,
                metadata: result.metadata || {}
            };

        } catch (error) {
            console.error('Transcription error:', error);
            resultContainer.innerHTML = `<span style="color: var(--nm-danger)"><strong>Error:</strong> ${error.message}</span>`;
        } finally {
            transcribeBtn.disabled = false;
            transcribeBtn.textContent = 'Transcribe File';
        }
    });

    // ==========================================================
    // ===== COPY & CLEAR BUTTONS =====
    // ==========================================================
    const clearLiveBtn = document.getElementById('clear-live-btn');
    const clearFileBtn = document.getElementById('clear-file-btn');
    const copyLiveBtn = document.getElementById('copy-live-btn');
    const copyFileBtn = document.getElementById('copy-file-btn');

    clearLiveBtn.addEventListener('click', () => {
        transcriptOutput.value = '';
        baseText = '';
    });

    clearFileBtn.addEventListener('click', () => {
        resultContainer.innerHTML = '<span class="text-neo-faint">The transcription will appear here after processing.</span>';
    });

    async function copyToClipboard(text, button) {
        if (!text || text.trim() === '') {
            alert('Nothing to copy!');
            return;
        }

        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                if (!document.execCommand('copy')) throw new Error('execCommand failed');
                document.body.removeChild(ta);
            }
            const originalText = button.innerHTML;
            button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';

            setTimeout(() => {
                button.innerHTML = originalText;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            alert('Failed to copy to clipboard');
        }
    }

    copyLiveBtn.addEventListener('click', () => {
        copyToClipboard(transcriptOutput.value, copyLiveBtn);
    });

    copyFileBtn.addEventListener('click', () => {
        const text = resultContainer.textContent;
        copyToClipboard(text, copyFileBtn);
    });

    // ==========================================================
    // ===== PODCAST GENERATOR =====
    // ==========================================================

    const podcastModeRadios = document.querySelectorAll('input[name="podcast-mode"]');
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

    // Prompt editor toggle
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

    // Reset prompt
    resetPromptBtn.addEventListener('click', function() {
        customPromptEditor.value = '';
    });

    // Generate script from raw text
    generateScriptBtn.addEventListener('click', async () => {
        const rawText = podcastRawText.value.trim();

        if (!rawText) {
            alert('Please enter some text in the Source Text field.');
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

        generateScriptBtn.disabled = true;
        generateScriptBtn.textContent = 'Generating...';

        try {
            const customPrompt = customPromptEditor.value.trim() || null;

            const response = await fetch('/format-dialogue-with-llm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
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
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await safeJSON(response);
            podcastScript.value = data.raw_formatted_text;

        } catch (error) {
            console.error('Script generation error:', error);
            alert('Error generating script: ' + error.message);
        } finally {
            generateScriptBtn.disabled = false;
            generateScriptBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path></svg> Generate Script from Text Above';
        }
    });

    // Generate podcast from script
    generatePodcastBtn.addEventListener('click', async () => {
        const scriptText = podcastScript.value.trim();

        if (!scriptText) {
            alert('Please enter or generate a script first.');
            return;
        }

        const mode = document.querySelector('input[name="podcast-mode"]:checked').value;

        const dialogue = [];
        const voiceMap = {
            'Kate': 'Zephyr',
            'Max': 'Charon'
        };

        for (const line of scriptText.split('\n')) {
            const trimmedLine = line.trim();
            if (!trimmedLine || trimmedLine.startsWith('#') || trimmedLine.startsWith('**')) {
                continue;
            }

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
                    dialogue.push({
                        speaker: voice,
                        style: style,
                        text: textPart
                    });
                }
            }
        }

        if (dialogue.length === 0) {
            alert('Could not parse the script. Please check the format:\nSpeakerName [style]: Text');
            return;
        }

        generatePodcastBtn.disabled = true;
        generatePodcastBtn.textContent = 'Starting...';
        podcastResultContainer.classList.add('hidden');

        try {
            const ttsModelSelect = document.getElementById('tts-model');
            const startResponse = await fetch('/generate-gemini-podcast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dialogue: dialogue,
                    language: podcastLanguageSelect.value,
                    tts_model: ttsModelSelect.value
                })
            });

            if (!startResponse.ok) {
                const errorData = await safeJSON(startResponse);
                throw new Error(errorData.error || `HTTP error! Status: ${startResponse.status}`);
            }

            const { job_id } = await safeJSON(startResponse);
            generatePodcastBtn.textContent = 'Generating...';

            let status = 'pending';
            let pollCount = 0;
            while (status === 'pending' || status === 'processing') {
                await new Promise(resolve => setTimeout(resolve, 2000));
                pollCount++;
                generatePodcastBtn.textContent = `Generating... (${pollCount * 2}s)`;

                const statusResponse = await fetch(`/podcast-status/${job_id}`);
                const statusData = await safeJSON(statusResponse);
                status = statusData.status;

                if (status === 'failed') {
                    throw new Error(statusData.error || 'Generation failed');
                }
            }

            const downloadResponse = await fetch(`/podcast-download/${job_id}`);
            if (!downloadResponse.ok) {
                throw new Error('Failed to download podcast');
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
            alert('Error generating podcast: ' + error.message);
        } finally {
            generatePodcastBtn.disabled = false;
            generatePodcastBtn.textContent = 'Generate Podcast';
        }
    });

    // ==========================================================
    // ===== SAVE TO LIBRARY =====
    // ==========================================================
    document.getElementById('save-transcription-btn').addEventListener('click', async function() {
        const btn = this;
        const data = btn._transcriptionData;
        if (!data) return;

        btn.disabled = true;
        btn.textContent = 'Saving...';

        try {
            const stem = data.filename.replace(/\.[^.]+$/, '');
            const response = await fetch('/api/conversions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
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
                btn.textContent = 'Saved!';
                btn.classList.add('saved');
            } else {
                throw new Error('Save failed');
            }
        } catch (err) {
            btn.textContent = 'Save to Library';
            btn.disabled = false;
            alert('Failed to save: ' + err.message);
        }
    });

    // ===== SAVE LIVE TRANSCRIPTION TO LIBRARY =====
    const saveLiveBtn = document.getElementById('save-live-btn');
    if (saveLiveBtn) {
        saveLiveBtn.addEventListener('click', async function() {
            const btn = this;
            const content = document.getElementById('live-transcript-output').value.trim();
            if (!content) return;

            btn.disabled = true;
            btn.textContent = 'Saving...';

            try {
                const now = new Date();
                const title = `Live Transcription ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`;
                const response = await fetch('/api/conversions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        conversion_type: 'audio_transcription',
                        title: title,
                        content: content,
                        source_filename: 'live-recording',
                        metadata: {source: 'live_transcription'}
                    })
                });
                if (response.ok) {
                    btn.textContent = 'Saved!';
                    btn.classList.add('saved');
                } else {
                    throw new Error('Save failed');
                }
            } catch (err) {
                btn.textContent = 'Save to Library';
                btn.disabled = false;
                alert('Failed to save: ' + err.message);
            }
        });
    }

});
