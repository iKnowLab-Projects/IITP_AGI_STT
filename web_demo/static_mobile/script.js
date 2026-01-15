document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const btnText = recordBtn.querySelector('.btn-text');
    const visualizer = document.getElementById('visualizer');
    const transcriptionText = document.getElementById('transcription-text');
    const processingStatus = document.getElementById('processing-status');
    const micStatus = document.getElementById('mic-status');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let stream;

    // Silence detection variables
    let audioContext;
    let analyser;
    let silenceStart = null;
    let animationId = null;
    const SILENCE_THRESHOLD = 0.015;
    const SILENCE_LIMIT = 2000;

    // Initialize Microphone
    async function initMic() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: true
            });
            micStatus.classList.add('ready');

            // Setup Web Audio API for silence detection
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

        } catch (err) {
            console.error("Error accessing microphone:", err);
            transcriptionText.innerHTML = `<p style="color: #ef4444">Error: Could not access microphone. Please ensure you have granted permissions and are using HTTPS.</p>`;
        }
    }

    initMic();

    // Handle Recording Logic
    recordBtn.addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    function startRecording() {
        if (!stream) {
            initMic();
            return;
        }

        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = sendAudioData;

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        btnText.textContent = 'Recording...';
        visualizer.classList.add('active');

        // Start monitoring for silence
        silenceStart = null;
        monitorSilence();
    }

    function monitorSilence() {
        if (!isRecording || !analyser) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Float32Array(bufferLength);
        analyser.getFloatTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / bufferLength);

        if (rms < SILENCE_THRESHOLD) {
            if (silenceStart === null) {
                silenceStart = Date.now();
            } else if (Date.now() - silenceStart > SILENCE_LIMIT) {
                stopRecording();
                return;
            }
        } else {
            silenceStart = null;
        }

        animationId = requestAnimationFrame(monitorSilence);
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        recordBtn.classList.remove('recording');
        btnText.textContent = 'Start Recording';
        visualizer.classList.remove('active');

        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }

    async function sendAudioData() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        processingStatus.style.opacity = '1';

        try {
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.transcription) {
                if (transcriptionText.querySelector('.placeholder')) {
                    transcriptionText.innerHTML = '';
                }

                const p = document.createElement('p');
                p.textContent = data.transcription;
                p.className = 'fade-in';
                transcriptionText.prepend(p);

                // Haptic feedback if available (Mobile support)
                if (window.navigator.vibrate) {
                    window.navigator.vibrate(50);
                }
            }
        } catch (err) {
            console.error("Fetch Error:", err);
        } finally {
            processingStatus.style.opacity = '0';
        }
    }
});
