document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const recordBtn = document.getElementById('record-btn');
    const btnText = recordBtn.querySelector('.btn-text');
    const recordingOverlay = document.getElementById('recording-overlay');
    const transcriptionText = document.getElementById('transcription-text');
    const processingStatus = document.getElementById('processing-status');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let stream;

    // Silence detection variables
    let audioContext;
    let analyser;
    let silenceStart = null;
    let animationId = null;
    const SILENCE_THRESHOLD = 0.015; // Volume threshold for silence
    const SILENCE_LIMIT = 2000;      // 2 seconds in ms

    // Initialize Webcam
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            video.srcObject = stream;
            document.getElementById('camera-status').style.backgroundColor = '#10b981';
            document.getElementById('camera-status').style.boxShadow = '0 0 10px #10b981';

            // Setup Web Audio API for silence detection
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

        } catch (err) {
            console.error("Error accessing webcam/microphone:", err);
            transcriptionText.innerHTML = `<p style="color: #ef4444">Error: Could not access camera or microphone. Please ensure you have granted permissions.</p>`;
            document.getElementById('camera-status').style.backgroundColor = '#ef4444';
            document.getElementById('camera-status').style.boxShadow = '0 0 10px #ef4444';
        }
    }

    initCamera();

    // Handle Recording Logic
    recordBtn.addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    function startRecording() {
        audioChunks = [];
        const audioStream = new MediaStream(stream.getAudioTracks());
        mediaRecorder = new MediaRecorder(audioStream);

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = sendAudioData;

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        btnText.textContent = 'Stop Recording';
        recordingOverlay.style.opacity = '1';

        // Start monitoring for silence
        silenceStart = null;
        monitorSilence();
    }

    function monitorSilence() {
        if (!isRecording) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Float32Array(bufferLength);
        analyser.getFloatTimeDomainData(dataArray);

        // Calculate RMS (Volume)
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / bufferLength);

        if (rms < SILENCE_THRESHOLD) {
            if (silenceStart === null) {
                silenceStart = Date.now();
            } else if (Date.now() - silenceStart > SILENCE_LIMIT) {
                console.log("Auto-stopping due to silence...");
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
        recordingOverlay.style.opacity = '0';

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
                // If it's the first transcription, clear placeholder
                if (transcriptionText.querySelector('.placeholder')) {
                    transcriptionText.innerHTML = '';
                }

                const p = document.createElement('p');
                p.textContent = data.transcription;
                p.className = 'fade-in';
                transcriptionText.prepend(p);
            } else if (data.error) {
                console.error("API Error:", data.error);
            }
        } catch (err) {
            console.error("Fetch Error:", err);
        } finally {
            processingStatus.style.opacity = '0';
        }
    }
});

// Add a simple fade-in animation via script for new transcriptions
const style = document.createElement('style');
style.textContent = `
    .fade-in {
        animation: fadeIn 0.5s ease forwards;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.5rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
