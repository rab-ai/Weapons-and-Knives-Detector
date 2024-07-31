document.addEventListener('DOMContentLoaded', () => {
    const imageForm = document.getElementById('image-form');
    const videoForm = document.getElementById('video-form');
    const startWebcamBtn = document.getElementById('start-webcam-btn');
    const webcamConfThreshold = document.getElementById('webcam_conf_threshold');
    const sliderValue = document.getElementById('sliderValue');

    if (imageForm) {
        imageForm.addEventListener('submit', () => {
            const confThreshold = document.querySelector('input[name="image_conf_threshold"]').value;
            localStorage.setItem('image_conf_threshold', confThreshold);
        });
    }

    if (videoForm) {
        videoForm.addEventListener('submit', () => {
            const confThreshold = document.querySelector('input[name="video_conf_threshold"]').value;
            localStorage.setItem('video_conf_threshold', confThreshold);
        });
    }

    if (webcamConfThreshold) {
        webcamConfThreshold.addEventListener('input', (event) => {
            if (sliderValue) {
                sliderValue.textContent = event.target.value;
            }
        });
    }

    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', () => {
            const confThreshold = webcamConfThreshold.value;
            localStorage.setItem('webcam_conf_threshold', confThreshold);
            window.location.href = `/start_webcam?conf_threshold=${confThreshold}`;
        });
    }
});
