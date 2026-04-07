/**
 * Face Age Predictor — Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ───────────────────────────────
    const uploadZone = document.getElementById('upload-zone');
    const uploadContent = document.getElementById('upload-content');
    const fileInput = document.getElementById('file-input');
    const previewWrapper = document.getElementById('preview-wrapper');
    const previewImage = document.getElementById('preview-image');
    const btnRemove = document.getElementById('btn-remove');
    const btnPredict = document.getElementById('btn-predict');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    const resultsBadge = document.getElementById('results-badge');
    const noFaceMessage = document.getElementById('no-face-message');
    const errorToast = document.getElementById('error-toast');
    const errorText = document.getElementById('error-text');

    let selectedFile = null;
    let errorTimeout = null;

    // ── Upload Zone: Click ─────────────────────────
    uploadZone.addEventListener('click', (e) => {
        if (selectedFile && e.target !== btnRemove) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // ── Upload Zone: Drag & Drop ──────────────────
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragging');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragging');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragging');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ── Remove Image ──────────────────────────────
    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // ── Predict Button ────────────────────────────
    btnPredict.addEventListener('click', () => {
        if (!selectedFile) return;
        predictAge();
    });

    // ── Handle File Selection ─────────────────────
    function handleFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            showError('지원하지 않는 이미지 형식입니다. JPEG, PNG, WebP, BMP만 가능합니다.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError('파일 크기가 10MB를 초과합니다.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewWrapper.style.display = 'block';
            uploadZone.classList.add('has-image');
            btnPredict.disabled = false;

            // Hide previous results
            resultsSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // ── Reset Upload ──────────────────────────────
    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewWrapper.style.display = 'none';
        uploadContent.style.display = 'flex';
        uploadZone.classList.remove('has-image');
        btnPredict.disabled = true;
        resultsSection.style.display = 'none';
    }

    // ── API Call ───────────────────────────────────
    async function predictAge() {
        if (!selectedFile) return;

        // Show loading state
        btnPredict.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'flex';
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || '서버 오류가 발생했습니다.');
            }

            renderResults(data);
        } catch (err) {
            showError(err.message || '예측 중 오류가 발생했습니다.');
        } finally {
            // Reset button state
            btnPredict.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
        }
    }

    // ── Render Results ────────────────────────────
    function renderResults(response) {
        const { data } = response;
        const { face_count, faces } = data;

        resultsSection.style.display = 'block';
        resultsGrid.innerHTML = '';

        if (face_count === 0) {
            noFaceMessage.style.display = 'block';
            resultsBadge.textContent = '0개 검출';
            return;
        }

        noFaceMessage.style.display = 'none';
        resultsBadge.textContent = `${face_count}개 검출`;

        faces.forEach((face, index) => {
            const card = createFaceCard(face, index);
            resultsGrid.appendChild(card);
        });

        // Smooth scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function createFaceCard(face, index) {
        const card = document.createElement('div');
        card.className = 'face-card';
        card.style.animationDelay = `${index * 0.1}s`;

        const ageConfPercent = Math.round(face.age_confidence * 100);
        const detConfPercent = Math.round(face.detection_confidence * 100);

        card.innerHTML = `
            <div class="face-card-header">
                <div class="face-label">
                    <span class="face-number">${face.face_id}</span>
                    얼굴 #${face.face_id}
                </div>
                <span class="face-confidence-badge">검출 ${detConfPercent}%</span>
            </div>
            <div class="age-display">
                <div class="age-value">${face.age_range}</div>
                <div class="age-label">예측 나이 구간</div>
            </div>
            <div class="confidence-section">
                <div class="confidence-row">
                    <span>나이 예측 신뢰도</span>
                    <span>${ageConfPercent}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: 0%;" data-target="${ageConfPercent}"></div>
                </div>
            </div>
        `;

        // Animate confidence bar after card is rendered
        requestAnimationFrame(() => {
            setTimeout(() => {
                const bar = card.querySelector('.confidence-bar-fill');
                bar.style.width = bar.dataset.target + '%';
            }, 200 + index * 100);
        });

        return card;
    }

    // ── Error Toast ───────────────────────────────
    function showError(message) {
        errorText.textContent = message;
        errorToast.style.display = 'flex';

        if (errorTimeout) clearTimeout(errorTimeout);
        errorTimeout = setTimeout(() => {
            errorToast.style.display = 'none';
        }, 5000);
    }
});
