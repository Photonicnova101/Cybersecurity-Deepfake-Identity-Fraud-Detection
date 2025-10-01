// Global variables
let selectedFile = null;
let fileType = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const imagePreview = document.getElementById('imagePreview');
const videoPreview = document.getElementById('videoPreview');
const audioPreview = document.getElementById('audioPreview');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // Click to browse
    uploadArea.addEventListener('click', () => fileInput.click());
    document.querySelector('.browse-link').addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    // File selection
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    
    if (!file) return;
    
    // Validate file size (100MB)
    if (file.size > 100 * 1024 * 1024) {
        alert('File size exceeds 100MB limit');
        return;
    }
    
    selectedFile = file;
    
    // Determine file type
    if (file.type.startsWith('image/')) {
        fileType = 'image';
        showImagePreview(file);
    } else if (file.type.startsWith('video/')) {
        fileType = 'video';
        showVideoPreview(file);
    } else if (file.type.startsWith('audio/')) {
        fileType = 'audio';
        showAudioPreview(file);
    } else {
        alert('Unsupported file type');
        return;
    }
    
    // Show preview and analyze button
    uploadArea.style.display = 'none';
    filePreview.style.display = 'block';
    analyzeBtn.style.display = 'block';
    results.style.display = 'none';
}

function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        videoPreview.style.display = 'none';
        audioPreview.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function showVideoPreview(file) {
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    videoPreview.style.display = 'block';
    imagePreview.style.display = 'none';
    audioPreview.style.display = 'none';
}

function showAudioPreview(file) {
    fileName.textContent = file.name;
    audioPreview.style.display = 'block';
    imagePreview.style.display = 'none';
    videoPreview.style.display = 'none';
}

function clearFile() {
    selectedFile = null;
    fileType = null;
    fileInput.value = '';
    
    uploadArea.style.display = 'block';
    filePreview.style.display = 'none';
    analyzeBtn.style.display = 'none';
    results.style.display = 'none';
    
    imagePreview.src = '';
    videoPreview.src = '';
}

async function analyzeFile() {
    if (!selectedFile) return;
    
    // Hide analyze button, show loading
    analyzeBtn.style.display = 'none';
    loading.style.display = 'block';
    results.style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Send to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
            analyzeBtn.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze file. Please try again.');
        analyzeBtn.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    const { prediction, confidence, details, file_type } = data;
    
    // Update prediction badge
    const resultBadge = document.getElementById('resultBadge');
    const predictionText = document.getElementById('predictionText');
    
    if (prediction === 'fake') {
        resultBadge.className = 'result-badge fake';
        resultBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>FAKE DETECTED</span>';
    } else if (prediction === 'real') {
        resultBadge.className = 'result-badge real';
        resultBadge.innerHTML = '<i class="fas fa-check-circle"></i><span>AUTHENTIC</span>';
    } else {
        resultBadge.className = 'result-badge';
        resultBadge.innerHTML = '<i class="fas fa-question-circle"></i><span>UNKNOWN</span>';
    }
    
    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    
    const confidencePercent = (confidence * 100).toFixed(1);
    confidenceValue.textContent = confidencePercent + '%';
    confidenceFill.style.width = confidencePercent + '%';
    
    // Update additional details
    const resultDetails = document.getElementById('resultDetails');
    resultDetails.innerHTML = '';
    
    // Add file type
    addDetailItem(resultDetails, 'File Type', file_type.toUpperCase());
    addDetailItem(resultDetails, 'Raw Score', details.raw_score ? details.raw_score.toFixed(4) : 'N/A');
    
    // Add face detection status for images
    if (details.face_detected !== undefined) {
        addDetailItem(resultDetails, 'Face Detected', details.face_detected ? 'Yes' : 'No');
    }
    
    // Add video-specific details
    if (file_type === 'video' && details.metadata) {
        const meta = details.metadata;
        if (meta.total_frames_analyzed) {
            addDetailItem(resultDetails, 'Frames Analyzed', meta.total_frames_analyzed);
        }
        if (meta.fake_frame_percentage !== undefined) {
            addDetailItem(resultDetails, 'Fake Frame %', meta.fake_frame_percentage.toFixed(1) + '%');
        }
        if (meta.fake_frames !== undefined) {
            addDetailItem(resultDetails, 'Fake Frames', meta.fake_frames);
        }
        if (meta.real_frames !== undefined) {
            addDetailItem(resultDetails, 'Real Frames', meta.real_frames);
        }
    }
    
    // Add audio-specific details
    if (file_type === 'audio' && details.voice_features) {
        const voice = details.voice_features;
        if (voice.pitch_mean) {
            addDetailItem(resultDetails, 'Pitch Mean', voice.pitch_mean.toFixed(2) + ' Hz');
        }
        if (voice.jitter !== undefined) {
            addDetailItem(resultDetails, 'Jitter', (voice.jitter * 100).toFixed(2) + '%');
        }
        if (voice.shimmer !== undefined) {
            addDetailItem(resultDetails, 'Shimmer', (voice.shimmer * 100).toFixed(2) + '%');
        }
    }
    
    // Show results
    results.style.display = 'block';
    
    // Smooth scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function addDetailItem(container, label, value) {
    const item = document.createElement('div');
    item.className = 'detail-item';
    item.innerHTML = `
        <span style="font-weight: 600;">${label}:</span>
        <span>${value}</span>
    `;
    container.appendChild(item);
}

function reset() {
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Smooth scrolling for navigation links
document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
