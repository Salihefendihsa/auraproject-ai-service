// AuraProject AI Service - Frontend Demo v1.4.4

const API_BASE = '';
const POLL_INTERVAL = 2000;
const MAX_POLL_ATTEMPTS = 60;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const userNote = document.getElementById('userNote');
const generateBtn = document.getElementById('generateBtn');
const statusSection = document.getElementById('statusSection');
const statusIcon = document.getElementById('statusIcon');
const jobIdSpan = document.getElementById('jobId');
const statusText = document.getElementById('statusText');
const resultsSection = document.getElementById('resultsSection');
const outfitsGrid = document.getElementById('outfitsGrid');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');

let selectedFile = null;

// Upload handling
uploadBox.addEventListener('click', () => imageInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#22c55e';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#3498db';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#3498db';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        document.querySelector('.upload-content').style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    generateBtn.disabled = false;
}

// Generate outfits
generateBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    generateBtn.disabled = true;
    statusSection.hidden = false;
    resultsSection.hidden = true;
    errorSection.hidden = true;
    
    statusIcon.textContent = '⏳';
    statusText.textContent = 'Uploading image...';
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);
        if (userNote.value.trim()) {
            formData.append('user_note', userNote.value.trim());
        }
        
        // Submit job
        const response = await fetch(`${API_BASE}/ai/outfit`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }
        
        const data = await response.json();
        const jobId = data.job_id;
        
        jobIdSpan.textContent = jobId;
        statusText.textContent = 'Processing...';
        
        // If already completed (cache hit)
        if (data.status === 'completed' && data.outfits?.length > 0) {
            displayResults(data);
            return;
        }
        
        // Poll for job completion
        await pollJob(jobId);
        
    } catch (error) {
        showError(error.message);
    }
});

async function pollJob(jobId) {
    let attempts = 0;
    
    while (attempts < MAX_POLL_ATTEMPTS) {
        attempts++;
        statusText.textContent = `Processing... (${attempts}/${MAX_POLL_ATTEMPTS})`;
        
        try {
            const response = await fetch(`${API_BASE}/ai/jobs/${jobId}`);
            
            if (!response.ok) {
                if (response.status === 404) {
                    // Job not in DB yet, wait
                    await sleep(POLL_INTERVAL);
                    continue;
                }
                throw new Error(`Poll failed: ${response.status}`);
            }
            
            const job = await response.json();
            
            if (job.status === 'completed') {
                displayResults(job);
                return;
            } else if (job.status === 'failed') {
                throw new Error(job.error || 'Job failed');
            }
            
            // Still pending
            await sleep(POLL_INTERVAL);
            
        } catch (error) {
            if (attempts >= MAX_POLL_ATTEMPTS) {
                throw error;
            }
            await sleep(POLL_INTERVAL);
        }
    }
    
    throw new Error('Timeout: Job took too long to complete');
}

function displayResults(data) {
    statusIcon.textContent = '✅';
    statusText.textContent = data.cached ? 'Completed (cached)' : 'Completed';
    
    outfitsGrid.innerHTML = '';
    
    const outfits = data.outfits || [];
    
    outfits.forEach((outfit, index) => {
        const card = document.createElement('div');
        card.className = 'outfit-card';
        
        const renderUrl = outfit.render_url || data.assets?.renders?.[index] || '';
        
        card.innerHTML = `
            <img class="outfit-render" 
                 src="${API_BASE}${renderUrl}" 
                 alt="Outfit ${index + 1}"
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%231a1a2e%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2250%22 text-anchor=%22middle%22 fill=%22%23666%22>No Image</text></svg>'">
            <div class="outfit-info">
                <span class="outfit-rank">#${outfit.rank || index + 1}</span>
                <h3 class="outfit-style">${outfit.style_tag || 'Outfit'}</h3>
                <div class="outfit-items">
                    ${renderOutfitItems(outfit.items || {})}
                </div>
                ${outfit.explanation ? `<p style="margin-top:0.75rem;font-size:0.85rem;color:#94a3b8">${outfit.explanation}</p>` : ''}
            </div>
        `;
        
        outfitsGrid.appendChild(card);
    });
    
    resultsSection.hidden = false;
}

function renderOutfitItems(items) {
    const categories = ['top', 'bottom', 'outerwear', 'shoes'];
    
    return categories.map(cat => {
        const item = items[cat];
        if (!item) return '';
        
        const sourceClass = item.source === 'user' ? 'source-user' : 'source-suggested';
        const name = item.name || cat;
        const color = item.color || '';
        
        return `
            <div class="outfit-item">
                <span class="label">${cat}</span>
                <span class="value ${sourceClass}">${color} ${name}</span>
            </div>
        `;
    }).join('');
}

function showError(message) {
    statusSection.hidden = true;
    errorSection.hidden = false;
    errorText.textContent = message;
    generateBtn.disabled = false;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Initial state
console.log('AuraProject AI Service - Frontend Demo v1.4.4 loaded');
