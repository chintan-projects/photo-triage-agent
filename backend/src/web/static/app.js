// Photo Triage Agent - Minimal JavaScript

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    if (seconds < 60) return Math.floor(seconds) + 's';
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
}

// Photo selection state
const selectedPhotos = new Set();

// Update selection UI
function updateSelectionUI() {
    const countEl = document.getElementById('selected-count');
    const trashBtn = document.getElementById('trash-btn');

    if (countEl) {
        countEl.textContent = `${selectedPhotos.size} selected`;
    }
    if (trashBtn) {
        trashBtn.disabled = selectedPhotos.size === 0;
    }
}

// Initialize selection handlers
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.photo-select').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const photoId = parseInt(e.target.dataset.id);
            if (e.target.checked) {
                selectedPhotos.add(photoId);
            } else {
                selectedPhotos.delete(photoId);
            }
            updateSelectionUI();
        });
    });
});

// API helpers
async function apiPost(url, data) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return response.json();
}

async function apiGet(url) {
    const response = await fetch(url);
    return response.json();
}

// Trash photos
async function trashPhotos(photoIds) {
    if (!photoIds.length) {
        alert('No photos selected');
        return false;
    }

    if (!confirm(`Move ${photoIds.length} photos to trash?`)) {
        return false;
    }

    try {
        const result = await apiPost('/actions/trash', { photo_ids: photoIds });
        if (result.success) {
            alert(result.message);
            window.location.reload();
            return true;
        } else {
            alert('Error: ' + (result.error || result.message));
            return false;
        }
    } catch (error) {
        alert('Error: ' + error.message);
        return false;
    }
}

// Undo action
async function undoAction(actionId) {
    try {
        const result = await apiPost('/actions/undo', { action_id: actionId });
        if (result.success) {
            alert(result.message);
            window.location.reload();
            return true;
        } else {
            alert('Error: ' + (result.error || result.message));
            return false;
        }
    } catch (error) {
        alert('Error: ' + error.message);
        return false;
    }
}
