/**
 * AI Fire Detection System — Dashboard JavaScript
 * Real-time status updates, controls, and alert management.
 */

// ─── SocketIO Connection ──────────────────────
const socket = io({ reconnection: true, reconnectionDelay: 2000, reconnectionAttempts: 50 });

// ─── State ────────────────────────────────────
let fireDetected = false;
let alertDismissed = false;
let startTime = Date.now();
let uptimeInterval = null;

// ─── DOM Elements ─────────────────────────────
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const statusRing = document.getElementById('statusRing');
const statusEmoji = document.getElementById('statusEmoji');
const statusLabel = document.getElementById('statusLabel');
const fpsDisplay = document.getElementById('fpsDisplay');
const totalDetections = document.getElementById('totalDetections');
const sessionAlerts = document.getElementById('sessionAlerts');
const detectionFPS = document.getElementById('detectionFPS');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const modelBadge = document.getElementById('modelBadge');
const videoPanel = document.querySelector('.hud-background');
const fireAlertOverlay = document.getElementById('fireAlertOverlay');
const alertDetails = document.getElementById('alertDetails');
const uptimeDisplay = document.getElementById('uptimeDisplay');
const connectionStatus = document.getElementById('connectionStatus');
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const cooldownSlider = document.getElementById('cooldownSlider');
const cooldownValue = document.getElementById('cooldownValue');
const soundToggle = document.getElementById('soundToggle');
const cameraSelect = document.getElementById('cameraSelect');
const historyList = document.getElementById('historyList');

// ─── SocketIO Event Handlers ──────────────────
socket.on('connect', () => {
    updateConnectionStatus(true);
    console.log('Connected to server');
    fetchStatus();
    fetchCameras();
    refreshHistory();
});

socket.on('disconnect', () => {
    updateConnectionStatus(false);
    console.log('Disconnected from server');
});

socket.on('status_update', (data) => {
    updateDashboard(data);
});

socket.on('fire_detected', (data) => {
    handleFireDetected(data);
});

// ─── Dashboard Update ─────────────────────────
function updateDashboard(data) {
    const isFire = data.fire_detected;

    // Update status badge
    statusBadge.className = 'sys-status ' + (isFire ? 'danger' : 'safe');
    statusText.textContent = isFire ? 'SYS.CRITICAL' : 'SYS.MONITORING';

    // Update status ring
    statusRing.className = 'analysis-ring ' + (isFire ? 'danger' : 'safe');
    statusEmoji.textContent = isFire ? '⚠' : '✓';
    statusLabel.textContent = isFire ? 'THREAT FOUND' : 'SECURE';

    // Video panel fire border
    if (isFire) {
        videoPanel.classList.add('fire-active');
    } else {
        videoPanel.classList.remove('fire-active');
    }

    // Stats
    detectionFPS.textContent = data.fps || 0;
    fpsDisplay.textContent = (data.fps || 0) + ' FPS';
    totalDetections.textContent = data.total_detections || 0;

    // Model badge
    if (data.mode) {
        modelBadge.textContent = data.mode.toUpperCase();
    }

    fireDetected = isFire;
}

function handleFireDetected(data) {
    sessionAlerts.textContent = data.alert_count || 0;

    // Show confidence
    if (data.detections && data.detections.length > 0) {
        const maxConf = Math.max(...data.detections.map(d => d.confidence));
        confidenceDisplay.textContent = Math.round(maxConf * 100) + '%';
    }

    // Show alert overlay (first detection only, until dismissed)
    if (!alertDismissed) {
        const det = data.detections && data.detections[0];
        const confText = det ? Math.round(det.confidence * 100) + '%' : 'N/A';
        alertDetails.textContent = `THERMAL ANOMALY: ${confText} CONFIDENCE`;
        fireAlertOverlay.classList.remove('hidden');
    }

    // Refresh history
    refreshHistory();
}

function dismissAlert() {
    fireAlertOverlay.classList.add('hidden');
    alertDismissed = true;
    // Re-enable alert after 30 seconds
    setTimeout(() => { alertDismissed = false; }, 30000);
}

// ─── API Calls ────────────────────────────────
async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.start_time) {
            startTime = new Date(data.start_time).getTime();
        }
        updateDashboard({
            fire_detected: data.fire_detected,
            fps: data.fps,
            total_detections: data.total_detections,
            mode: data.detection_mode,
        });
        sessionAlerts.textContent = data.alert_count || 0;
    } catch (e) { console.error('Failed to fetch status:', e); }
}

async function fetchCameras() {
    try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        cameraSelect.innerHTML = '';
        data.cameras.forEach(idx => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `Camera ${idx}${idx === data.current ? ' (Active)' : ''}`;
            opt.selected = idx === data.current;
            cameraSelect.appendChild(opt);
        });
    } catch (e) { console.error('Failed to fetch cameras:', e); }
}

async function refreshHistory() {
    try {
        const res = await fetch('/api/detections?limit=30');
        const data = await res.json();
        renderHistory(data);
    } catch (e) { console.error('Failed to fetch history:', e); }
}

function renderHistory(events) {
    if (!events || events.length === 0) {
        historyList.innerHTML = '<div class="history-empty"><span>🛡️</span><p>No fire detections recorded</p></div>';
        return;
    }

    // Show newest first
    const reversed = [...events].reverse();
    historyList.innerHTML = reversed.map(ev => {
        const time = new Date(ev.timestamp).toLocaleTimeString();
        const date = new Date(ev.timestamp).toLocaleDateString();
        const maxConf = ev.detections && ev.detections.length > 0
            ? Math.max(...ev.detections.map(d => d.confidence))
            : 0;
        return `
            <div class="history-item">
                <span class="history-item-icon">🔥</span>
                <div class="history-item-info">
                    <div class="history-item-title">Fire Alert #${ev.alert_number}</div>
                    <div class="history-item-time">${date} ${time}</div>
                </div>
                <span class="history-item-conf">${Math.round(maxConf * 100)}%</span>
            </div>
        `;
    }).join('');
}

// ─── Controls ─────────────────────────────────
sensitivitySlider.addEventListener('input', () => {
    sensitivityValue.textContent = sensitivitySlider.value + '%';
});
sensitivitySlider.addEventListener('change', () => {
    updateSettings({ confidence: parseInt(sensitivitySlider.value) / 100 });
});

cooldownSlider.addEventListener('input', () => {
    cooldownValue.textContent = cooldownSlider.value + 's';
});
cooldownSlider.addEventListener('change', () => {
    updateSettings({ cooldown: parseInt(cooldownSlider.value) });
});

soundToggle.addEventListener('change', () => {
    updateSettings({ sound_enabled: soundToggle.checked });
});

cameraSelect.addEventListener('change', () => {
    updateSettings({ camera_index: parseInt(cameraSelect.value) });
    // Reload video feed
    const feed = document.getElementById('videoFeed');
    feed.src = '/video_feed?' + Date.now();
});

async function updateSettings(settings) {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });
    } catch (e) { console.error('Settings update failed:', e); }
}

async function testAlarm() {
    const btn = document.getElementById('testAlarmBtn');
    btn.textContent = '🔔 Playing...';
    btn.disabled = true;
    try {
        const res = await fetch('/api/test_alarm', { method: 'POST' });
        const data = await res.json();
        if (data && data.arduino_connected === false) {
            console.warn('Arduino not connected. Buzzer hardware test was skipped.');
            alert('Arduino not connected. Go to /api/hardware_status to see ports and connection state.');
        }
    } catch (e) { console.error('Alarm test failed:', e); }
    setTimeout(() => { btn.textContent = '🔔 Test Alarm'; btn.disabled = false; }, 3000);
}

async function reloadModel() {
    const btn = document.getElementById('reloadModelBtn');
    btn.textContent = '⏳ Loading...';
    btn.disabled = true;
    try {
        const res = await fetch('/api/reload_model', { method: 'POST' });
        const data = await res.json();
        if (data.mode) modelBadge.textContent = data.mode.toUpperCase();
    } catch (e) { console.error('Model reload failed:', e); }
    setTimeout(() => { btn.textContent = '🤖 Reload Model'; btn.disabled = false; }, 2000);
}

// ─── Utility ──────────────────────────────────
function updateConnectionStatus(connected) {
    const dot = connectionStatus.querySelector('.conn-dot');
    if (connected) {
        dot.className = 'conn-dot connected';
        connectionStatus.innerHTML = '<span class="conn-dot connected"></span> Connected';
    } else {
        dot.className = 'conn-dot disconnected';
        connectionStatus.innerHTML = '<span class="conn-dot disconnected"></span> Reconnecting...';
    }
}

function updateUptime() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    uptimeDisplay.textContent = `${hrs}:${mins}:${secs}`;
}

// ─── Init ─────────────────────────────────────
uptimeInterval = setInterval(updateUptime, 1000);
setInterval(fetchStatus, 5000);
setInterval(refreshHistory, 15000);

// Handle video feed errors
document.getElementById('videoFeed').addEventListener('error', () => {
    setTimeout(() => {
        document.getElementById('videoFeed').src = '/video_feed?' + Date.now();
    }, 2000);
});

console.log('🔥 FireGuard AI Dashboard initialized');
