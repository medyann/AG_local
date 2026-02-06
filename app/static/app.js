// ─── AlphaGenome Web Interface ───────────────────────────────────────────────

const API_BASE = '';
let lastResults = null;
let charts = [];

// ─── Model Status Polling ────────────────────────────────────────────────────

async function checkModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const badge = document.getElementById('model-status');
        const dot = badge.querySelector('.status-dot');
        const text = badge.querySelector('.status-text');

        badge.className = 'status-badge';

        if (data.model_loaded) {
            badge.classList.add('status-ready');
            text.textContent = 'Model Ready';
        } else if (data.model_loading) {
            badge.classList.add('status-loading');
            text.textContent = 'Loading Model...';
        } else if (data.model_error) {
            badge.classList.add('status-error');
            text.textContent = 'Model Error';
        } else {
            badge.classList.add('status-loading');
            text.textContent = 'Waiting...';
        }

        return data.model_loaded;
    } catch (e) {
        const badge = document.getElementById('model-status');
        badge.className = 'status-badge status-error';
        badge.querySelector('.status-text').textContent = 'Disconnected';
        return false;
    }
}

// Poll status every 5 seconds until model is ready
let statusInterval = setInterval(async () => {
    const ready = await checkModelStatus();
    if (ready) clearInterval(statusInterval);
}, 5000);

// Initial check
checkModelStatus();

// ─── Tab Navigation ──────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ─── Sequence Length Counter ─────────────────────────────────────────────────

document.getElementById('sequence-input').addEventListener('input', function() {
    const clean = this.value.replace(/[^ACGTacgt]/g, '');
    document.getElementById('seq-length').textContent = `Length: ${clean.length.toLocaleString()} bp`;
});

// ─── Form Submissions ────────────────────────────────────────────────────────

function getSelectedValues(selectId) {
    const select = document.getElementById(selectId);
    return Array.from(select.selectedOptions).map(o => o.value);
}

function parseOntologyTerms(inputId) {
    const val = document.getElementById(inputId).value.trim();
    if (!val) return null;
    return val.split(',').map(t => t.trim()).filter(t => t);
}

function setLoading(submitBtnId, loading) {
    const btn = document.getElementById(submitBtnId);
    const textEl = btn.querySelector('.btn-text');
    const loadingEl = btn.querySelector('.btn-loading');

    btn.disabled = loading;
    if (loading) {
        textEl.classList.add('hidden');
        loadingEl.classList.remove('hidden');
    } else {
        textEl.classList.remove('hidden');
        loadingEl.classList.add('hidden');
    }
}

function showError(message) {
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('error-section').classList.remove('hidden');
    document.getElementById('error-message').textContent = message;
}

function hideError() {
    document.getElementById('error-section').classList.add('hidden');
}

// Interval Prediction
document.getElementById('interval-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    setLoading('interval-submit', true);

    const payload = {
        chromosome: document.getElementById('interval-chr').value,
        start: parseInt(document.getElementById('interval-start').value),
        end: parseInt(document.getElementById('interval-end').value),
        organism: document.getElementById('interval-organism').value,
        output_types: getSelectedValues('interval-outputs'),
        ontology_terms: parseOntologyTerms('interval-ontology'),
    };

    try {
        const res = await fetch(`${API_BASE}/api/predict/interval`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        displayResults(data, 'interval', payload);
    } catch (err) {
        showError(err.message);
    } finally {
        setLoading('interval-submit', false);
    }
});

// Variant Prediction
document.getElementById('variant-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    setLoading('variant-submit', true);

    const payload = {
        chromosome: document.getElementById('variant-chr').value,
        start: parseInt(document.getElementById('variant-start').value),
        end: parseInt(document.getElementById('variant-end').value),
        variant_position: parseInt(document.getElementById('variant-pos').value),
        reference_bases: document.getElementById('variant-ref').value.toUpperCase(),
        alternate_bases: document.getElementById('variant-alt').value.toUpperCase(),
        organism: document.getElementById('variant-organism').value,
        output_types: getSelectedValues('variant-outputs'),
        ontology_terms: parseOntologyTerms('variant-ontology'),
    };

    try {
        const res = await fetch(`${API_BASE}/api/predict/variant`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        displayResults(data, 'variant', payload);
    } catch (err) {
        showError(err.message);
    } finally {
        setLoading('variant-submit', false);
    }
});

// Sequence Prediction
document.getElementById('sequence-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();
    setLoading('sequence-submit', true);

    const rawSeq = document.getElementById('sequence-input').value;
    const sequence = rawSeq.replace(/[^ACGTacgt]/g, '').toUpperCase();

    const payload = {
        sequence: sequence,
        organism: document.getElementById('sequence-organism').value,
        output_types: getSelectedValues('sequence-outputs'),
        ontology_terms: parseOntologyTerms('sequence-ontology'),
    };

    try {
        const res = await fetch(`${API_BASE}/api/predict/sequence`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        displayResults(data, 'sequence', payload);
    } catch (err) {
        showError(err.message);
    } finally {
        setLoading('sequence-submit', false);
    }
});

// ─── Display Results ─────────────────────────────────────────────────────────

const OUTPUT_TYPE_LABELS = {
    rna_seq: 'RNA-seq',
    cage: 'CAGE',
    dnase: 'DNase-seq',
    atac: 'ATAC-seq',
    chip_histone: 'Histone ChIP-seq',
    chip_tf: 'TF ChIP-seq',
    splice_sites: 'Splice Sites',
    splice_site_usage: 'Splice Site Usage',
    splice_junctions: 'Splice Junctions',
    contact_maps: 'Contact Maps',
    procap: 'PRO-cap',
};

const TRACK_COLORS = [
    'rgba(99, 102, 241, 0.8)',   // indigo
    'rgba(239, 68, 68, 0.8)',    // red
    'rgba(16, 185, 129, 0.8)',   // emerald
    'rgba(245, 158, 11, 0.8)',   // amber
    'rgba(139, 92, 246, 0.8)',   // violet
    'rgba(236, 72, 153, 0.8)',   // pink
    'rgba(6, 182, 212, 0.8)',    // cyan
    'rgba(132, 204, 22, 0.8)',   // lime
];

function displayResults(response, predictionType, request) {
    lastResults = response;

    // Destroy existing charts
    charts.forEach(c => c.destroy());
    charts = [];

    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('error-section').classList.add('hidden');

    // Info bar
    const info = document.getElementById('results-info');
    if (predictionType === 'variant') {
        info.innerHTML = `
            <strong>Variant Effect Prediction</strong> &mdash;
            ${request.chromosome}:${request.start.toLocaleString()}-${request.end.toLocaleString()} |
            Variant: ${request.chromosome}:${request.variant_position.toLocaleString()} ${request.reference_bases}>${request.alternate_bases}
        `;
    } else if (predictionType === 'interval') {
        info.innerHTML = `
            <strong>Interval Prediction</strong> &mdash;
            ${request.chromosome}:${request.start.toLocaleString()}-${request.end.toLocaleString()} |
            Size: ${(request.end - request.start).toLocaleString()} bp
        `;
    } else {
        info.innerHTML = `
            <strong>Sequence Prediction</strong> &mdash;
            Length: ${request.sequence.length.toLocaleString()} bp
        `;
    }

    const container = document.getElementById('charts-container');
    container.innerHTML = '';

    if (!response.success || !response.data) {
        container.innerHTML = '<p style="color: var(--text-muted);">No data returned.</p>';
        return;
    }

    if (predictionType === 'variant') {
        renderVariantCharts(response.data, container);
    } else {
        renderTrackCharts(response.data, container);
    }

    // Raw JSON
    const truncated = truncateJSON(response.data);
    document.getElementById('raw-json').textContent = JSON.stringify(truncated, null, 2);

    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function truncateJSON(obj, maxArrayLen = 20) {
    if (Array.isArray(obj)) {
        if (obj.length > maxArrayLen) {
            return [...obj.slice(0, maxArrayLen), `... (${obj.length - maxArrayLen} more items)`];
        }
        return obj.map(item => truncateJSON(item, maxArrayLen));
    }
    if (obj && typeof obj === 'object') {
        const result = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = truncateJSON(value, maxArrayLen);
        }
        return result;
    }
    return obj;
}

function renderTrackCharts(data, container) {
    for (const [outputType, trackData] of Object.entries(data)) {
        if (!trackData || !trackData.values) continue;

        const wrapper = document.createElement('div');
        wrapper.className = 'chart-wrapper';

        const label = OUTPUT_TYPE_LABELS[outputType] || outputType;
        const numTracks = trackData.num_tracks || 1;
        const resolution = trackData.resolution || 1;

        let metaHtml = `
            <div class="chart-meta">
                <span>Tracks: ${numTracks}</span>
                <span>Resolution: ${resolution} bp</span>
                <span>Data points: ${trackData.values.length.toLocaleString()}</span>
        `;
        if (trackData.interval) {
            metaHtml += `<span>Region: ${trackData.interval.chromosome}:${trackData.interval.start.toLocaleString()}-${trackData.interval.end.toLocaleString()}</span>`;
        }
        metaHtml += '</div>';

        wrapper.innerHTML = `<h3>${label}</h3><canvas></canvas>${metaHtml}`;
        container.appendChild(wrapper);

        const canvas = wrapper.querySelector('canvas');
        createTrackChart(canvas, trackData, label);
    }
}

function renderVariantCharts(data, container) {
    const refData = data.reference || {};
    const altData = data.alternate || {};

    const allOutputTypes = new Set([...Object.keys(refData), ...Object.keys(altData)]);

    for (const outputType of allOutputTypes) {
        const refTrack = refData[outputType];
        const altTrack = altData[outputType];

        if (!refTrack && !altTrack) continue;

        const wrapper = document.createElement('div');
        wrapper.className = 'chart-wrapper';

        const label = OUTPUT_TYPE_LABELS[outputType] || outputType;
        const trackInfo = refTrack || altTrack;
        const resolution = trackInfo.resolution || 1;

        wrapper.innerHTML = `
            <h3>${label} - Variant Effect</h3>
            <canvas></canvas>
            <div class="chart-meta">
                <span>Resolution: ${resolution} bp</span>
                ${refTrack ? `<span>Data points: ${refTrack.values.length.toLocaleString()}</span>` : ''}
            </div>
        `;
        container.appendChild(wrapper);

        const canvas = wrapper.querySelector('canvas');
        createVariantChart(canvas, refTrack, altTrack, label);
    }
}

function createTrackChart(canvas, trackData, label) {
    const values = trackData.values;
    const numTracks = trackData.num_tracks || 1;
    const maxTracksToShow = 4;

    const datasets = [];
    const numPoints = values.length;
    const labels = Array.from({ length: numPoints }, (_, i) => i);

    if (numTracks === 1 || !Array.isArray(values[0])) {
        // Single track: values is 1D
        const dataPoints = Array.isArray(values[0]) ? values.map(v => v[0]) : values;
        datasets.push({
            label: trackData.names ? trackData.names[0] : label,
            data: dataPoints,
            borderColor: TRACK_COLORS[0],
            backgroundColor: TRACK_COLORS[0].replace('0.8', '0.1'),
            borderWidth: 1,
            pointRadius: 0,
            fill: true,
        });
    } else {
        // Multi-track: values is 2D [positions, tracks]
        const tracksToShow = Math.min(numTracks, maxTracksToShow);
        for (let t = 0; t < tracksToShow; t++) {
            const dataPoints = values.map(row => row[t]);
            datasets.push({
                label: trackData.names ? trackData.names[t] : `Track ${t + 1}`,
                data: dataPoints,
                borderColor: TRACK_COLORS[t % TRACK_COLORS.length],
                backgroundColor: TRACK_COLORS[t % TRACK_COLORS.length].replace('0.8', '0.05'),
                borderWidth: 1,
                pointRadius: 0,
                fill: false,
            });
        }
    }

    const chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 3,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: datasets.length > 1,
                    position: 'top',
                    labels: {
                        color: '#9ca0b0',
                        font: { size: 11 },
                        boxWidth: 12,
                        padding: 10,
                    },
                },
                tooltip: {
                    backgroundColor: '#1e2130',
                    titleColor: '#e4e5ea',
                    bodyColor: '#9ca0b0',
                    borderColor: '#2d3148',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Position (bin)', color: '#6b7089', font: { size: 11 } },
                    ticks: { color: '#6b7089', maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: 'rgba(45, 49, 72, 0.5)' },
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Predicted Signal', color: '#6b7089', font: { size: 11 } },
                    ticks: { color: '#6b7089', font: { size: 10 } },
                    grid: { color: 'rgba(45, 49, 72, 0.5)' },
                },
            },
            elements: {
                line: { tension: 0 },
            },
        },
    });

    charts.push(chart);
}

function createVariantChart(canvas, refTrack, altTrack, label) {
    const datasets = [];
    const refValues = refTrack?.values || [];
    const altValues = altTrack?.values || [];
    const numPoints = Math.max(refValues.length, altValues.length);
    const labels = Array.from({ length: numPoints }, (_, i) => i);

    // For multi-track data, show first track only
    function extractFirstTrack(values) {
        if (values.length === 0) return [];
        return Array.isArray(values[0]) ? values.map(v => v[0]) : values;
    }

    if (refTrack) {
        datasets.push({
            label: 'Reference',
            data: extractFirstTrack(refValues),
            borderColor: 'rgba(107, 114, 128, 0.9)',
            backgroundColor: 'rgba(107, 114, 128, 0.05)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
        });
    }

    if (altTrack) {
        datasets.push({
            label: 'Alternate',
            data: extractFirstTrack(altValues),
            borderColor: 'rgba(239, 68, 68, 0.9)',
            backgroundColor: 'rgba(239, 68, 68, 0.05)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
        });
    }

    const chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 3,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#9ca0b0',
                        font: { size: 11 },
                        boxWidth: 12,
                        padding: 10,
                    },
                },
                tooltip: {
                    backgroundColor: '#1e2130',
                    titleColor: '#e4e5ea',
                    bodyColor: '#9ca0b0',
                    borderColor: '#2d3148',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Position (bin)', color: '#6b7089', font: { size: 11 } },
                    ticks: { color: '#6b7089', maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: 'rgba(45, 49, 72, 0.5)' },
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Predicted Signal', color: '#6b7089', font: { size: 11 } },
                    ticks: { color: '#6b7089', font: { size: 10 } },
                    grid: { color: 'rgba(45, 49, 72, 0.5)' },
                },
            },
            elements: {
                line: { tension: 0 },
            },
        },
    });

    charts.push(chart);
}

// ─── Download Results ────────────────────────────────────────────────────────

function downloadResults() {
    if (!lastResults) return;
    const blob = new Blob([JSON.stringify(lastResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alphagenome-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Load Examples ───────────────────────────────────────────────────────────

async function loadExample(type, index) {
    try {
        const res = await fetch(`${API_BASE}/api/example-queries`);
        const data = await res.json();

        if (type === 'interval') {
            const ex = data.interval_examples[index];
            if (!ex) return;
            document.getElementById('interval-chr').value = ex.chromosome;
            document.getElementById('interval-start').value = ex.start;
            document.getElementById('interval-end').value = ex.end;
            document.getElementById('interval-ontology').value = (ex.ontology_terms || []).join(', ');

            // Set output types
            const select = document.getElementById('interval-outputs');
            Array.from(select.options).forEach(opt => {
                opt.selected = ex.output_types.includes(opt.value);
            });
        } else if (type === 'variant') {
            const ex = data.variant_examples[index];
            if (!ex) return;
            document.getElementById('variant-chr').value = ex.chromosome;
            document.getElementById('variant-start').value = ex.start;
            document.getElementById('variant-end').value = ex.end;
            document.getElementById('variant-pos').value = ex.variant_position;
            document.getElementById('variant-ref').value = ex.reference_bases;
            document.getElementById('variant-alt').value = ex.alternate_bases;
            document.getElementById('variant-ontology').value = (ex.ontology_terms || []).join(', ');

            const select = document.getElementById('variant-outputs');
            Array.from(select.options).forEach(opt => {
                opt.selected = ex.output_types.includes(opt.value);
            });
        }
    } catch (e) {
        console.error('Failed to load example:', e);
    }
}
