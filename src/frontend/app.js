/**
 * EDA Insights - Frontend Application
 * Automated Exploratory Data Analysis
 */

// ============================
// Configuration
// ============================

const API_BASE = 'http://localhost:8000/api';

// Plotly theme configuration
const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false
};

const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(22,27,34,0.8)',
    font: {
        family: 'DM Sans, sans-serif',
        color: '#8b949e'
    },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: {
        gridcolor: 'rgba(48,54,61,0.6)',
        zerolinecolor: 'rgba(48,54,61,0.8)'
    },
    yaxis: {
        gridcolor: 'rgba(48,54,61,0.6)',
        zerolinecolor: 'rgba(48,54,61,0.8)'
    }
};

// ============================
// State Management
// ============================

const state = {
    csvFile: null,
    schemaFile: null,
    jobId: null,
    results: null,
    columns: [],
    selectedPlotColumn: null,
    customSelectedColumns: []
};

// ============================
// DOM Elements
// ============================

const elements = {
    // Screens
    uploadScreen: document.getElementById('upload-screen'),
    loadingScreen: document.getElementById('loading-screen'),
    resultsScreen: document.getElementById('results-screen'),
    
    // Upload
    csvZone: document.getElementById('csv-zone'),
    csvInput: document.getElementById('csv-input'),
    csvFilename: document.getElementById('csv-filename'),
    schemaZone: document.getElementById('schema-zone'),
    schemaInput: document.getElementById('schema-input'),
    schemaFilename: document.getElementById('schema-filename'),
    analyzeBtn: document.getElementById('analyze-btn'),
    
    // Loading
    loadingStatus: document.getElementById('loading-status'),
    progressBar: document.getElementById('progress-bar'),
    progressText: document.getElementById('progress-text'),
    
    // Results Header
    backBtn: document.getElementById('back-btn'),
    datasetName: document.getElementById('dataset-name'),
    rowCount: document.getElementById('row-count'),
    colCount: document.getElementById('col-count'),
    
    // Left Panel
    summaryText: document.getElementById('summary-text'),
    insightsList: document.getElementById('insights-list'),
    limitationsList: document.getElementById('limitations-list'),
    
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    dataTab: document.getElementById('data-tab'),
    columnsTab: document.getElementById('columns-tab'),
    plotsTab: document.getElementById('plots-tab'),
    customTab: document.getElementById('custom-tab'),
    
    // Data Table
    dataTable: document.getElementById('data-table'),
    
    // Columns
    columnsGrid: document.getElementById('columns-grid'),
    
    // Plots - Redesigned
    plotsColumnSelector: document.getElementById('plots-column-selector'),
    plotsDisplay: document.getElementById('plots-display'),
    
    // Custom Builder
    columnCheckboxes: document.getElementById('column-checkboxes'),
    llmSuggestBtn: document.getElementById('llm-suggest-btn'),
    generateCustomBtn: document.getElementById('generate-custom-btn'),
    aiSuggestionBox: document.getElementById('ai-suggestion-box'),
    aiSuggestionText: document.getElementById('ai-suggestion-text'),
    customPlotArea: document.getElementById('custom-plot-area')
};

// ============================
// Screen Management
// ============================

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
}

// ============================
// File Upload Handling
// ============================

function setupFileUpload(zone, input, filenameEl, fileType) {
    zone.addEventListener('click', () => input.click());
    
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, zone, filenameEl, fileType);
        }
    });
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileSelect(file, zone, filenameEl, fileType);
        }
    });
}

function handleFileSelect(file, zone, filenameEl, fileType) {
    if (fileType === 'csv') {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a CSV file');
            return;
        }
        state.csvFile = file;
    } else {
        if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
            alert('Please select a CSV or TXT file');
            return;
        }
        state.schemaFile = file;
    }
    
    zone.classList.add('has-file');
    filenameEl.textContent = file.name;
    
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    elements.analyzeBtn.disabled = !state.csvFile;
}

// ============================
// API Communication
// ============================

async function uploadAndAnalyze() {
    showScreen('loading-screen');
    updateProgress(0, 'Uploading file...');
    
    try {
        const formData = new FormData();
        formData.append('file', state.csvFile);
        if (state.schemaFile) {
            formData.append('schema', state.schemaFile);
        }
        
        const uploadResponse = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }
        
        const uploadData = await uploadResponse.json();
        state.jobId = uploadData.job_id;
        
        updateProgress(10, 'File uploaded, starting analysis...');
        
        await pollProgress();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed: ' + error.message);
        showScreen('upload-screen');
    }
}

async function pollProgress() {
    const pollInterval = 1000;
    
    while (true) {
        try {
            const response = await fetch(`${API_BASE}/status/${state.jobId}`);
            const data = await response.json();
            
            updateProgress(data.progress, getStatusMessage(data.progress));
            
            if (data.status === 'completed') {
                await fetchResults();
                break;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Analysis failed');
            }
            
            await sleep(pollInterval);
        } catch (error) {
            console.error('Poll error:', error);
            throw error;
        }
    }
}

function getStatusMessage(progress) {
    if (progress < 10) return 'Parsing CSV file...';
    if (progress < 20) return 'Detecting column types...';
    if (progress < 40) return 'Computing statistics...';
    if (progress < 60) return 'Generating visualizations...';
    if (progress < 80) return 'Running AI analysis...';
    return 'Preparing results...';
}

async function fetchResults() {
    const response = await fetch(`${API_BASE}/results/${state.jobId}`);
    if (!response.ok) {
        throw new Error('Failed to fetch results');
    }
    
    state.results = await response.json();
    state.columns = state.results.columns;
    
    renderResults();
    showScreen('results-screen');
}

function updateProgress(percent, message) {
    elements.progressBar.style.width = `${percent}%`;
    elements.progressText.textContent = `${Math.round(percent)}%`;
    elements.loadingStatus.textContent = message;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================
// Results Rendering
// ============================

function renderResults() {
    const results = state.results;
    
    // Header info
    elements.datasetName.textContent = results.dataset_info.filename;
    elements.rowCount.textContent = `${results.dataset_info.rows.toLocaleString()} rows`;
    elements.colCount.textContent = `${results.dataset_info.columns} columns`;
    
    // Left panel
    renderSummary(results.llm_analysis);
    renderInsights(results.llm_analysis.insights);
    renderLimitations(results.llm_analysis.limitations);
    
    // Tabs
    renderDataPreview(results.columns, results.data_preview);
    renderColumnsGrid(results.columns);
    renderPlotsColumnSelector(results.columns, results.plots);
    setupCustomBuilder(results.columns);
}

function renderSummary(llmAnalysis) {
    const summary = llmAnalysis.summary || 'No summary available.';
    elements.summaryText.textContent = summary;
}

function renderInsights(insights) {
    elements.insightsList.innerHTML = insights
        .map(insight => `<li>${escapeHtml(insight)}</li>`)
        .join('');
}

function renderLimitations(limitations) {
    elements.limitationsList.innerHTML = limitations
        .map(limitation => `<li>${escapeHtml(limitation)}</li>`)
        .join('');
}

function renderDataPreview(columns, data) {
    const thead = elements.dataTable.querySelector('thead');
    const tbody = elements.dataTable.querySelector('tbody');
    
    thead.innerHTML = `<tr>${columns.map(col => 
        `<th>${escapeHtml(col.name)}</th>`
    ).join('')}</tr>`;
    
    tbody.innerHTML = data.slice(0, 100).map(row => 
        `<tr>${row.map(cell => 
            `<td>${formatCell(cell)}</td>`
        ).join('')}</tr>`
    ).join('');
}

function formatCell(value) {
    if (value === null || value === undefined) {
        return '<span style="color: var(--text-muted)">null</span>';
    }
    if (typeof value === 'number') {
        return Number.isInteger(value) ? value : value.toFixed(4);
    }
    return escapeHtml(String(value));
}

function renderColumnsGrid(columns) {
    elements.columnsGrid.innerHTML = columns.map((col, index) => `
        <div class="column-card expanded" data-index="${index}">
            <div class="column-card-header" onclick="toggleColumnCard(${index})">
                <div>
                    <div class="column-name">${escapeHtml(col.name)}</div>
                </div>
                <span class="column-type">${formatDtype(col.dtype)}</span>
            </div>
            <div class="column-card-body">
                <div class="column-meta">
                    <span class="meta-item">Missing: <span class="meta-value">${col.missing} (${col.missing_pct}%)</span></span>
                </div>
                ${renderQualityFlags(col.quality_flags)}
                ${renderColumnStats(col)}
            </div>
        </div>
    `).join('');
}

function formatDtype(dtype) {
    const dtypeMap = {
        'numeric_int': 'Integer',
        'numeric_float': 'Float',
        'categorical': 'Category',
        'text': 'Text',
        'datetime': 'DateTime',
        'identifier': 'ID',
        'unknown': 'Unknown'
    };
    return dtypeMap[dtype] || dtype;
}

function renderQualityFlags(flags) {
    if (!flags || flags.length === 0) return '';
    
    const flagLabels = {
        'duplicate_column': 'Duplicate',
        'single_value': 'Single Value',
        'high_missing': 'High Missing',
        'potential_id': 'Potential ID'
    };
    
    return `<div class="quality-flags">
        ${flags.map(flag => 
            `<span class="quality-flag ${flag === 'potential_id' ? 'warning' : ''}">${flagLabels[flag] || flag}</span>`
        ).join('')}
    </div>`;
}

function renderColumnStats(col) {
    if (!col.stats) return '';
    
    const stats = col.stats;
    let statsHtml = '';
    
    if (col.dtype.startsWith('numeric')) {
        statsHtml = `
            <div class="stat-item"><span class="stat-label">Min</span><span class="stat-value">${formatNumber(stats.min)}</span></div>
            <div class="stat-item"><span class="stat-label">Max</span><span class="stat-value">${formatNumber(stats.max)}</span></div>
            <div class="stat-item"><span class="stat-label">Mean</span><span class="stat-value">${formatNumber(stats.mean)}</span></div>
            <div class="stat-item"><span class="stat-label">Median</span><span class="stat-value">${formatNumber(stats.median)}</span></div>
            <div class="stat-item"><span class="stat-label">Std Dev</span><span class="stat-value">${formatNumber(stats.std)}</span></div>
            <div class="stat-item"><span class="stat-label">Mode</span><span class="stat-value">${formatNumber(stats.mode)}</span></div>
            <div class="stat-item"><span class="stat-label">Q1</span><span class="stat-value">${formatNumber(stats.q1)}</span></div>
            <div class="stat-item"><span class="stat-label">Q3</span><span class="stat-value">${formatNumber(stats.q3)}</span></div>
            <div class="stat-item"><span class="stat-label">IQR</span><span class="stat-value">${formatNumber(stats.iqr)}</span></div>
            <div class="stat-item"><span class="stat-label">Outliers Low</span><span class="stat-value">${stats.outliers_low}</span></div>
            <div class="stat-item"><span class="stat-label">Outliers High</span><span class="stat-value">${stats.outliers_high}</span></div>
        `;
    } else if (col.dtype === 'categorical' && stats.top_values) {
        statsHtml = `
            <div class="stat-item"><span class="stat-label">Unique</span><span class="stat-value">${stats.unique_count}</span></div>
            ${stats.top_values.slice(0, 5).map(v => 
                `<div class="stat-item"><span class="stat-label">${escapeHtml(String(v.value))}</span><span class="stat-value">${v.count} (${v.percent}%)</span></div>`
            ).join('')}
        `;
    }
    
    return `<div class="column-stats">${statsHtml}</div>`;
}

function formatNumber(num) {
    if (num === null || num === undefined) return '-';
    if (Number.isInteger(num)) return num.toLocaleString();
    return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function toggleColumnCard(index) {
    const card = document.querySelector(`.column-card[data-index="${index}"]`);
    card.classList.toggle('expanded');
}

window.toggleColumnCard = toggleColumnCard;

// ============================
// Plots Tab - Redesigned
// ============================

function renderPlotsColumnSelector(columns, plots) {
    // Get columns that have plots
    const columnsWithPlots = new Set();
    if (plots && plots.auto) {
        plots.auto.forEach(plot => columnsWithPlots.add(plot.column));
    }
    
    // Filter out columns with no plots (single value, ID, etc.)
    const plottableColumns = columns.filter(col => {
        const flags = col.quality_flags || [];
        return !flags.includes('single_value') && !flags.includes('potential_id');
    });
    
    elements.plotsColumnSelector.innerHTML = plottableColumns.map(col => {
        const icon = getColumnIcon(col.dtype);
        const hasPlots = columnsWithPlots.has(col.name);
        return `
            <button class="plot-column-btn ${hasPlots ? '' : 'no-plots'}" 
                    data-column="${escapeHtml(col.name)}"
                    onclick="selectPlotColumn('${escapeHtml(col.name)}')">
                <span class="col-icon">${icon}</span>
                <span class="col-name">${escapeHtml(col.name)}</span>
                <span class="col-type">${formatDtype(col.dtype)}</span>
            </button>
        `;
    }).join('');
}

function getColumnIcon(dtype) {
    const icons = {
        'numeric_int': '🔢',
        'numeric_float': '📊',
        'categorical': '🏷️',
        'text': '📝',
        'datetime': '📅',
        'identifier': '🔑',
        'unknown': '❓'
    };
    return icons[dtype] || '📊';
}

function selectPlotColumn(columnName) {
    // Update button states
    document.querySelectorAll('.plot-column-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`.plot-column-btn[data-column="${columnName}"]`)?.classList.add('active');
    
    state.selectedPlotColumn = columnName;
    
    // Get plots for this column
    const plots = state.results.plots?.auto?.filter(p => p.column === columnName) || [];
    
    // Render plots display
    renderPlotsDisplay(columnName, plots);
}

function renderPlotsDisplay(columnName, plots) {
    if (plots.length === 0) {
        elements.plotsDisplay.innerHTML = `
            <div class="plots-display-header">
                <h3>No plots available for ${escapeHtml(columnName)}</h3>
                <button class="close-plots-btn" onclick="closePlotsDisplay()">Close</button>
            </div>
            <p style="color: var(--text-secondary);">This column may have too few unique values or be flagged as an identifier.</p>
        `;
        elements.plotsDisplay.classList.add('active');
        return;
    }
    
    elements.plotsDisplay.innerHTML = `
        <div class="plots-display-header">
            <h3>Visualizations for ${escapeHtml(columnName)}</h3>
            <button class="close-plots-btn" onclick="closePlotsDisplay()">Close</button>
        </div>
        <div class="plots-grid">
            ${plots.map((plot, index) => `
                <div class="plot-card">
                    <div class="plot-title">${capitalizeFirst(plot.type)}</div>
                    <div class="plot-container" id="col-plot-${index}"></div>
                </div>
            `).join('')}
        </div>
    `;
    
    elements.plotsDisplay.classList.add('active');
    
    // Render Plotly charts
    setTimeout(() => {
        plots.forEach((plot, index) => {
            const container = document.getElementById(`col-plot-${index}`);
            if (plot.data && plot.data.data && container) {
                const layout = { ...PLOTLY_LAYOUT, ...plot.data.layout };
                Plotly.newPlot(container, plot.data.data, layout, PLOTLY_CONFIG);
            }
        });
    }, 100);
}

function closePlotsDisplay() {
    elements.plotsDisplay.classList.remove('active');
    document.querySelectorAll('.plot-column-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    state.selectedPlotColumn = null;
}

window.selectPlotColumn = selectPlotColumn;
window.closePlotsDisplay = closePlotsDisplay;

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// ============================
// Custom Plot Builder
// ============================

function setupCustomBuilder(columns) {
    // Render column checkboxes
    elements.columnCheckboxes.innerHTML = columns
        .filter(col => {
            const flags = col.quality_flags || [];
            return !flags.includes('single_value');
        })
        .map(col => `
            <label class="column-checkbox">
                <input type="checkbox" value="${escapeHtml(col.name)}" onchange="updateCustomSelection()">
                <span class="cb-name">${escapeHtml(col.name)}</span>
                <span class="cb-type">${formatDtype(col.dtype)}</span>
            </label>
        `).join('');
    
    // Setup event listeners
    elements.llmSuggestBtn.addEventListener('click', getLLMSuggestion);
    elements.generateCustomBtn.addEventListener('click', generateCustomPlot);
}

function updateCustomSelection() {
    const checkboxes = elements.columnCheckboxes.querySelectorAll('input:checked');
    state.customSelectedColumns = Array.from(checkboxes).map(cb => cb.value);
    
    // Hide previous suggestion
    elements.aiSuggestionBox.classList.add('hidden');
}

window.updateCustomSelection = updateCustomSelection;

async function getLLMSuggestion() {
    if (state.customSelectedColumns.length === 0) {
        alert('Please select at least one column');
        return;
    }
    
    elements.llmSuggestBtn.disabled = true;
    elements.llmSuggestBtn.innerHTML = '<span>Thinking...</span>';
    
    try {
        const response = await fetch(`${API_BASE}/suggest-plot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                columns: state.customSelectedColumns
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get suggestion');
        }
        
        const suggestion = await response.json();
        
        // Display suggestion
        elements.aiSuggestionText.textContent = suggestion.reasoning;
        elements.aiSuggestionBox.classList.remove('hidden');
        
        // Auto-select the suggested plot type
        const radioBtn = document.querySelector(`input[name="plot-type"][value="${suggestion.plot_type}"]`);
        if (radioBtn) {
            radioBtn.checked = true;
        }
        
    } catch (error) {
        console.error('Suggestion error:', error);
        alert('Failed to get AI suggestion');
    } finally {
        elements.llmSuggestBtn.disabled = false;
        elements.llmSuggestBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
            </svg>
            AI Suggest
        `;
    }
}

async function generateCustomPlot() {
    const selectedType = document.querySelector('input[name="plot-type"]:checked');
    
    if (!selectedType) {
        alert('Please select a plot type');
        return;
    }
    
    if (state.customSelectedColumns.length === 0) {
        alert('Please select at least one column');
        return;
    }
    
    const plotType = selectedType.value;
    const columns = state.customSelectedColumns;
    
    // Validate column count for plot type
    const singleVarPlots = ['histogram', 'boxplot', 'violin', 'bar'];
    const twoVarPlots = ['scatter', 'line', 'grouped_bar'];
    const threeVarPlots = ['scatter_color', 'bubble'];
    const specialPlots = ['correlation', 'pairplot'];
    
    if (singleVarPlots.includes(plotType) && columns.length !== 1) {
        if (columns.length > 1) {
            // Just use first column
        } else {
            alert('Please select 1 column for this plot type');
            return;
        }
    }
    
    if (twoVarPlots.includes(plotType) && columns.length < 2) {
        alert('Please select 2 columns for this plot type');
        return;
    }
    
    if (threeVarPlots.includes(plotType) && columns.length < 3) {
        alert('Please select 3 columns for this plot type');
        return;
    }
    
    elements.generateCustomBtn.disabled = true;
    elements.generateCustomBtn.textContent = 'Generating...';
    
    try {
        const response = await fetch(`${API_BASE}/generate-plot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                plot_type: plotType,
                x_column: columns[0],
                y_column: columns[1] || null,
                color_column: columns[2] || null,
                columns: columns
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate plot');
        }
        
        const plotData = await response.json();
        
        // Render the plot
        elements.customPlotArea.innerHTML = '<div id="custom-plot-render" style="width: 100%; height: 500px;"></div>';
        
        const layout = { ...PLOTLY_LAYOUT, ...plotData.plot_layout };
        Plotly.newPlot('custom-plot-render', plotData.plot_data, layout, PLOTLY_CONFIG);
        
    } catch (error) {
        console.error('Generate error:', error);
        alert('Failed to generate plot');
    } finally {
        elements.generateCustomBtn.disabled = false;
        elements.generateCustomBtn.textContent = 'Generate Plot';
    }
}

// ============================
// Tab Navigation
// ============================

function setupTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Update buttons
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update panes
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            document.getElementById(`${tabId}-tab`).classList.add('active');
            
            // Trigger resize for plots
            if (tabId === 'plots' || tabId === 'custom') {
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            }
        });
    });
}

// ============================
// Utilities
// ============================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================
// Initialization
// ============================

function init() {
    // Setup file uploads
    setupFileUpload(elements.csvZone, elements.csvInput, elements.csvFilename, 'csv');
    setupFileUpload(elements.schemaZone, elements.schemaInput, elements.schemaFilename, 'schema');
    
    // Analyze button
    elements.analyzeBtn.addEventListener('click', uploadAndAnalyze);
    
    // Back button
    elements.backBtn.addEventListener('click', () => {
        state.csvFile = null;
        state.schemaFile = null;
        state.jobId = null;
        state.results = null;
        state.selectedPlotColumn = null;
        state.customSelectedColumns = [];
        
        elements.csvZone.classList.remove('has-file');
        elements.schemaZone.classList.remove('has-file');
        elements.csvFilename.textContent = '';
        elements.schemaFilename.textContent = '';
        elements.csvInput.value = '';
        elements.schemaInput.value = '';
        updateAnalyzeButton();
        
        showScreen('upload-screen');
    });
    
    // Tabs
    setupTabs();
    
    console.log('EDA Insights initialized');
}

// Start app
document.addEventListener('DOMContentLoaded', init);