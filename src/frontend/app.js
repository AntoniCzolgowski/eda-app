/**
 * EDA Insights - Frontend Application
 * Automated Exploratory Data Analysis
 * Enhanced with multi-value filters (tags), no row limit, OR logic within columns
 */

// ============================
// Configuration
// ============================

const API_BASE = 'http://localhost:8000/api';

// Display limit for table rendering (filtering searches ALL data)
const DISPLAY_LIMIT = 1000;

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
    margin: { t: 40, r: 20, b: 100, l: 50 },
    xaxis: {
        gridcolor: 'rgba(48,54,61,0.6)',
        zerolinecolor: 'rgba(48,54,61,0.8)',
        automargin: true
    },
    yaxis: {
        gridcolor: 'rgba(48,54,61,0.6)',
        zerolinecolor: 'rgba(48,54,61,0.8)',
        automargin: true
    }
};

// Available aggregation functions
const AGGREGATION_FUNCTIONS = [
    { value: '', label: '-- None --' },
    { value: 'count', label: 'Count' },
    { value: 'sum', label: 'Sum' },
    { value: 'avg', label: 'Average' },
    { value: 'min', label: 'Minimum' },
    { value: 'max', label: 'Maximum' },
    { value: 'median', label: 'Median' },
    { value: 'std', label: 'Std Dev' },
    { value: 'nunique', label: 'Unique Count' }
];

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
    customSelectedColumns: [],
    // Data table state
    fullData: [],
    filteredData: [],
    // Filters: { columnIdx: [value1, value2, ...] } - array of values per column
    filters: {},
    sortColumn: null,
    sortDirection: 'asc',
    // Plot state
    lastPlotResult: null
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
    summaryList: document.getElementById('summary-list'),
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
    clearFiltersBtn: document.getElementById('clear-filters-btn'),
    filterStatus: document.getElementById('filter-status'),
    rowDisplayCount: document.getElementById('row-display-count'),
    
    // Columns
    columnsGrid: document.getElementById('columns-grid'),
    
    // Plots - Redesigned
    plotsColumnSelector: document.getElementById('plots-column-selector'),
    plotsDisplay: document.getElementById('plots-display'),
    
    // Custom Builder
    xAxisSelect: document.getElementById('x-axis-select'),
    yAxisSelect: document.getElementById('y-axis-select'),
    colorAxisSelect: document.getElementById('color-axis-select'),
    aggregationSelect: document.getElementById('aggregation-select'),
    plotOptionsSection: document.getElementById('plot-options-section'),
    binSizeOption: document.getElementById('bin-size-option'),
    binCountInput: document.getElementById('bin-count-input'),
    showAllToggle: document.getElementById('show-all-toggle'),
    categoryHint: document.getElementById('category-hint'),
    xMinInput: document.getElementById('x-min-input'),
    xMaxInput: document.getElementById('x-max-input'),
    yMinInput: document.getElementById('y-min-input'),
    yMaxInput: document.getElementById('y-max-input'),
    xTickInput: document.getElementById('x-tick-input'),
    yTickInput: document.getElementById('y-tick-input'),
    markerSizeInput: document.getElementById('marker-size-input'),
    markerSizeValue: document.getElementById('marker-size-value'),
    opacityInput: document.getElementById('opacity-input'),
    opacityValue: document.getElementById('opacity-value'),
    scaleTypeInput: document.getElementById('scale-type-input'),
    gridToggle: document.getElementById('grid-toggle'),
    customTitleInput: document.getElementById('custom-title-input'),
    llmSuggestBtn: document.getElementById('llm-suggest-btn'),
    resetBuilderBtn: document.getElementById('reset-builder-btn'),
    generateCustomBtn: document.getElementById('generate-custom-btn'),
    aiSuggestionBox: document.getElementById('ai-suggestion-box'),
    aiSuggestionText: document.getElementById('ai-suggestion-text'),
    customPlotArea: document.getElementById('custom-plot-area'),
    categoryInfo: document.getElementById('category-info'),
    categoryInfoText: document.getElementById('category-info-text'),
    showAllBtn: document.getElementById('show-all-btn')
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
    console.log('handleFileSelect:', fileType, file.name);
    
    if (fileType === 'csv') {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a CSV file');
            return;
        }
        state.csvFile = file;
        console.log('state.csvFile set to:', state.csvFile);
    } else {
        if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt') && !file.name.endsWith('.pdf')) {
            alert('Please select a CSV, TXT, or PDF file');
            return;
        }
        state.schemaFile = file;
    }
    
    zone.classList.add('has-file');
    filenameEl.textContent = file.name;
    
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    const hasFile = !!state.csvFile;
    console.log('updateAnalyzeButton - hasFile:', hasFile);
    elements.analyzeBtn.disabled = !hasFile;
}

// ============================
// API Communication
// ============================

async function uploadAndAnalyze() {
    console.log('uploadAndAnalyze called');
    console.log('state.csvFile:', state.csvFile);
    
    if (!state.csvFile) {
        alert('No file selected');
        return;
    }
    
    // Save file reference before clearing
    const csvFile = state.csvFile;
    const schemaFile = state.schemaFile;
    
    // Clear previous results (but keep file inputs)
    clearPreviousResults();
    
    // Restore file references
    state.csvFile = csvFile;
    state.schemaFile = schemaFile;
    
    showScreen('loading-screen');
    updateProgress(0, 'Uploading file...');
    
    try {
        const formData = new FormData();
        formData.append('file', state.csvFile);
        if (state.schemaFile) {
            formData.append('schema', state.schemaFile);
        }
        
        console.log('Sending request to:', `${API_BASE}/upload`);
        
        const uploadResponse = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Upload response status:', uploadResponse.status);
        
        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text();
            console.error('Upload error:', errorText);
            throw new Error('Upload failed: ' + errorText);
        }
        
        const uploadData = await uploadResponse.json();
        console.log('Upload data:', uploadData);
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
            
            // Handle non-JSON responses
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                console.error('Non-JSON response from status endpoint');
                await sleep(pollInterval);
                continue;
            }
            
            const text = await response.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (parseError) {
                console.error('JSON parse error:', parseError, 'Response:', text.substring(0, 200));
                await sleep(pollInterval);
                continue;
            }
            
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
    
    // Store full data for filtering - ALL rows, no limit
    state.fullData = state.results.data_preview || [];
    state.filteredData = [...state.fullData];
    state.filters = {};
    state.sortColumn = null;
    state.sortDirection = 'asc';
    
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
    renderDataPreview(results.columns, state.filteredData);
    renderColumnsGrid(results.columns);
    renderPlotsColumnSelector(results.columns, results.plots);
    setupCustomBuilder(results.columns);
}

function renderSummary(llmAnalysis) {
    const summary = llmAnalysis.summary;
    
    // Handle both array (new format) and string (old format)
    if (Array.isArray(summary)) {
        elements.summaryList.innerHTML = summary
            .map(item => `<li>${escapeHtml(item)}</li>`)
            .join('');
    } else {
        // Old string format - split into bullets or show as single item
        const text = summary || 'No summary available.';
        elements.summaryList.innerHTML = `<li>${escapeHtml(text)}</li>`;
    }
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

// ============================
// Data Table with Multi-Value Filtering & Sorting
// ============================

function renderDataPreview(columns, data) {
    const thead = elements.dataTable.querySelector('thead');
    const tbody = elements.dataTable.querySelector('tbody');
    
    // Build header with filter inputs (multi-value tags)
    thead.innerHTML = `
        <tr class="header-row">
            ${columns.map((col, idx) => `
                <th data-column="${idx}">
                    <div class="th-content">
                        <span class="col-name" onclick="toggleSort(${idx})">${escapeHtml(col.name)}</span>
                        <span class="sort-icon" id="sort-icon-${idx}"></span>
                    </div>
                </th>
            `).join('')}
        </tr>
        <tr class="filter-row">
            ${columns.map((col, idx) => `
                <th>
                    <div class="filter-cell" id="filter-cell-${idx}">
                        <div class="filter-tags" id="filter-tags-${idx}"></div>
                        <div class="filter-input-wrapper">
                            <input type="text" 
                                   class="filter-input" 
                                   id="filter-input-${idx}"
                                   data-column="${idx}" 
                                   placeholder="Filter..." 
                                   onkeydown="handleFilterKeydown(event, ${idx})">
                            <button class="filter-add-btn" onclick="addFilterFromInput(${idx})" title="Add filter">+</button>
                        </div>
                    </div>
                </th>
            `).join('')}
        </tr>
    `;
    
    // Render body - ALL rows
    renderTableBody(data);
    
    // Update display count
    updateRowDisplayCount();
}

function renderTableBody(data) {
    const tbody = elements.dataTable.querySelector('tbody');
    
    // Apply display limit for rendering (filtering still uses ALL data)
    const displayData = data.slice(0, DISPLAY_LIMIT);
    
    tbody.innerHTML = displayData.map(row => 
        `<tr>${row.map(cell => 
            `<td>${formatCell(cell)}</td>`
        ).join('')}</tr>`
    ).join('');
}

function handleFilterKeydown(event, columnIdx) {
    if (event.key === 'Enter') {
        event.preventDefault();
        addFilterFromInput(columnIdx);
    }
}

function addFilterFromInput(columnIdx) {
    const input = document.getElementById(`filter-input-${columnIdx}`);
    const value = input.value.trim();
    
    if (!value) return;
    
    // Initialize array if needed
    if (!state.filters[columnIdx]) {
        state.filters[columnIdx] = [];
    }
    
    // Check if value already exists (case-insensitive)
    const lowerValue = value.toLowerCase();
    if (state.filters[columnIdx].some(v => v.toLowerCase() === lowerValue)) {
        input.value = '';
        return;
    }
    
    // Add value to filter array
    state.filters[columnIdx].push(value);
    
    // Clear input
    input.value = '';
    
    // Update UI and apply filters
    renderFilterTags(columnIdx);
    applyFiltersAndSort();
}

function removeFilter(columnIdx, valueIndex) {
    if (state.filters[columnIdx]) {
        state.filters[columnIdx].splice(valueIndex, 1);
        
        // Remove empty arrays
        if (state.filters[columnIdx].length === 0) {
            delete state.filters[columnIdx];
        }
        
        renderFilterTags(columnIdx);
        applyFiltersAndSort();
    }
}

function renderFilterTags(columnIdx) {
    const tagsContainer = document.getElementById(`filter-tags-${columnIdx}`);
    if (!tagsContainer) return;
    
    const values = state.filters[columnIdx] || [];
    
    tagsContainer.innerHTML = values.map((value, idx) => `
        <span class="filter-tag">
            <span class="filter-tag-text">${escapeHtml(value)}</span>
            <button class="filter-tag-remove" onclick="removeFilter(${columnIdx}, ${idx})">×</button>
        </span>
    `).join('');
}

function renderAllFilterTags() {
    state.columns.forEach((_, idx) => {
        renderFilterTags(idx);
    });
}

function toggleSort(columnIdx) {
    if (state.sortColumn === columnIdx) {
        // Toggle direction or clear
        if (state.sortDirection === 'asc') {
            state.sortDirection = 'desc';
        } else if (state.sortDirection === 'desc') {
            state.sortColumn = null;
            state.sortDirection = 'asc';
        }
    } else {
        state.sortColumn = columnIdx;
        state.sortDirection = 'asc';
    }
    
    applyFiltersAndSort();
    updateSortIcons();
}

function updateSortIcons() {
    // Clear all sort icons
    state.columns.forEach((_, idx) => {
        const icon = document.getElementById(`sort-icon-${idx}`);
        if (icon) {
            icon.textContent = '';
            icon.className = 'sort-icon';
        }
    });
    
    // Set active sort icon
    if (state.sortColumn !== null) {
        const icon = document.getElementById(`sort-icon-${state.sortColumn}`);
        if (icon) {
            icon.textContent = state.sortDirection === 'asc' ? '▲' : '▼';
            icon.className = 'sort-icon active';
        }
    }
}

function applyFiltersAndSort() {
    let data = [...state.fullData];
    
    // Apply filters
    // Logic: AND between columns, OR within column values
    const filterKeys = Object.keys(state.filters);
    if (filterKeys.length > 0) {
        data = data.filter(row => {
            // For each column with filters (AND logic)
            return filterKeys.every(colIdx => {
                const filterValues = state.filters[colIdx];
                const cellValue = String(row[colIdx] ?? '').toLowerCase();
                
                // Check if cell matches ANY of the filter values (OR logic)
                return filterValues.some(filterVal => 
                    cellValue.includes(filterVal.toLowerCase())
                );
            });
        });
    }
    
    // Apply sort
    if (state.sortColumn !== null) {
        data.sort((a, b) => {
            let valA = a[state.sortColumn];
            let valB = b[state.sortColumn];
            
            // Handle nulls
            if (valA === null || valA === undefined) valA = '';
            if (valB === null || valB === undefined) valB = '';
            
            // Try numeric comparison
            const numA = parseFloat(valA);
            const numB = parseFloat(valB);
            
            if (!isNaN(numA) && !isNaN(numB)) {
                return state.sortDirection === 'asc' ? numA - numB : numB - numA;
            }
            
            // String comparison
            const strA = String(valA).toLowerCase();
            const strB = String(valB).toLowerCase();
            
            if (state.sortDirection === 'asc') {
                return strA.localeCompare(strB);
            } else {
                return strB.localeCompare(strA);
            }
        });
    }
    
    state.filteredData = data;
    renderTableBody(data);
    updateRowDisplayCount();
    updateFilterStatus();
}

function updateRowDisplayCount() {
    const total = state.fullData.length;
    const filtered = state.filteredData.length;
    const displayed = Math.min(filtered, DISPLAY_LIMIT);
    
    // Count total filter values
    const filterCount = Object.values(state.filters).reduce((sum, arr) => sum + arr.length, 0);
    
    if (filterCount > 0) {
        if (filtered > DISPLAY_LIMIT) {
            elements.rowDisplayCount.textContent = `Showing ${displayed.toLocaleString()} of ${filtered.toLocaleString()} matches (${total.toLocaleString()} total)`;
        } else {
            elements.rowDisplayCount.textContent = `Showing ${filtered.toLocaleString()} matches of ${total.toLocaleString()} total`;
        }
    } else {
        if (total > DISPLAY_LIMIT) {
            elements.rowDisplayCount.textContent = `Showing first ${displayed.toLocaleString()} of ${total.toLocaleString()} rows`;
        } else {
            elements.rowDisplayCount.textContent = `Showing all ${total.toLocaleString()} rows`;
        }
    }
}

function updateFilterStatus() {
    // Count total filter values across all columns
    const filterCount = Object.values(state.filters).reduce((sum, arr) => sum + arr.length, 0);
    
    if (filterCount > 0) {
        elements.filterStatus.textContent = `${filterCount} filter${filterCount > 1 ? 's' : ''} active`;
        elements.filterStatus.style.display = 'inline';
        elements.clearFiltersBtn.style.display = 'flex';
    } else {
        elements.filterStatus.style.display = 'none';
        elements.clearFiltersBtn.style.display = 'none';
    }
}

function clearAllFilters() {
    state.filters = {};
    state.sortColumn = null;
    state.sortDirection = 'asc';
    
    // Clear all filter inputs and tags
    state.columns.forEach((_, idx) => {
        const input = document.getElementById(`filter-input-${idx}`);
        if (input) input.value = '';
        renderFilterTags(idx);
    });
    
    applyFiltersAndSort();
    updateSortIcons();
}

// Expose to global scope for onclick handlers
window.toggleSort = toggleSort;
window.handleFilterKeydown = handleFilterKeydown;
window.addFilterFromInput = addFilterFromInput;
window.removeFilter = removeFilter;

function formatCell(value) {
    if (value === null || value === undefined) {
        return '<span style="color: var(--text-muted)">null</span>';
    }
    if (typeof value === 'number') {
        return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
    }
    return escapeHtml(String(value));
}

// ============================
// Columns Grid
// ============================

function renderColumnsGrid(columns) {
    elements.columnsGrid.innerHTML = columns.map((col, index) => `
        <div class="column-card expanded" data-index="${index}">
            <div class="column-card-header" onclick="toggleColumnCard(${index})">
                <div>
                    <div class="column-name">${escapeHtml(col.name)}</div>
                </div>
                <div class="column-header-right">
                    <span class="column-type">${formatDtype(col.dtype)}</span>
                    ${renderTypeConversionButton(col)}
                </div>
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

function renderTypeConversionButton(col) {
    // Show conversion button for text columns that could be categorical
    if (col.dtype === 'text') {
        const uniqueCount = col.stats?.unique_count || 0;
        if (uniqueCount > 0 && uniqueCount <= 50) {
            return `<button class="convert-type-btn" onclick="event.stopPropagation(); convertColumnType('${escapeHtml(col.name)}', 'categorical')" title="Convert to Categorical">
                → Cat
            </button>`;
        }
    }
    
    // Show conversion button for categorical that might be text
    if (col.dtype === 'categorical') {
        const uniqueCount = col.stats?.unique_count || 0;
        if (uniqueCount > 15) {
            return `<button class="convert-type-btn" onclick="event.stopPropagation(); convertColumnType('${escapeHtml(col.name)}', 'text')" title="Convert to Text">
                → Text
            </button>`;
        }
    }
    
    return '';
}

async function convertColumnType(columnName, newType) {
    if (!state.jobId) {
        alert('No active analysis');
        return;
    }
    
    const confirmMsg = `Convert column "${columnName}" to ${newType}?`;
    if (!confirm(confirmMsg)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/convert-column-type`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: state.jobId,
                column_name: columnName,
                new_type: newType
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Conversion failed');
        }
        
        const result = await response.json();
        console.log('Conversion result:', result);
        
        // Refresh results from server
        await fetchResults();
        
        alert(`Column "${columnName}" converted to ${newType} successfully!`);
        
    } catch (error) {
        console.error('Conversion error:', error);
        alert('Conversion failed: ' + error.message);
    }
}

window.convertColumnType = convertColumnType;

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
    
    if (col.dtype === 'datetime' && stats.is_datetime) {
        // Datetime stats - no mean/median/std
        statsHtml = `
            <div class="stat-item"><span class="stat-label">Earliest</span><span class="stat-value">${stats.min_date || '-'}</span></div>
            <div class="stat-item"><span class="stat-label">Latest</span><span class="stat-value">${stats.max_date || '-'}</span></div>
            <div class="stat-item"><span class="stat-label">Range (days)</span><span class="stat-value">${formatNumber(stats.date_range_days)}</span></div>
            <div class="stat-item"><span class="stat-label">Unique Dates</span><span class="stat-value">${formatNumber(stats.unique_count)}</span></div>
        `;
    } else if (col.dtype === 'identifier' || stats.is_identifier) {
        // ID stats - only min, max, count
        statsHtml = `
            <div class="stat-item full-width"><span class="stat-label">ℹ️ ${stats.message || 'Identifier column'}</span></div>
            <div class="stat-item"><span class="stat-label">Unique Values</span><span class="stat-value">${formatNumber(stats.unique_count)}</span></div>
        `;
        if (stats.min !== undefined) {
            statsHtml += `
                <div class="stat-item"><span class="stat-label">Min ID</span><span class="stat-value">${formatNumber(stats.min)}</span></div>
                <div class="stat-item"><span class="stat-label">Max ID</span><span class="stat-value">${formatNumber(stats.max)}</span></div>
            `;
        }
        if (stats.sample_values) {
            statsHtml += `
                <div class="stat-item full-width"><span class="stat-label">Sample</span><span class="stat-value">${stats.sample_values.join(', ')}</span></div>
            `;
        }
    } else if (col.dtype.startsWith('numeric')) {
        // Numeric stats
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
        // Categorical stats
        statsHtml = `
            <div class="stat-item"><span class="stat-label">Unique</span><span class="stat-value">${stats.unique_count}</span></div>
            ${stats.top_values.slice(0, 5).map(v => 
                `<div class="stat-item"><span class="stat-label">${escapeHtml(String(v.value))}</span><span class="stat-value">${v.count} (${v.percent}%)</span></div>`
            ).join('')}
        `;
    } else if (col.dtype === 'text') {
        // Text stats - show LLM summary
        if (stats.text_summary) {
            statsHtml = `
                <div class="stat-item full-width text-summary-box">
                    <span class="stat-label">📝 Content Summary</span>
                    <p class="text-summary-content">${escapeHtml(stats.text_summary)}</p>
                </div>
                <div class="stat-item"><span class="stat-label">Unique Values</span><span class="stat-value">${formatNumber(stats.unique_count)}</span></div>
                <div class="stat-item"><span class="stat-label">Avg Length</span><span class="stat-value">${formatNumber(stats.avg_length)} chars</span></div>
            `;
        } else {
            statsHtml = `
                <div class="stat-item"><span class="stat-label">Unique</span><span class="stat-value">${formatNumber(stats.unique_count)}</span></div>
                <div class="stat-item"><span class="stat-label">Avg Length</span><span class="stat-value">${formatNumber(stats.avg_length)} chars</span></div>
            `;
            if (stats.sample_values && stats.sample_values.length > 0) {
                statsHtml += `
                    <div class="stat-item full-width"><span class="stat-label">Sample</span><span class="stat-value" style="font-size: 0.75rem">${stats.sample_values.slice(0, 2).map(s => escapeHtml(String(s).substring(0, 40))).join(', ')}</span></div>
                `;
            }
        }
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
    // Filter out single-value columns
    const validColumns = columns.filter(col => {
        const flags = col.quality_flags || [];
        return !flags.includes('single_value');
    });
    
    // Populate axis dropdowns
    const options = validColumns.map(col => 
        `<option value="${escapeHtml(col.name)}" data-dtype="${col.dtype}">${escapeHtml(col.name)} (${formatDtype(col.dtype)})</option>`
    ).join('');
    
    elements.xAxisSelect.innerHTML = '<option value="">-- Select column --</option>' + options;
    elements.yAxisSelect.innerHTML = '<option value="">-- None --</option>' + options;
    elements.colorAxisSelect.innerHTML = '<option value="">-- None --</option>' + options;
    
    // Populate aggregation dropdown
    if (elements.aggregationSelect) {
        elements.aggregationSelect.innerHTML = AGGREGATION_FUNCTIONS.map(agg => 
            `<option value="${agg.value}">${agg.label}</option>`
        ).join('');
    }
    
    // Setup event listeners
    elements.llmSuggestBtn.addEventListener('click', getLLMSuggestion);
    elements.generateCustomBtn.addEventListener('click', generateCustomPlot);
    
    // Reset button
    if (elements.resetBuilderBtn) {
        elements.resetBuilderBtn.addEventListener('click', resetCustomBuilder);
    }
    
    // Show All button under plot
    if (elements.showAllBtn) {
        elements.showAllBtn.addEventListener('click', () => {
            if (elements.showAllToggle) {
                elements.showAllToggle.checked = true;
            }
            generateCustomPlot();
        });
    }
    
    // Show/hide options based on plot type
    document.querySelectorAll('input[name="plot-type"]').forEach(radio => {
        radio.addEventListener('change', updatePlotOptions);
    });
    
    // Update slider display values
    elements.markerSizeInput?.addEventListener('input', () => {
        elements.markerSizeValue.textContent = elements.markerSizeInput.value;
    });
    
    elements.opacityInput?.addEventListener('input', () => {
        elements.opacityValue.textContent = elements.opacityInput.value + '%';
    });
    
    // Update aggregation visibility based on Y column selection
    elements.yAxisSelect?.addEventListener('change', updateAggregationVisibility);
}

function resetCustomBuilder() {
    // Reset dropdowns
    if (elements.xAxisSelect) elements.xAxisSelect.value = '';
    if (elements.yAxisSelect) elements.yAxisSelect.value = '';
    if (elements.colorAxisSelect) elements.colorAxisSelect.value = '';
    if (elements.aggregationSelect) elements.aggregationSelect.value = '';
    
    // Uncheck all plot types
    document.querySelectorAll('input[name="plot-type"]').forEach(radio => {
        radio.checked = false;
    });
    
    // Reset options
    if (elements.binCountInput) elements.binCountInput.value = '20';
    if (elements.showAllToggle) elements.showAllToggle.checked = false;
    if (elements.xMinInput) elements.xMinInput.value = '';
    if (elements.xMaxInput) elements.xMaxInput.value = '';
    if (elements.yMinInput) elements.yMinInput.value = '';
    if (elements.yMaxInput) elements.yMaxInput.value = '';
    if (elements.xTickInput) elements.xTickInput.value = '';
    if (elements.yTickInput) elements.yTickInput.value = '';
    if (elements.markerSizeInput) {
        elements.markerSizeInput.value = '8';
        elements.markerSizeValue.textContent = '8';
    }
    if (elements.opacityInput) {
        elements.opacityInput.value = '70';
        elements.opacityValue.textContent = '70%';
    }
    if (elements.scaleTypeInput) elements.scaleTypeInput.value = 'linear';
    if (elements.gridToggle) elements.gridToggle.checked = true;
    if (elements.customTitleInput) elements.customTitleInput.value = '';
    
    // Hide options section
    if (elements.plotOptionsSection) elements.plotOptionsSection.style.display = 'none';
    
    // Hide AI suggestion
    if (elements.aiSuggestionBox) elements.aiSuggestionBox.classList.add('hidden');
    
    // Reset plot area
    if (elements.customPlotArea) {
        elements.customPlotArea.innerHTML = `
            <div class="plot-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                    <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                </svg>
                <p>Select columns and plot type, then click Generate</p>
            </div>
        `;
    }
    
    // Hide category info
    if (elements.categoryInfo) {
        elements.categoryInfo.classList.add('hidden');
    }
    
    // Clear state
    state.lastPlotResult = null;
}

function updateAggregationVisibility() {
    const aggregationOption = document.getElementById('aggregation-option');
    const yColumn = elements.yAxisSelect?.value;
    
    if (aggregationOption) {
        // Show aggregation when Y column is selected
        aggregationOption.style.display = yColumn ? 'flex' : 'none';
    }
}

function updatePlotOptions() {
    const selectedType = document.querySelector('input[name="plot-type"]:checked')?.value;
    
    if (!selectedType) {
        elements.plotOptionsSection.style.display = 'none';
        return;
    }
    
    // Show options section
    elements.plotOptionsSection.style.display = 'block';
    
    // Show/hide bin count for histogram
    const binOption = document.getElementById('bin-size-option');
    if (binOption) {
        binOption.style.display = selectedType === 'histogram' ? 'flex' : 'none';
    }
    
    // Show/hide show all option for categorical plots
    const showAllOption = document.getElementById('show-all-option');
    const categoricalPlots = ['bar', 'grouped_bar', 'stacked_bar', 'heatmap_cat', 'box_grouped', 'line', 'pie'];
    if (showAllOption) {
        showAllOption.style.display = categoricalPlots.includes(selectedType) ? 'flex' : 'none';
    }
    
    // Show/hide marker size for scatter plots
    const markerOption = document.getElementById('marker-size-option');
    if (markerOption) {
        markerOption.style.display = ['scatter', 'scatter_color', 'bubble'].includes(selectedType) ? 'flex' : 'none';
    }
    
    // Show/hide aggregation for relevant plot types
    const aggregationOption = document.getElementById('aggregation-option');
    if (aggregationOption) {
        const aggPlots = ['bar', 'grouped_bar', 'stacked_bar', 'line', 'heatmap_cat', 'pie'];
        aggregationOption.style.display = aggPlots.includes(selectedType) ? 'flex' : 'none';
    }
    
    // Show/hide Y axis options for single-variable plots
    const yRangeOption = document.getElementById('y-range-option');
    const yTickOption = document.getElementById('y-tick-option');
    const isSingleVar = ['histogram', 'boxplot', 'violin', 'pie', 'wordcloud'].includes(selectedType);
    
    // Y-axis options are now shown for all charts that have Y axis (including histogram for frequency)
    if (yRangeOption) yRangeOption.style.display = 'flex';
    if (yTickOption) yTickOption.style.display = isSingleVar ? 'none' : 'flex';
}

function getPlotOptions() {
    return {
        binCount: parseInt(elements.binCountInput?.value) || 20,
        aggregation: elements.aggregationSelect?.value || null,
        showAll: elements.showAllToggle?.checked || false,
        xRange: {
            min: elements.xMinInput?.value ? parseFloat(elements.xMinInput.value) : null,
            max: elements.xMaxInput?.value ? parseFloat(elements.xMaxInput.value) : null
        },
        yRange: {
            min: elements.yMinInput?.value ? parseFloat(elements.yMinInput.value) : null,
            max: elements.yMaxInput?.value ? parseFloat(elements.yMaxInput.value) : null
        },
        xTick: elements.xTickInput?.value ? parseFloat(elements.xTickInput.value) : null,
        yTick: elements.yTickInput?.value ? parseFloat(elements.yTickInput.value) : null,
        markerSize: parseInt(elements.markerSizeInput?.value) || 8,
        opacity: (parseInt(elements.opacityInput?.value) || 70) / 100,
        scaleType: elements.scaleTypeInput?.value || 'linear',
        showGrid: elements.gridToggle?.checked !== false,
        customTitle: elements.customTitleInput?.value || null
    };
}

async function getLLMSuggestion() {
    const xColumn = elements.xAxisSelect.value;
    const yColumn = elements.yAxisSelect.value;
    const colorColumn = elements.colorAxisSelect.value;
    
    // Build columns list for suggestion
    const selectedColumns = [];
    if (xColumn) {
        const col = state.columns.find(c => c.name === xColumn);
        if (col) selectedColumns.push(col);
    }
    if (yColumn) {
        const col = state.columns.find(c => c.name === yColumn);
        if (col) selectedColumns.push(col);
    }
    if (colorColumn) {
        const col = state.columns.find(c => c.name === colorColumn);
        if (col) selectedColumns.push(col);
    }
    
    if (selectedColumns.length === 0) {
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
                columns: selectedColumns.map(c => c.name)
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
            updatePlotOptions();
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
    const xColumn = elements.xAxisSelect.value;
    const yColumn = elements.yAxisSelect.value;
    const colorColumn = elements.colorAxisSelect.value;
    
    if (!selectedType) {
        alert('Please select a plot type');
        return;
    }
    
    if (!xColumn) {
        alert('Please select at least X axis column');
        return;
    }
    
    const plotType = selectedType.value;
    const options = getPlotOptions();
    
    // Validate column requirements based on plot type
    const twoVarPlots = ['scatter', 'line', 'grouped_bar', 'stacked_bar', 'heatmap_cat', 'box_grouped'];
    const threeVarPlotsRequired = ['bubble']; // Only bubble strictly requires 3
    const threeVarPlotsOptional = ['scatter_color']; // scatter_color works with 2 or 3
    
    if (twoVarPlots.includes(plotType) && !yColumn) {
        alert('Please select Y axis column for this plot type');
        return;
    }
    
    if (threeVarPlotsRequired.includes(plotType) && (!yColumn || !colorColumn)) {
        alert('Please select Y axis and Color/Size column for this plot type');
        return;
    }
    
    elements.generateCustomBtn.disabled = true;
    elements.generateCustomBtn.textContent = 'Generating...';
    
    try {
        const requestBody = {
            job_id: state.jobId,
            plot_type: plotType,
            x_column: xColumn,
            y_column: yColumn || null,
            color_column: colorColumn || null,
            columns: [xColumn, yColumn, colorColumn].filter(Boolean),
            options: options
        };
        
        console.log('Generating plot with:', requestBody);
        
        const response = await fetch(`${API_BASE}/generate-plot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate plot');
        }
        
        const plotData = await response.json();
        state.lastPlotResult = plotData;
        
        // Check if it's a word cloud image
        if (plotData.is_image && plotData.image_base64) {
            elements.customPlotArea.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <h3 style="color: var(--text-primary); margin-bottom: 15px;">Word Cloud: ${escapeHtml(xColumn)}</h3>
                    <img src="data:image/png;base64,${plotData.image_base64}" 
                         alt="Word Cloud" 
                         style="max-width: 100%; height: auto; border-radius: 8px;">
                </div>
            `;
            elements.categoryInfo.classList.add('hidden');
            return;
        }
        
        // Apply custom options to layout
        const customLayout = { ...PLOTLY_LAYOUT, ...plotData.plot_layout };
        
        // Apply user customizations
        if (options.customTitle) {
            customLayout.title = options.customTitle;
        }
        
        if (options.xRange.min !== null || options.xRange.max !== null) {
            customLayout.xaxis = customLayout.xaxis || {};
            customLayout.xaxis.range = [
                options.xRange.min ?? customLayout.xaxis.range?.[0],
                options.xRange.max ?? customLayout.xaxis.range?.[1]
            ].filter(v => v !== undefined);
            if (customLayout.xaxis.range.length === 0) delete customLayout.xaxis.range;
        }
        
        if (options.yRange.min !== null || options.yRange.max !== null) {
            customLayout.yaxis = customLayout.yaxis || {};
            customLayout.yaxis.range = [
                options.yRange.min ?? customLayout.yaxis.range?.[0],
                options.yRange.max ?? customLayout.yaxis.range?.[1]
            ].filter(v => v !== undefined);
            if (customLayout.yaxis.range.length === 0) delete customLayout.yaxis.range;
        }
        
        if (options.xTick) {
            customLayout.xaxis = customLayout.xaxis || {};
            customLayout.xaxis.dtick = options.xTick;
        }
        
        if (options.yTick) {
            customLayout.yaxis = customLayout.yaxis || {};
            customLayout.yaxis.dtick = options.yTick;
        }
        
        if (options.scaleType === 'log') {
            customLayout.yaxis = customLayout.yaxis || {};
            customLayout.yaxis.type = 'log';
        }
        
        if (!options.showGrid) {
            customLayout.xaxis = customLayout.xaxis || {};
            customLayout.yaxis = customLayout.yaxis || {};
            customLayout.xaxis.showgrid = false;
            customLayout.yaxis.showgrid = false;
        }
        
        // Apply marker customizations to data
        const customData = plotData.plot_data.map(trace => {
            if (trace.marker) {
                trace.marker.size = trace.marker.size || options.markerSize;
                trace.marker.opacity = trace.marker.opacity || options.opacity;
            }
            if (trace.opacity !== undefined) {
                trace.opacity = options.opacity;
            }
            return trace;
        });
        
        // Create scrollable container if chart is wide
        const chartWidth = customLayout.width || 800;
        const containerHtml = chartWidth > 900 
            ? `<div class="plot-scroll-container"><div id="custom-plot-render" style="width: ${chartWidth}px; height: 500px;"></div></div>`
            : `<div id="custom-plot-render" style="width: 100%; height: 500px;"></div>`;
        
        elements.customPlotArea.innerHTML = containerHtml;
        
        // Set responsive false if we have a custom width
        const plotConfig = { ...PLOTLY_CONFIG };
        if (chartWidth > 900) {
            plotConfig.responsive = false;
        }
        
        Plotly.newPlot('custom-plot-render', customData, customLayout, plotConfig);
        
        // Show category info if applicable
        if (plotData.total_categories && plotData.showing_categories && 
            plotData.total_categories > plotData.showing_categories) {
            elements.categoryInfoText.textContent = 
                `Showing ${plotData.showing_categories} of ${plotData.total_categories} categories`;
            elements.categoryInfo.classList.remove('hidden');
        } else {
            elements.categoryInfo.classList.add('hidden');
        }
        
    } catch (error) {
        console.error('Generate error:', error);
        alert('Failed to generate plot: ' + error.message);
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
// State Reset
// ============================

function clearPreviousResults() {
    // Clear state (but keep files)
    state.jobId = null;
    state.results = null;
    state.columns = [];
    state.selectedPlotColumn = null;
    state.customSelectedColumns = [];
    state.fullData = [];
    state.filteredData = [];
    state.filters = {};
    state.sortColumn = null;
    state.sortDirection = 'asc';
    state.lastPlotResult = null;
    
    // Clear results UI
    if (elements.summaryList) elements.summaryList.innerHTML = '';
    if (elements.insightsList) elements.insightsList.innerHTML = '';
    if (elements.limitationsList) elements.limitationsList.innerHTML = '';
    
    // Clear data table
    if (elements.dataTable) {
        const thead = elements.dataTable.querySelector('thead');
        const tbody = elements.dataTable.querySelector('tbody');
        if (thead) thead.innerHTML = '';
        if (tbody) tbody.innerHTML = '';
    }
    
    // Clear columns grid
    if (elements.columnsGrid) elements.columnsGrid.innerHTML = '';
    
    // Clear plots tab
    if (elements.plotsColumnSelector) elements.plotsColumnSelector.innerHTML = '';
    if (elements.plotsDisplay) {
        elements.plotsDisplay.innerHTML = '';
        elements.plotsDisplay.classList.remove('active');
    }
    
    // Reset custom builder
    resetCustomBuilder();
    
    // Reset tabs to first tab
    elements.tabBtns.forEach(btn => btn.classList.remove('active'));
    document.querySelector('.tab-btn[data-tab="data"]')?.classList.add('active');
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    document.getElementById('data-tab')?.classList.add('active');
    
    // Clear any Plotly plots to free memory
    const plotContainers = document.querySelectorAll('[id^="col-plot-"], [id^="auto-plot-"], #custom-plot-render');
    plotContainers.forEach(container => {
        if (container && typeof Plotly !== 'undefined') {
            Plotly.purge(container);
        }
    });
}

function resetAppState() {
    // Clear all results first
    clearPreviousResults();
    
    // Also clear file inputs
    state.csvFile = null;
    state.schemaFile = null;
    
    elements.csvZone.classList.remove('has-file');
    elements.schemaZone.classList.remove('has-file');
    elements.csvFilename.textContent = '';
    elements.schemaFilename.textContent = '';
    elements.csvInput.value = '';
    elements.schemaInput.value = '';
    updateAnalyzeButton();
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
    console.log('Initializing EDA Insights...');
    
    // Check if all elements exist
    console.log('Elements check:');
    console.log('- csvZone:', elements.csvZone);
    console.log('- analyzeBtn:', elements.analyzeBtn);
    
    // Setup file uploads
    setupFileUpload(elements.csvZone, elements.csvInput, elements.csvFilename, 'csv');
    setupFileUpload(elements.schemaZone, elements.schemaInput, elements.schemaFilename, 'schema');
    
    // Analyze button
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', () => {
            console.log('Analyze button clicked!');
            uploadAndAnalyze();
        });
        console.log('Analyze button listener attached');
    } else {
        console.error('Analyze button not found!');
    }
    
    // Back button
    elements.backBtn.addEventListener('click', () => {
        resetAppState();
        showScreen('upload-screen');
    });
    
    // Clear filters button
    if (elements.clearFiltersBtn) {
        elements.clearFiltersBtn.addEventListener('click', clearAllFilters);
    }
    
    // Tabs
    setupTabs();
    
    console.log('EDA Insights initialized');
}

// Start app
document.addEventListener('DOMContentLoaded', init);