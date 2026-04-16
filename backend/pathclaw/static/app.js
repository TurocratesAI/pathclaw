        const API = window.location.origin;
        let sessionId = '';
        let sessionTitle = '';
        let sessionDrawerOpen = false;
        let sending = false;
        let activeJobId = sessionStorage.getItem('activeJobId') || null;
        let jobPollInterval = null;

        // === Error toast ===
        function showError(message, duration = 5000) {
            const t = document.createElement('div');
            t.className = 'error-toast';
            t.textContent = message;
            document.body.appendChild(t);
            setTimeout(() => t.remove(), duration);
        }

        // Fetch helper that throws a readable Error on non-2xx (prevents
        // "Unexpected token 'I'" when the server returns HTML for 500s).
        async function apiJson(url, opts) {
            const r = await fetch(url, opts);
            if (!r.ok) {
                let msg = `HTTP ${r.status}`;
                try {
                    const txt = await r.text();
                    try { msg = (JSON.parse(txt).detail || txt).toString(); }
                    catch { msg = txt; }
                } catch {}
                throw new Error(msg.slice(0, 400));
            }
            return r.json();
        }

        // Modal replacement for window.confirm — returns a Promise<boolean>.
        function confirmModal(message) {
            return new Promise(resolve => {
                const ov = document.createElement('div');
                ov.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;display:flex;align-items:center;justify-content:center';
                ov.innerHTML = `<div style="background:var(--bg,#1a1a1a);color:var(--fg,#eee);padding:20px;border-radius:8px;max-width:420px;box-shadow:0 8px 32px rgba(0,0,0,.5)"><div style="margin-bottom:16px;white-space:pre-line">${message.replace(/</g,'&lt;')}</div><div style="display:flex;gap:8px;justify-content:flex-end"><button id="_cm_no" style="padding:6px 14px">Cancel</button><button id="_cm_yes" style="padding:6px 14px;background:#c44;color:white;border:0;border-radius:4px">OK</button></div></div>`;
                document.body.appendChild(ov);
                ov.querySelector('#_cm_no').onclick = () => { ov.remove(); resolve(false); };
                ov.querySelector('#_cm_yes').onclick = () => { ov.remove(); resolve(true); };
                ov.onclick = (e) => { if (e.target === ov) { ov.remove(); resolve(false); } };
            });
        }

        // === Onboarding ===
        // === Provider card switching ===
        const PROVIDER_MODELS = {
            ollama: [],   // dynamic from /api/status/llm
            anthropic: ['claude-sonnet-4-6', 'claude-opus-4-20250514', 'claude-haiku-4-5-20251001'],
            openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'o4-mini', 'o3-mini'],
            google: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-lite'],
        };

        function _setActiveProvider(p) {
            el('providerSelect').value = p;
            document.querySelectorAll('.pcard').forEach(c => c.classList.toggle('active', c.dataset.p === p));
            el('anthropicRow').style.display = p === 'anthropic' ? '' : 'none';
            const obr = el('openaiBaseRow'); if (obr) obr.style.display = p === 'openai' ? '' : 'none';
            const olr = el('ollamaBaseRow'); if (olr) olr.style.display = p === 'ollama' ? '' : 'none';
            el('openaiRow').style.display = p === 'openai' ? '' : 'none';
            el('googleRow').style.display = p === 'google' ? '' : 'none';
            el('ollamaRow').style.display = p === 'ollama' ? '' : 'none';
        }

        async function _loadModelsForProvider(p, current) {
            const sel = el('modelSelect');
            sel.innerHTML = '';
            let models = PROVIDER_MODELS[p] || [];
            if (p === 'ollama') {
                try {
                    const r = await fetch(`${API}/api/status/llm`);
                    const d = await r.json();
                    if (d.models && d.models.length) models = d.models;
                } catch (e) { models = []; }
            }
            if (!models.length) {
                sel.innerHTML = '<option value="">— no models found —</option>';
                return;
            }
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m; opt.textContent = m;
                if (m === current) opt.selected = true;
                sel.appendChild(opt);
            });
        }

        document.querySelectorAll('.pcard').forEach(c => {
            c.onclick = async () => {
                const p = c.dataset.p;
                _setActiveProvider(p);
                await _loadModelsForProvider(p, '');
            };
        });

        const DISCLAIMER_VERSION = 1;

        function _showDisclaimerStep() {
            el('disclaimerStep').style.display = 'block';
            el('credentialsStep').style.display = 'none';
            el('onboardModal').classList.remove('hidden');
            const cb = el('disclaimerCheck');
            const btn = el('disclaimerContinue');
            cb.checked = false;
            btn.disabled = true;
            cb.onchange = () => { btn.disabled = !cb.checked; };
            btn.onclick = async () => {
                if (!cb.checked) return;
                await fetch(`${API}/api/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        disclaimer_acknowledged: true,
                        disclaimer_version: DISCLAIMER_VERSION,
                        disclaimer_at: new Date().toISOString(),
                    }),
                }).catch(() => {});
                // Proceed to credentials step or close if already onboarded
                try {
                    const d = await fetch(`${API}/api/config`).then(r => r.json());
                    if (!d.onboarding_complete) {
                        _showCredentialsStep();
                    } else {
                        el('onboardModal').classList.add('hidden');
                        initSession();
                    }
                } catch {
                    _showCredentialsStep();
                }
            };
        }

        async function _showCredentialsStep() {
            el('disclaimerStep').style.display = 'none';
            el('credentialsStep').style.display = 'block';
            el('modalTitle').textContent = 'Welcome to PathClaw';
            el('modalSub').textContent = 'Set up your credentials to get started.';
            el('onboardModal').classList.remove('hidden');
            const lr = await fetch(`${API}/api/status/llm`).then(r => r.json()).catch(() => ({}));
            _setActiveProvider(lr.provider || 'ollama');
            await _loadModelsForProvider(lr.provider || 'ollama', lr.model || '');
        }

        async function checkOnboard() {
            try {
                const r = await fetch(`${API}/api/config`);
                const d = await r.json();
                el('sHf').textContent = d.huggingface_token_set ? 'Set' : 'Not set';
                el('sHf').className = d.huggingface_token_set ? 'val ok' : 'val warn';
                const needDisclaimer = !d.disclaimer_acknowledged || (d.disclaimer_version || 0) < DISCLAIMER_VERSION;
                if (needDisclaimer) {
                    _showDisclaimerStep();
                } else if (!d.onboarding_complete) {
                    await _showCredentialsStep();
                } else {
                    initSession();
                }
            } catch (e) { el('sHf').textContent = '—'; initSession(); }
        }

        async function openSettings() {
            el('disclaimerStep').style.display = 'none';
            el('credentialsStep').style.display = 'block';
            el('modalTitle').textContent = 'Settings';
            el('modalSub').textContent = 'Update API keys, switch LLM provider or model.';
            el('skipBtn').textContent = 'Cancel';
            el('saveBtn').textContent = 'Save & Apply';
            // Populate current values
            try {
                const [cfg, llm] = await Promise.all([
                    fetch(`${API}/api/config`).then(r => r.json()),
                    fetch(`${API}/api/status/llm`).then(r => r.json()),
                ]);
                const keysSet = llm.keys_set || {};
                el('anthropicStatus').textContent = keysSet.anthropic ? '✓ key set' : '';
                el('anthropicStatus').className = 'key-status' + (keysSet.anthropic ? ' ok' : '');
                el('openaiStatus').textContent = keysSet.openai ? '✓ key set' : '';
                el('openaiStatus').className = 'key-status' + (keysSet.openai ? ' ok' : '');
                el('googleStatus').textContent = keysSet.google ? '✓ key set' : '';
                el('googleStatus').className = 'key-status' + (keysSet.google ? ' ok' : '');
                _setActiveProvider(llm.provider || 'ollama');
                await _loadModelsForProvider(llm.provider || 'ollama', llm.model || '');
            } catch (e) { }
            el('onboardModal').classList.remove('hidden');
        }

        el('saveBtn').onclick = async () => {
            const isOnboard = el('modalTitle').textContent === 'Welcome to PathClaw';
            const hf = el('hfInput').value.trim();
            if (isOnboard && !hf) { el('hfInput').style.borderColor = 'var(--red)'; return; }
            const provider = el('providerSelect').value;
            const customModel = (el('modelCustom')?.value || '').trim();
            const dropdownModel = el('modelSelect').value;
            const body = {
                huggingface_token: hf,
                gdc_token: el('gdcInput').value.trim(),
                llm_provider: provider,
                llm_model: customModel || dropdownModel,
                anthropic_api_key: el('anthropicInput').value.trim(),
                openai_api_key: el('openaiInput').value.trim(),
                google_api_key: el('googleInput').value.trim(),
                openai_base: (el('openaiBaseInput')?.value || '').trim(),
                ollama_base: (el('ollamaBaseInput')?.value || '').trim(),
            };
            await fetch(`${API}/api/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            el('onboardModal').classList.add('hidden');
            if (hf) { el('sHf').textContent = 'Set'; el('sHf').className = 'val ok'; }
            // Reset button labels
            el('skipBtn').textContent = 'Skip';
            el('saveBtn').textContent = 'Save & Apply';
            checkOllama();
            if (isOnboard) initSession();
        };
        el('skipBtn').onclick = async () => {
            await fetch(`${API}/api/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }).catch(() => { });
            el('onboardModal').classList.add('hidden');
            el('skipBtn').textContent = 'Skip';
            initSession();
        };

        // === Status ===
        async function checkStatus() {
            try {
                const r = await fetch(`${API}/api/status`); const d = await r.json();
                el('connPill').className = 'pill ok'; el('connPill').querySelector('span').textContent = 'Online';
                el('sBackend').textContent = 'Online'; el('sBackend').className = 'val ok';
                if (d.gpu.available) {
                    el('sGpu').textContent = `${d.gpu.name} ×${d.gpu.count}`; el('sGpu').className = 'val ok';
                    el('gpuTag').textContent = d.gpu.name;
                } else { el('sGpu').textContent = 'CPU only'; el('sGpu').className = 'val warn'; el('gpuTag').textContent = 'CPU'; }
                el('sStorage').textContent = `${d.storage.free_gb} GB free / ${d.storage.total_gb} GB`;
            } catch (e) {
                el('connPill').className = 'pill err'; el('connPill').querySelector('span').textContent = 'Offline';
                el('sBackend').textContent = 'Offline'; el('sBackend').className = 'val err';
            }
        }
        async function checkOllama() {
            try {
                const r = await fetch(`${API}/api/status/llm`);
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const d = await r.json();
                const provider = d.provider || 'ollama';
                const model = d.model || '—';
                el('sOllama').textContent = `${provider}/${model}`;
                el('sOllama').className = d.online ? 'val ok' : 'val warn';
            } catch (e) { el('sOllama').textContent = 'Offline'; el('sOllama').className = 'val err'; }
        }

        // === Sidebar: Datasets (legacy hidden element kept for compat) ===
        async function loadDs() {
            try {
                await renderFileTree();
            } catch (e) { showError('Failed to load datasets'); }
        }

        // === Sidebar: Experiments (legacy hidden element kept for compat) ===
        async function loadExp() {
            try {
                await renderFileTree();
            } catch (e) { showError('Failed to load experiments'); }
        }

        // === VS Code-style File Tree ===
        // Top-level sections + individual item open state
        let _treeExpanded = { datasets: true, experiments: true };
        let _treeItemOpen = {}; // persists dataset/experiment expand state across re-renders

        function togTree(section) {
            _treeExpanded[section] = !_treeExpanded[section];
            const body = el(`tree-${section}`), chev = el(`tchev-${section}`);
            if (body) body.style.display = _treeExpanded[section] ? 'block' : 'none';
            if (chev) chev.textContent = _treeExpanded[section] ? '▼' : '▶';
        }

        function togTreeItem(id) {
            const body = el(`ti-${id}`), chev = el(`tichev-${id}`);
            if (!body) return;
            const open = body.style.display !== 'none';
            _treeItemOpen[id] = !open;
            body.style.display = open ? 'none' : 'block';
            if (chev) chev.textContent = open ? '▶' : '▼';
        }

        function openSlidesTab(datasetId) {
            const tab = document.querySelector('.ws-tab[data-t="slides"]');
            if (tab) tab.click();
        }

        async function openDatasetInViewer(datasetId) {
            // Open the first slide in the dataset in the viewer
            const sr = await fetch(`${API}/api/datasets/${datasetId}/slides`).then(r => r.json()).catch(() => ({ slides: [] }));
            const slides = sr.slides || [];
            if (!slides.length) { showError('No slides found in this dataset'); return; }
            const stem = slides[0].filename.replace(/\.[^.]+$/, '');
            openSlideInViewer(datasetId, stem);
        }

        let _lastTreeHash = null;
        async function renderFileTree(force = false) {
            const treeEl = el('fileTree');
            if (!treeEl) return;

            // Sessions are isolated: each session is a parallel PhD student with their own
            // datasets, experiments, workspace, and jobs. We pass session_id to the list APIs
            // so the sidebar only shows what belongs to the current session.
            // (Empty sessionId before initSession resolves → sidebar stays empty until the
            //  first session is established, which is correct behavior.)
            const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            const [dsRes, expRes] = await Promise.all([
                fetch(`${API}/api/datasets${sidQ}`).then(r => r.json()).catch(() => ({ datasets: [] })),
                fetch(`${API}/api/artifacts${sidQ}`).then(r => r.json()).catch(() => ({ artifacts: [] }))
            ]);
            const ds = dsRes.datasets || [], exps = expRes.artifacts || [];

            // Fetch experiment statuses for colored dots (best-effort)
            const statuses = {};
            await Promise.all(exps.map(async a => {
                try {
                    const r = await fetch(`${API}/api/training/${a.experiment_id}`);
                    if (r.ok) statuses[a.experiment_id] = await r.json();
                } catch {}
            }));

            // Skip full DOM rewrite if nothing relevant changed — avoids flicker & preserves
            // scroll position / expanded state across the 30s poll.
            // Hash only the fields that should trigger a re-render.
            const hashSource = JSON.stringify({
                sid: sessionId,
                ds: ds.map(d => ({
                    id: d.id, name: d.name, sc: d.slide_count, fc: d.feature_count,
                    sl: (d.slides || []).map(s => s.filename),
                    g: d.genomics ? Object.keys(d.genomics).sort() : null,
                })),
                ex: exps.map(a => {
                    const s = statuses[a.experiment_id] || {};
                    return { id: a.experiment_id, st: s.status, acc: s.metrics?.best_val_accuracy };
                }),
                open: _treeExpanded,
            });
            let hash = 0;
            for (let i = 0; i < hashSource.length; i++) {
                hash = ((hash << 5) - hash + hashSource.charCodeAt(i)) | 0;
            }
            if (!force && hash === _lastTreeHash) return;
            _lastTreeHash = hash;

            const statusColor = st => ({ completed: '#22c55e', failed: '#ef4444', running: '#3b82f6', queued: '#f59e0b' }[st] || '#71717a');
            const dsOpen = _treeExpanded.datasets, expOpen = _treeExpanded.experiments;

            // Datasets section
            let dsNodes = '<div class="tree-empty">No datasets registered</div>';
            if (ds.length) {
                dsNodes = ds.map(d => {
                    const sid = 'ds_' + d.id.replace(/[^a-zA-Z0-9]/g, '_');
                    const slideCount = d.slide_count || 0, featCount = d.feature_count || 0;
                    const slides = d.slides || [];
                    // Default: open if only 1 dataset, otherwise remember last state
                    const isOpen = sid in _treeItemOpen ? _treeItemOpen[sid] : (ds.length === 1);
                    _treeItemOpen[sid] = isOpen; // persist default
                    const slideRows = slides.map(s => {
                        const stem = s.filename.replace(/\.[^.]+$/, '');
                        return `<div class="tree-leaf tree-slide-row" onclick="openSlideInViewer('${d.id}','${stem}')" title="${s.filename}">
                            <span class="tree-slide-icon2"></span>
                            <span class="tree-slide-name">${s.filename}</span>
                            <span class="tree-meta" style="margin-left:auto;flex-shrink:0">${s.size_mb ? s.size_mb + ' MB' : ''}</span>
                        </div>`;
                    }).join('');
                    // Genomics block (MAF, clinical, expression, cnv, labels) — attached to tcga-* cohorts
                    let genomicsRows = '';
                    if (d.genomics) {
                        const g = d.genomics;
                        const subIcon = { maf: '🧬', clinical: '📋', expression: '📊', cnv: '🔢' };
                        for (const subName of ['maf', 'clinical', 'expression', 'cnv']) {
                            const sub = g[subName];
                            if (!sub) continue;
                            const gsid = sid + '_' + subName;
                            const gopen = !!_treeItemOpen[gsid];
                            const sampleRows = (sub.sample || []).map(fn =>
                                `<div class="tree-leaf" title="${fn}" style="font-size:11px;opacity:.75;padding-left:24px">${fn}</div>`
                            ).join('');
                            const moreRow = sub.count > (sub.sample || []).length
                                ? `<div class="tree-leaf" style="font-size:11px;opacity:.5;padding-left:24px">… and ${sub.count - sub.sample.length} more</div>` : '';
                            genomicsRows += `<div class="tree-item-wrap" style="margin-left:4px;margin-top:2px">
                                <div class="tree-item-head" onclick="togTreeItem('${gsid}')">
                                    <span id="tichev-${gsid}" class="tree-chev">${gopen ? '▼' : '▶'}</span>
                                    <span>${subIcon[subName]}</span>
                                    <span class="tree-item-lbl">${subName}</span>
                                    <span class="tree-badge">${sub.count}</span>
                                    <span class="tree-meta" style="margin-left:auto">${sub.size_mb} MB</span>
                                </div>
                                <div id="ti-${gsid}" style="display:${gopen ? 'block' : 'none'}" class="tree-item-body">
                                    ${sampleRows}${moreRow}
                                </div>
                            </div>`;
                        }
                        for (const lab of (g.labels || [])) {
                            const escPath = lab.path.replace(/'/g, "\\'");
                            genomicsRows += `<div class="tree-leaf tree-action" style="margin-left:4px" onclick="openCsvViewer('${escPath}')" title="${lab.path}">
                                <span>🏷️</span><span class="tree-item-lbl">${lab.name}</span>
                                <span class="tree-meta" style="margin-left:auto">${lab.size_kb} KB</span>
                            </div>`;
                        }
                        if (genomicsRows) {
                            genomicsRows = `<div style="margin-top:4px;border-top:1px solid var(--border);padding-top:4px"></div>` + genomicsRows;
                        }
                    }
                    return `<div class="tree-item-wrap">
                        <div class="tree-item-head" onclick="togTreeItem('${sid}')">
                            <span id="tichev-${sid}" class="tree-chev">${isOpen ? '▼' : '▶'}</span>
                            <span class="tree-folder-icon">📁</span>
                            <span class="tree-item-lbl">${d.name}</span>
                            <span class="tree-badge" style="margin-left:4px">${slideCount}</span>
                        </div>
                        <div id="ti-${sid}" style="display:${isOpen ? 'block' : 'none'}" class="tree-item-body">
                            ${slideRows || `<div class="tree-leaf"><span class="tree-meta">No slides found — check path</span></div>`}
                            ${genomicsRows}
                            <div class="tree-leaf tree-action" style="margin-top:4px;border-top:1px solid var(--border);padding-top:4px" onclick="sendFromUI('Preprocess dataset ${d.name} with patch size 256 at 20x')"><span class="tree-icon-sm">⚙️</span>Preprocess</div>
                            ${featCount > 0 ? `<div class="tree-leaf tree-action" onclick="sendFromUI('Train a model on ${d.name}')"><span class="tree-icon-sm">🚀</span>Train model</div>` : ''}
                        </div>
                    </div>`;
                }).join('');
            }

            // Experiments section
            let expNodes = '<div class="tree-empty">No experiments yet</div>';
            if (exps.length) {
                expNodes = exps.slice().reverse().map(a => {
                    const s = statuses[a.experiment_id] || {};
                    const dot = statusColor(s.status);
                    const acc = s.metrics?.best_val_accuracy;
                    const accStr = acc !== undefined ? `${(acc * 100).toFixed(1)}%` : '';
                    const sid = 'exp_' + a.experiment_id.replace(/[^a-zA-Z0-9]/g, '_');
                    const isOpen = !!_treeItemOpen[sid];
                    return `<div class="tree-item-wrap">
                        <div class="tree-item-head" onclick="togTreeItem('${sid}')">
                            <span id="tichev-${sid}" class="tree-chev">${isOpen ? '▼' : '▶'}</span>
                            <span class="tree-dot" style="background:${dot}"></span>
                            <span class="tree-item-lbl mono">${a.experiment_id}</span>
                            ${accStr ? `<span class="tree-acc">${accStr}</span>` : ''}
                        </div>
                        <div id="ti-${sid}" style="display:${isOpen ? 'block' : 'none'}" class="tree-item-body">
                            ${s.config?.mil_method ? `<div class="tree-leaf"><span class="tree-meta">${s.config.mil_method} · ${s.config.feature_backbone || '—'}</span></div>` : ''}
                            <div class="tree-leaf tree-action" onclick="viewExperiment('${a.experiment_id}')"><span class="tree-exp-icon">📈</span>View plots</div>
                            <div class="tree-leaf tree-action" onclick="document.querySelector('.ws-tab[data-t=\\"logs\\"]').click();setTimeout(()=>loadLog('${a.experiment_id}'),100)"><span class="tree-icon-sm">📄</span>View logs</div>
                            <div class="tree-leaf tree-action" onclick="window.location.href='${API}/api/artifacts/${a.experiment_id}/export'"><span class="tree-icon-sm">📦</span>Export .zip (model + config + plots)</div>
                        </div>
                    </div>`;
                }).join('');
            }

            treeEl.innerHTML = `
                <div class="tree-section">
                    <div class="tree-sect-head" onclick="togTree('datasets')">
                        <span id="tchev-datasets" class="tree-chev">${dsOpen ? '▼' : '▶'}</span>
                        <span class="tree-sect-lbl">Datasets</span>
                        <span class="tree-badge">${ds.length}</span>
                    </div>
                    <div id="tree-datasets" style="display:${dsOpen ? 'block' : 'none'}">${dsNodes}</div>
                </div>
                <div class="tree-section">
                    <div class="tree-sect-head" onclick="togTree('experiments')">
                        <span id="tchev-experiments" class="tree-chev">${expOpen ? '▼' : '▶'}</span>
                        <span class="tree-sect-lbl">Experiments</span>
                        <span class="tree-badge">${exps.length}</span>
                    </div>
                    <div id="tree-experiments" style="display:${expOpen ? 'block' : 'none'}">${expNodes}</div>
                </div>`;
        }

        function viewExperiment(jobId) {
            activeJobId = jobId;
            sessionStorage.setItem('activeJobId', jobId);
            document.querySelector('.ws-tab[data-t="plots"]').click();
        }

        // === Job polling ===
        function startJobPolling(jobId) {
            activeJobId = jobId;
            sessionStorage.setItem('activeJobId', jobId);
            if (jobPollInterval) clearInterval(jobPollInterval);
            jobPollInterval = setInterval(() => _pollJob(jobId), 3000);
            _pollJob(jobId);
        }

        async function _pollJob(jobId) {
            try {
                const r = await fetch(`${API}/api/training/${jobId}`);
                if (!r.ok) return;
                const j = await r.json();
                _updateJobCard(j);
                // Only force a tree re-render on state transitions, not every 3s.
                if (j.status === 'completed' || j.status === 'failed') {
                    clearInterval(jobPollInterval);
                    jobPollInterval = null;
                    renderFileTree(true);
                }
            } catch (e) { console.warn('Job poll failed:', e.message); }
        }

        function _updateJobCard(j) {
            const card = el('liveJobCard');
            if (!card) return;
            const pct = Math.round((j.progress || 0) * 100);
            const m = j.metrics || {};
            card.className = `job-card ${j.status}`;
            card.querySelector('.job-badge').className = `job-badge ${j.status}`;
            card.querySelector('.job-badge').textContent = j.status;
            card.querySelector('.progress-fill').style.width = `${pct}%`;
            card.querySelector('.jep').textContent = `${j.epoch || 0} / ${j.total_epochs || '?'}`;
            card.querySelector('.jpct').textContent = `${pct}%`;
            card.querySelector('.jtl').textContent = m.train_loss ? m.train_loss.toFixed(4) : '—';
            card.querySelector('.jvl').textContent = m.val_loss ? m.val_loss.toFixed(4) : '—';
            card.querySelector('.jacc').textContent = m.best_val_accuracy ? `${(m.best_val_accuracy * 100).toFixed(1)}%` : '—';
            // Show errors if any
            const errEl = card.querySelector('.job-errors');
            if (errEl) errEl.textContent = (j.errors || []).slice(-1)[0] || '';
        }

        // === Markdown renderer ===
        // Configure marked once on load
        if (typeof marked !== 'undefined') {
            marked.use({ gfm: true, breaks: true });
        }

        // KaTeX auto-render: walks an element, rendering $...$ and $$...$$ in-place.
        function _renderMath(el) {
            if (!el || typeof renderMathInElement === 'undefined') return;
            try {
                renderMathInElement(el, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\(', right: '\\)', display: false },
                    ],
                    throwOnError: false,
                    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                });
            } catch (e) { /* ignore */ }
        }

        function formatMd(text) {
            if (typeof marked !== 'undefined') {
                return marked.parse(text);
            }
            // Fallback (no CDN): minimal renderer with table support
            let s = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            s = s.replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
            s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
            s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // Tables
            s = s.replace(/((?:^\|.+\|\n?)+)/gm, (match) => {
                const rows = match.trim().split('\n');
                if (rows.length < 2 || !/^\|[\s\-:|]+\|/.test(rows[1])) return match;
                const ths = rows[0].split('|').slice(1,-1).map(c => `<th>${c.trim()}</th>`).join('');
                const trs = rows.slice(2).map(r => {
                    const tds = r.split('|').slice(1,-1).map(c => `<td>${c.trim()}</td>`).join('');
                    return `<tr>${tds}</tr>`;
                }).join('');
                return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
            });
            s = s.replace(/^### (.+)$/gm, '<h4>$1</h4>');
            s = s.replace(/^## (.+)$/gm, '<h3>$1</h3>');
            s = s.replace(/^# (.+)$/gm, '<h2>$1</h2>');
            s = s.replace(/((?:^[-*] .+\n?)+)/gm, m => '<ul>' + m.replace(/^[-*] (.+)$/gm, '<li>$1</li>') + '</ul>');
            s = s.replace(/^---+$/gm, '<hr>');
            // Don't convert newlines inside pre blocks
            const parts = s.split(/(<pre[\s\S]*?<\/pre>)/g);
            return parts.map((p, i) => i % 2 === 0 ? p.replace(/\n/g, '<br>') : p).join('');
        }

        // === Chat ===
        function el(id) { return document.getElementById(id) }
        function addMsg(html, type = 'agent', tools = []) {
            const c = el('msgs'), d = document.createElement('div');
            d.className = `msg ${type}`;
            const who = type === 'user' ? 'You' : 'PathClaw';
            let toolHtml = tools.length ? '<div class="tool-row">' + tools.map(t => `<span class="tool-badge">${t}</span>`).join('') + '</div>' : '';
            d.innerHTML = `<div class="who">${who}</div>${type === 'agent' ? formatMd(html) : html.replace(/\n/g, '<br>')}${toolHtml}`;
            c.appendChild(d); c.scrollTop = c.scrollHeight;
            if (type === 'agent') _renderMath(d);
        }
        function showTyping() {
            const d = document.createElement('div'); d.className = 'typing'; d.id = 'typingIndicator';
            d.innerHTML = '<span></span><span></span><span></span>';
            el('msgs').appendChild(d); el('msgs').scrollTop = el('msgs').scrollHeight;
        }
        function hideTyping() { const t = el('typingIndicator'); if (t) t.remove(); }

        let _abortController = null;
        el('stopBtn').onclick = () => { if (_abortController) { _abortController.abort(); } };

        async function sendMessage(text) {
            if (sending) return;
            sending = true;
            el('sendBtn').disabled = true;
            el('sendBtn').style.display = 'none';
            el('stopBtn').style.display = '';
            addMsg(text, 'user');
            showTyping();
            _abortController = new AbortController();
            let agentBubble = null, agentContent = '', toolDetails = [], typingRemoved = false;
            function ensureBubble() {
                if (agentBubble) return;
                if (!typingRemoved) { hideTyping(); typingRemoved = true; }
                const c = el('msgs');
                agentBubble = document.createElement('div');
                agentBubble.className = 'msg agent';
                agentBubble.innerHTML = '<div class="who">PathClaw</div><div class="body"></div><div class="tools"></div>';
                c.appendChild(agentBubble);
            }
            function updateBubble(thinking = false) {
                if (!agentBubble) return;
                const bodyEl = agentBubble.querySelector('.body');
                bodyEl.innerHTML = formatMd(agentContent);
                _renderMath(bodyEl);
                const pending = toolDetails.filter(t => !t.result);
                agentBubble.querySelector('.tools').innerHTML = toolDetails.map((t, i) => {
                    const isLoading = !t.result;
                    const spinner = isLoading ? `<i class="tool-spinner">↻</i>` : '';
                    const durationTag = t.duration_ms ? ` <span style="opacity:0.6;font-size:9px">${t.duration_ms}ms</span>` : '';
                    const badgeCls = isLoading ? 'tool-badge loading' : 'tool-badge';
                    const hasDetail = t.args || t.result;
                    if (!hasDetail) return `<span class="${badgeCls}">${spinner}${t.name}</span>`;
                    return `<div class="tool-detail" onclick="this.classList.toggle('expanded')">
                        <span class="${badgeCls}">${spinner}${t.name}${durationTag}</span>
                        <div class="tool-detail-body">
                            ${t.args ? `<div class="tool-args">${JSON.stringify(t.args, null, 2)}</div>` : ''}
                            ${t.result ? `<div class="tool-result">${t.result}</div>` : ''}
                        </div>
                    </div>`;
                }).join('');
                // Show "thinking between rounds" when all tools done but no new tokens yet
                const allDone = toolDetails.length > 0 && toolDetails.every(t => t.result);
                if (thinking && allDone) {
                    agentBubble.querySelector('.tools').innerHTML +=
                        `<div class="agent-thinking"><div class="think-dot"></div><div class="think-dot"></div><div class="think-dot"></div><span>thinking…</span></div>`;
                }
                el('msgs').scrollTop = el('msgs').scrollHeight;
            }
            const _resetSend = () => {
                sending = false;
                el('sendBtn').disabled = false;
                el('sendBtn').style.display = '';
                el('stopBtn').style.display = 'none';
                _abortController = null;
            };
            try {
                const resp = await fetch(`${API}/api/chat/stream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, session_id: sessionId }),
                    signal: _abortController.signal,
                });
                if (!resp.ok) {
                    hideTyping();
                    const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
                    addMsg(`Error: ${err.detail || 'Request failed'}`, 'agent');
                } else {
                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    let buf = '';
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += decoder.decode(value, { stream: true });
                        const lines = buf.split('\n');
                        buf = lines.pop();
                        for (const line of lines) {
                            if (!line.startsWith('data: ')) continue;
                            try {
                                const ev = JSON.parse(line.slice(6));
                                if (ev.type === 'token') { ensureBubble(); agentContent += ev.content; updateBubble(false); }
                                else if (ev.type === 'tool_start') { ensureBubble(); toolDetails.push({ name: ev.name, args: ev.args || null, result: null }); updateBubble(false); }
                                else if (ev.type === 'tool_result') {
                                    const last = toolDetails.findLast ? toolDetails.findLast(t => t.name === ev.name && !t.result) : [...toolDetails].reverse().find(t => t.name === ev.name && !t.result);
                                    if (last) { last.result = ev.result; last.duration_ms = ev.duration_ms; }
                                    // Clear any status line when tool finishes
                                    if (agentBubble) { const sl = agentBubble.querySelector('.agent-status'); if (sl) sl.remove(); }
                                    updateBubble(true);
                                }
                                else if (ev.type === 'skills') {
                                    ensureBubble();
                                    const existing = agentBubble.querySelector('.skill-pills');
                                    if (!existing && ev.skills && ev.skills.length) {
                                        const pills = document.createElement('div');
                                        pills.className = 'skill-pills';
                                        pills.innerHTML = ev.skills.map(s => `<span class="skill-pill">${s}</span>`).join('');
                                        agentBubble.querySelector('.body').prepend(pills);
                                    }
                                }
                                else if (ev.type === 'status') {
                                    ensureBubble();
                                    let sl = agentBubble.querySelector('.agent-status');
                                    if (!sl) { sl = document.createElement('div'); sl.className = 'agent-status'; agentBubble.querySelector('.tools').appendChild(sl); }
                                    sl.textContent = ev.message || '';
                                }
                                else if (ev.type === 'task_plan') {
                                    try { renderTaskPlan(ev.plan); } catch {}
                                }
                                else if (ev.type === 'code_exec') {
                                    ensureBubble();
                                    const desc = ev.description ? `<div class="code-exec-desc">${ev.description}</div>` : '';
                                    const block = document.createElement('div');
                                    block.className = 'code-exec-wrap';
                                    block.innerHTML = `${desc}<pre class="code-exec-block" onclick="this.classList.toggle('expanded')">${ev.code.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre>`;
                                    agentBubble.querySelector('.tools').appendChild(block);
                                }
                                else if (ev.type === 'start') { if (ev.session_id) sessionId = ev.session_id; }
                                else if (ev.type === 'done') {
                                    if (ev.session_id) { sessionId = ev.session_id; localStorage.setItem('pathclaw_active_session', sessionId); }
                                    if (!agentBubble && ev.tool_calls_made && ev.tool_calls_made.length === 0) {
                                        if (!typingRemoved) { hideTyping(); typingRemoved = true; }
                                        addMsg('(no response)', 'agent');
                                    }
                                    renderFileTree();
                                    loadSessionList();
                                    try { refreshTaskPlan(); } catch {}
                                } else if (ev.type === 'error') {
                                    if (!typingRemoved) { hideTyping(); typingRemoved = true; }
                                    addMsg(`Error: ${ev.message}`, 'agent');
                                }
                            } catch (_) { }
                        }
                    }
                }
            } catch (e) {
                if (!typingRemoved) { hideTyping(); typingRemoved = true; }
                if (e.name === 'AbortError') {
                    if (agentContent) { /* keep partial response */ }
                    else addMsg('Stopped.', 'agent');
                } else {
                    addMsg(`Cannot reach backend: ${e.message}`, 'agent');
                }
            }
            _resetSend();
        }

        function autoResize(ta) { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'; }
        el('chatIn').addEventListener('input', () => autoResize(el('chatIn')));
        el('sendBtn').onclick = () => { const v = el('chatIn').value.trim(); if (!v) return; el('chatIn').value = ''; autoResize(el('chatIn')); sendMessage(v); };
        el('chatIn').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el('sendBtn').click(); } };

        // === Workspace Tab System ===
        const wsEl = el('wsContent');
        const TAB_FNS = { overview: renderOverview, config: renderConfig, plots: renderPlots, slides: renderSlides, logs: renderLogs, viewer: renderViewer, notebook: () => renderNotebook(), folders: () => renderFolders(), manuscript: () => renderManuscript(), csv: () => renderCsv(), editor: () => renderEditor() };

        // === WSI Viewer (OpenSeadragon + DZI tileserver) ===
        let osdViewer = null;
        let _viewerDatasetId = null, _viewerSlideStem = null;
        let _heatmapLayer = null, _heatmapExpId = null;
        let _geoCanvas = null, _geoData = null, _geoVisible = true;

        // GeoJSON class → colour mapping (QuPath convention)
        const _GEO_PALETTE = {
            tumor: '#ef4444', stroma: '#3b82f6', lymphocyte: '#22c55e',
            immune: '#22c55e', necrosis: '#f97316', epithelial: '#a855f7',
            gland: '#a855f7', annotation: '#facc15', region: '#facc15',
        };
        function _geoColor(cls = '') {
            const k = cls.toLowerCase();
            for (const [key, val] of Object.entries(_GEO_PALETTE)) if (k.includes(key)) return val;
            return '#71717a';
        }

        function _initGeoOverlay(datasetId, slideStem) {
            const container = el('osd-container');
            if (_geoCanvas) _geoCanvas.remove();
            _geoCanvas = document.createElement('canvas');
            _geoCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2';
            container.style.position = 'relative';
            container.appendChild(_geoCanvas);
            const redraw = () => _redrawGeo();
            osdViewer.addHandler('animation', redraw);
            osdViewer.addHandler('open', redraw);
            osdViewer.addHandler('resize', redraw);
            // Try loading existing GeoJSON from server
            fetch(`${API}/api/tiles/${datasetId}/${slideStem}/geojson`)
                .then(r => r.ok ? r.json() : null)
                .then(d => { if (d) { _geoData = d; _redrawGeo(); const t = el('geoTogBtn'); if (t) t.style.display = ''; } })
                .catch(() => {});
        }

        function _redrawGeo() {
            if (!_geoCanvas || !osdViewer || !_geoData || !_geoVisible) {
                if (_geoCanvas) { const ctx = _geoCanvas.getContext('2d'); ctx.clearRect(0, 0, _geoCanvas.width, _geoCanvas.height); }
                return;
            }
            const container = el('osd-container');
            _geoCanvas.width = container.offsetWidth;
            _geoCanvas.height = container.offsetHeight;
            const ctx = _geoCanvas.getContext('2d');
            ctx.clearRect(0, 0, _geoCanvas.width, _geoCanvas.height);
            const tiledImage = osdViewer.world.getItemAt(0);
            if (!tiledImage) return;

            const toScreen = (x, y) => {
                const vp = tiledImage.imageToViewportCoordinates(new OpenSeadragon.Point(x, y));
                return osdViewer.viewport.viewportToViewerElementCoordinates(vp);
            };

            const features = _geoData.features || (_geoData.type === 'Feature' ? [_geoData] : []);
            for (const feat of features) {
                const cls = feat.properties?.classification?.name || feat.properties?.class || feat.properties?.label || '';
                const col = _geoColor(cls);
                const geom = feat.geometry;
                if (!geom) continue;
                const rings = geom.type === 'Polygon' ? geom.coordinates
                    : geom.type === 'MultiPolygon' ? geom.coordinates.flat(1) : [];
                for (const ring of rings) {
                    ctx.beginPath();
                    ring.forEach(([x, y], i) => {
                        const sp = toScreen(x, y);
                        i === 0 ? ctx.moveTo(sp.x, sp.y) : ctx.lineTo(sp.x, sp.y);
                    });
                    ctx.closePath();
                    ctx.fillStyle = col + '26';   // ~15% opacity fill
                    ctx.strokeStyle = col;
                    ctx.lineWidth = 1.5;
                    ctx.fill();
                    ctx.stroke();
                }
            }
        }

        async function _loadGeoJSONFile(event, datasetId, slideStem) {
            const file = event.target.files[0];
            if (!file) return;
            const text = await file.text();
            try {
                _geoData = JSON.parse(text);
            } catch { showError('Invalid GeoJSON file'); return; }
            _geoVisible = true;
            const tog = el('geoTogBtn');
            if (tog) { tog.textContent = 'Hide GeoJSON'; tog.style.display = ''; }

            // Upload to server for persistence
            const fd = new FormData();
            fd.append('file', file);
            await fetch(`${API}/api/tiles/${datasetId}/${slideStem}/geojson`, { method: 'POST', body: fd }).catch(() => {});

            _redrawGeo();
        }

        function _toggleGeo() {
            _geoVisible = !_geoVisible;
            const tog = el('geoTogBtn');
            if (tog) tog.textContent = _geoVisible ? 'Hide GeoJSON' : 'Show GeoJSON';
            _redrawGeo();
        }

        async function renderViewer() {
            wsEl.className = 'ws-content fill';
            if (_viewerDatasetId && _viewerSlideStem) {
                _mountOSD(_viewerDatasetId, _viewerSlideStem);
                return;
            }
            // Show slide picker (session-scoped)
            const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            const dsRes = await fetch(`${API}/api/datasets${sidQ}`).then(r => r.json()).catch(() => ({ datasets: [] }));
            const ds = dsRes.datasets || [];
            if (!ds.length) {
                wsEl.innerHTML = `<div class="ws-empty"><h3>No datasets</h3><p>Register a dataset first, then click a slide in the Explorer to open it here.</p></div>`;
                wsEl.className = 'ws-content'; return;
            }
            // Build slide options from all datasets
            let opts = '';
            for (const d of ds) {
                const sr = await fetch(`${API}/api/datasets/${d.id}/slides`).then(r => r.json()).catch(() => ({ slides: [] }));
                for (const s of (sr.slides || [])) {
                    const stem = s.filename.replace(/\.[^.]+$/, '');
                    opts += `<option value="${d.id}|${stem}">${d.name} / ${s.filename}</option>`;
                }
            }
            if (!opts) {
                wsEl.innerHTML = `<div class="ws-empty"><h3>No slides found</h3><p>Register a dataset with WSI files (.svs, .tiff, etc.).</p></div>`;
                wsEl.className = 'ws-content'; return;
            }
            wsEl.innerHTML = `<div style="padding:24px;width:100%">
                <p class="sec-title">Open Slide</p>
                <div style="display:flex;gap:10px;align-items:center;margin-top:8px">
                    <select class="cfg-sel" id="slidePickSel" style="flex:1">${opts}</select>
                    <button class="qa-btn" onclick="_launchPickedSlide()">Open Viewer</button>
                </div>
            </div>`;
        }

        function _launchPickedSlide() {
            const v = el('slidePickSel')?.value;
            if (!v) return;
            const [dsId, stem] = v.split('|');
            openSlideInViewer(dsId, stem);
        }

        let _csvActivePath = '';

        function openCsvViewer(path) {
            _csvActivePath = path;
            const tab = document.querySelector('.ws-tab[data-t="csv"]');
            if (tab) tab.click();
            else renderCsv();
        }

        async function renderCsv() {
            const path = _csvActivePath;
            if (!path) {
                wsEl.innerHTML = `<div class="ws-empty"><h3>No CSV selected</h3><p>Click a label CSV in the left sidebar (under a dataset) to open it here.</p></div>`;
                return;
            }
            wsEl.innerHTML = `
                <div class="ws-content fill" style="align-items:stretch;display:flex;flex-direction:column;height:100%;padding:0">
                    <div style="display:flex;align-items:center;padding:10px 14px;border-bottom:1px solid var(--border);gap:10px">
                        <strong style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${path}">${path.split('/').pop()}</strong>
                        <span id="csvMeta" class="tree-meta"></span>
                        <button class="btn-ghost" style="font-size:12px" onclick="csvReload()">Reload</button>
                        <button class="btn" style="font-size:12px;padding:4px 12px" onclick="csvSave()">Save</button>
                    </div>
                    <textarea id="csvText" spellcheck="false" style="flex:1;border:0;padding:10px 14px;background:var(--bg);color:var(--fg);font-family:monospace;font-size:12px;resize:none;outline:none;min-height:400px">Loading…</textarea>
                    <div id="csvStatus" style="padding:6px 14px;border-top:1px solid var(--border);font-size:12px;opacity:.7">${path}</div>
                </div>`;
            await csvReload();
        }

        async function csvReload() {
            const path = _csvActivePath;
            const ta = document.getElementById('csvText');
            if (!ta) return;
            ta.value = 'Loading…';
            try {
                const r = await fetch(`${API}/api/datasets/csv?path=${encodeURIComponent(path)}`);
                const j = await r.json();
                if (r.ok) {
                    ta.value = j.content;
                    const meta = document.getElementById('csvMeta'); if (meta) meta.textContent = `${j.size_kb} KB`;
                } else {
                    ta.value = `Error: ${j.detail || r.status}`;
                }
            } catch (e) { ta.value = 'Error: ' + e.message; }
        }

        async function csvSave() {
            const path = _csvActivePath;
            const content = document.getElementById('csvText').value;
            const status = document.getElementById('csvStatus');
            status.textContent = 'Saving…';
            try {
                const r = await fetch(`${API}/api/datasets/csv`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path, content })
                });
                const j = await r.json();
                status.textContent = r.ok ? `Saved ${j.bytes} bytes · ${path}` : `Error: ${j.detail || r.status}`;
            } catch (e) { status.textContent = 'Save failed: ' + e.message; }
        }

        // === Monaco code editor ===
        let _monacoEditor = null;
        let _monacoPath = null;      // currently-open workspace-relative path
        let _monacoDirty = false;

        function _monacoLang(path) {
            const ext = (path.split('.').pop() || '').toLowerCase();
            return ({ py: 'python', js: 'javascript', ts: 'typescript', json: 'json',
                     md: 'markdown', yml: 'yaml', yaml: 'yaml', toml: 'ini', tex: 'latex',
                     sh: 'shell', css: 'css', html: 'html', xml: 'xml' }[ext]) || 'plaintext';
        }

        async function renderEditor() {
            wsEl.className = 'ws-content editor-mode';
            wsEl.style.padding = '0';
            wsEl.innerHTML = `
                <div style="display:flex;flex:1;min-height:0;height:100%;width:100%">
                    <div style="width:260px;flex-shrink:0;border-right:1px solid var(--border);display:flex;flex-direction:column;background:var(--bg)">
                        <div style="padding:8px 10px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:6px">
                            <strong style="font-size:12px;flex:1">Files</strong>
                            <button class="btn-ghost" title="New file" style="font-size:12px;padding:2px 8px" onclick="editorNewFile()">+</button>
                            <button class="btn-ghost" title="Refresh" style="font-size:12px;padding:2px 8px" onclick="editorRefreshTree()">↻</button>
                        </div>
                        <div id="editorTree" style="flex:1;min-height:0;overflow:auto;padding:4px;font-family:var(--mono);font-size:12px">Loading…</div>
                        <div style="padding:6px 8px;border-top:1px solid var(--border);font-size:11px;opacity:.7">Ctrl+S to save</div>
                    </div>
                    <div style="flex:1;min-width:0;display:flex;flex-direction:column">
                        <div style="padding:6px 12px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;font-size:12px">
                            <span id="editorPath" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono)">(no file open)</span>
                            <span id="editorDirty" style="color:#f59e0b;display:none">●</span>
                            <button class="btn" style="font-size:12px;padding:4px 10px" onclick="editorSave()">Save</button>
                        </div>
                        <div id="monacoHost" style="flex:1;min-height:0;min-width:0;position:relative"></div>
                    </div>
                </div>`;
            await editorRefreshTree();
            _mountMonaco();
        }

        // Persisted expand state for editor tree directories
        const _editorDirOpen = _editorDirOpen_init();
        function _editorDirOpen_init() {
            try {
                const saved = JSON.parse(sessionStorage.getItem('editorDirOpen') || '{}');
                return saved;
            } catch { return {}; }
        }
        function _editorToggleDir(path) {
            _editorDirOpen[path] = !_editorDirOpen[path];
            sessionStorage.setItem('editorDirOpen', JSON.stringify(_editorDirOpen));
            editorRefreshTree();
        }
        window._editorToggleDir = _editorToggleDir;

        async function editorRefreshTree() {
            const host = el('editorTree');
            if (!host) return;
            try {
                const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
                const r = await fetch(`${API}/api/workspace/tree${sidQ}`);
                const d = await r.json();
                const render = (items, depth = 0) => items.map(item => {
                    const pad = 4 + depth * 12;
                    if (item.type === 'dir') {
                        const open = _editorDirOpen[item.path] !== false; // default open
                        const chev = open ? '▼' : '▶';
                        const sub = (open && item.children && item.children.length) ? render(item.children, depth + 1) : '';
                        const escDir = item.path.replace(/'/g, "\\'");
                        return `<div style="padding:3px 0 3px ${pad}px;cursor:pointer;opacity:.9;user-select:none" onclick="_editorToggleDir('${escDir}')" onmouseover="this.style.background='rgba(255,255,255,.04)'" onmouseout="this.style.background=''"><span style="display:inline-block;width:10px">${chev}</span> 📁 ${item.name}</div>${sub}`;
                    }
                    const escPath = item.path.replace(/'/g, "\\'");
                    const kb = (item.size / 1024).toFixed(1);
                    const lower = item.name.toLowerCase();
                    const icon = /\.(png|jpe?g|gif|bmp|webp)$/.test(lower) ? '🖼️'
                               : /\.(py|ipynb)$/.test(lower) ? '🐍'
                               : /\.(js|ts|jsx|tsx)$/.test(lower) ? '📜'
                               : /\.(md|txt|rst)$/.test(lower) ? '📝'
                               : /\.(json|ya?ml|toml)$/.test(lower) ? '⚙️'
                               : '📄';
                    return `<div style="padding:3px 0 3px ${pad + 10}px;cursor:pointer;display:flex;align-items:center;gap:6px;user-select:none" onclick="editorOpenFile('${escPath}')" onmouseover="this.style.background='rgba(255,255,255,.04)'" onmouseout="this.style.background=''"><span style="font-size:12px">${icon}</span><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${item.name}</span><span style="opacity:.5;font-size:10px">${kb}k</span></div>`;
                }).join('');
                host.innerHTML = d.children.length ? render(d.children) : '<div style="padding:10px;opacity:.6">(empty — create a file with +)</div>';
            } catch (e) {
                host.innerHTML = `<div style="padding:10px;color:#ef4444">Error: ${e.message}</div>`;
            }
        }

        function _mountMonaco() {
            const host = el('monacoHost');
            if (!host || _monacoEditor) return;
            if (typeof require === 'undefined') {
                host.innerHTML = '<div style="padding:20px;opacity:.6">Monaco loader not available.</div>';
                return;
            }
            require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs' }});
            require(['vs/editor/editor.main'], function () {
                _monacoEditor = monaco.editor.create(host, {
                    value: '// Select a file from the workspace tree, or click + to create one.\n',
                    language: 'javascript',
                    theme: 'vs-dark',
                    automaticLayout: true,
                    minimap: { enabled: false },
                    fontSize: 15,
                    lineHeight: 22,
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
                    renderLineHighlight: 'all',
                    smoothScrolling: true,
                    padding: { top: 12, bottom: 12 },
                    scrollBeyondLastLine: false,
                    wordWrap: 'on',
                });
                _monacoEditor.onDidChangeModelContent(() => {
                    if (!_monacoDirty) { _monacoDirty = true; const d = el('editorDirty'); if (d) d.style.display = ''; }
                });
                _monacoEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => editorSave());
            });
        }

        const _IMG_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'];

        function _showImagePreview(path) {
            _monacoPath = null;
            _monacoDirty = false;
            const host = el('monacoHost');
            if (!host) return;
            // Hide Monaco (keep the instance, just cover it)
            let overlay = el('imgPreviewOverlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'imgPreviewOverlay';
                overlay.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#1e1e1e;overflow:auto;padding:20px';
                host.style.position = 'relative';
                host.appendChild(overlay);
            }
            const sidQ = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : '';
            const url = `${API}/api/workspace/raw?path=${encodeURIComponent(path)}&t=${Date.now()}${sidQ}`;
            overlay.innerHTML = `<img src="${url}" alt="${path}" style="max-width:100%;max-height:100%;object-fit:contain;background:#111;border:1px solid var(--border);box-shadow:0 4px 20px rgba(0,0,0,.4)">`;
            overlay.style.display = 'flex';
            el('editorPath').textContent = path + '  (image preview)';
            el('editorDirty').style.display = 'none';
        }

        function _hideImagePreview() {
            const o = el('imgPreviewOverlay');
            if (o) o.style.display = 'none';
        }

        async function editorOpenFile(path) {
            const lower = path.toLowerCase();
            const isImage = _IMG_EXTS.some(ext => lower.endsWith(ext));
            if (isImage) {
                if (!_monacoEditor) _mountMonaco();
                _showImagePreview(path);
                return;
            }
            if (!_monacoEditor) _mountMonaco();
            if (_monacoDirty && !confirm('Unsaved changes. Discard?')) return;
            try {
                const sidQ = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : '';
                const r = await fetch(`${API}/api/workspace/file?path=${encodeURIComponent(path)}${sidQ}`);
                if (!r.ok) {
                    const txt = await r.text();
                    showError(`Failed to open: ${txt}`);
                    return;
                }
                const d = await r.json();
                _hideImagePreview();
                _monacoPath = d.path;
                const lang = _monacoLang(d.path);
                const wait = () => {
                    if (!_monacoEditor || !window.monaco) { setTimeout(wait, 100); return; }
                    const model = monaco.editor.createModel(d.content, lang);
                    _monacoEditor.setModel(model);
                    _monacoDirty = false;
                    el('editorDirty').style.display = 'none';
                    el('editorPath').textContent = d.path;
                };
                wait();
            } catch (e) {
                showError('Open error: ' + e.message);
            }
        }

        async function editorSave() {
            if (!_monacoEditor || !_monacoPath) { showError('No file open.'); return; }
            const content = _monacoEditor.getValue();
            try {
                const r = await fetch(`${API}/api/workspace/file`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: _monacoPath, content, session_id: sessionId || '' })
                });
                const raw = await r.text();
                let d = null; try { d = JSON.parse(raw); } catch {}
                if (r.ok && d && d.status === 'ok') {
                    _monacoDirty = false;
                    el('editorDirty').style.display = 'none';
                    editorRefreshTree();
                } else {
                    showError('Save failed: ' + ((d && d.detail) || raw));
                }
            } catch (e) {
                showError('Save error: ' + e.message);
            }
        }

        async function editorNewFile() {
            const path = prompt('New file path (e.g. user_code/my_script.py):', 'user_code/untitled.py');
            if (!path) return;
            try {
                const r = await fetch(`${API}/api/workspace/file`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path, content: '', session_id: sessionId || '' })
                });
                if (!r.ok) {
                    const txt = await r.text();
                    showError('Create failed: ' + txt);
                    return;
                }
                await editorRefreshTree();
                await editorOpenFile(path);
            } catch (e) { showError('Create error: ' + e.message); }
        }

        function openSlideInViewer(datasetId, slideStem) {
            _viewerDatasetId = datasetId;
            _viewerSlideStem = slideStem;
            const tab = document.querySelector('.ws-tab[data-t="viewer"]');
            if (tab) { tab.click(); return; }
            _mountOSD(datasetId, slideStem);
        }

        async function _mountOSD(datasetId, slideStem) {
            if (!window.OpenSeadragon) {
                wsEl.innerHTML = `<div class="ws-empty"><h3>Loading viewer...</h3><p>OpenSeadragon is loading. Please try again in a moment.</p></div>`;
                wsEl.className = 'ws-content'; return;
            }
            wsEl.className = 'ws-content fill';
            wsEl.style.padding = '0';

            // Fetch completed experiments for heatmap selector
            const artRes = await fetch(`${API}/api/artifacts`).then(r => r.json()).catch(() => ({ artifacts: [] }));
            const expOpts = (artRes.artifacts || [])
                .filter(a => a.has_model)
                .map(a => `<option value="${a.experiment_id}">${a.experiment_id}</option>`)
                .join('');

            wsEl.innerHTML = `<div class="viewer-wrap">
                <div class="viewer-toolbar">
                    <span class="vtlbl" style="font-family:var(--mono);color:var(--blue);font-weight:600">${slideStem}</span>
                    <span class="vtlbl" id="vZoom">—</span>
                    <span style="flex:1"></span>
                    ${expOpts ? `<select class="cfg-sel" id="hmExpSel" style="max-width:160px;font-size:11px">${expOpts}</select>
                    <button class="qa-btn" id="hmLoadBtn" onclick="_loadHeatmap('${datasetId}','${slideStem}')">Heatmap</button>
                    <button class="qa-btn" id="hmTogBtn" onclick="_toggleHeatmap()" style="display:none">Hide</button>
                    <input type="range" id="hmOpacity" min="0" max="100" value="60" style="width:72px;display:none" oninput="_setHeatmapOpacity(this.value)">` : ''}
                    <label class="qa-btn" style="cursor:pointer;position:relative">
                        GeoJSON
                        <input type="file" accept=".geojson,.json" style="position:absolute;inset:0;opacity:0;cursor:pointer" onchange="_loadGeoJSONFile(event,'${datasetId}','${slideStem}')">
                    </label>
                    <button class="qa-btn" id="geoTogBtn" onclick="_toggleGeo()" style="display:none">Hide GeoJSON</button>
                    <button class="qa-btn" onclick="_resetViewerChoice()">Close</button>
                </div>
                <div id="osd-container"></div>
            </div>`;

            if (osdViewer) { osdViewer.destroy(); osdViewer = null; }
            _heatmapLayer = null; _heatmapExpId = null;

            osdViewer = OpenSeadragon({
                id: 'osd-container',
                prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/',
                tileSources: `${API}/api/tiles/${datasetId}/${slideStem}/dzi.dzi`,
                showNavigator: true,
                navigatorPosition: 'BOTTOM_RIGHT',
                navigatorSizeRatio: 0.15,
                maxZoomPixelRatio: 4,
                minZoomLevel: 0.1,
                timeout: 90000,
                imageLoaderLimit: 8,
                crossOriginPolicy: 'Anonymous',
                showFullPageControl: false,
                gestureSettingsMouse: { clickToZoom: false },
            });

            osdViewer.addHandler('zoom', e => {
                const pct = Math.round(e.zoom * 100);
                const z = el('vZoom'); if (z) z.textContent = `${pct}%`;
            });

            // Init GeoJSON canvas overlay (auto-loads from server if file exists)
            _geoData = null; _geoVisible = true;
            _initGeoOverlay(datasetId, slideStem);
        }

        async function _loadHeatmap(datasetId, slideStem) {
            const expId = el('hmExpSel')?.value;
            if (!expId) return;
            const btn = el('hmLoadBtn');
            if (btn) { btn.textContent = 'Generating…'; btn.disabled = true; }

            // Check if heatmap exists, generate if not
            const statusRes = await fetch(`${API}/api/eval/${expId}/heatmap/status/${slideStem}`)
                .then(r => r.json()).catch(() => ({ status: 'error' }));

            if (statusRes.status !== 'ready') {
                const genRes = await fetch(`${API}/api/eval/${expId}/heatmap`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_id: datasetId, slide_stem: slideStem })
                }).then(r => r.json()).catch(() => ({}));

                // Poll until ready
                let tries = 0;
                while (tries < 120) {
                    await new Promise(r => setTimeout(r, 2000));
                    const s = await fetch(`${API}/api/eval/${expId}/heatmap/status/${slideStem}`)
                        .then(r => r.json()).catch(() => ({ status: 'error' }));
                    if (s.status === 'ready') break;
                    if (s.status === 'failed') {
                        showError('Heatmap generation failed');
                        if (btn) { btn.textContent = 'Heatmap'; btn.disabled = false; }
                        return;
                    }
                    tries++;
                }
            }

            if (btn) { btn.textContent = 'Heatmap'; btn.disabled = false; }

            // Remove old heatmap layer if any
            if (_heatmapLayer) { osdViewer.world.removeItem(_heatmapLayer); _heatmapLayer = null; }
            _heatmapExpId = expId;

            const opacity = (parseInt(el('hmOpacity')?.value || '60')) / 100;
            osdViewer.addTiledImage({
                tileSource: `${API}/api/tiles/${datasetId}/${slideStem}/heatmap/${expId}/dzi.dzi`,
                opacity: opacity,
                success: e => {
                    _heatmapLayer = e.item;
                    const tog = el('hmTogBtn'); if (tog) tog.style.display = '';
                    const opc = el('hmOpacity'); if (opc) opc.style.display = '';
                },
            });
        }

        function _toggleHeatmap() {
            if (!_heatmapLayer) return;
            const visible = _heatmapLayer.getOpacity() > 0;
            _heatmapLayer.setOpacity(visible ? 0 : (parseInt(el('hmOpacity')?.value || '60') / 100));
            const btn = el('hmTogBtn');
            if (btn) btn.textContent = visible ? 'Show' : 'Hide';
        }

        function _setHeatmapOpacity(val) {
            if (_heatmapLayer) _heatmapLayer.setOpacity(parseInt(val) / 100);
        }

        function _resetViewerChoice() {
            if (osdViewer) { osdViewer.destroy(); osdViewer = null; }
            _viewerDatasetId = null; _viewerSlideStem = null;
            wsEl.style.padding = '';
            renderViewer();
        }

        // Tabs are split into two independent groups (data-g="browser" | "workspace").
        // Each group tracks its own active tab in sessionStorage so switching in one
        // group doesn't reset the other. The LAST clicked tab drives the center panel.
        document.querySelectorAll('.ws-tab').forEach(t => t.onclick = () => {
            const g = t.dataset.g || '';
            // Only clear active within the same group
            const selector = g ? `.ws-tab[data-g="${g}"]` : '.ws-tab';
            document.querySelectorAll(selector).forEach(x => x.classList.remove('active'));
            t.classList.add('active');
            if (g) {
                try { sessionStorage.setItem(`ws_active_${g}`, t.dataset.t); } catch {}
            }
            const fn = TAB_FNS[t.dataset.t];
            if (fn) fn();
        });
        // Restore per-group active state on load
        (function _restoreTabGroups() {
            for (const g of ['browser', 'workspace']) {
                let stored = '';
                try { stored = sessionStorage.getItem(`ws_active_${g}`) || ''; } catch {}
                if (!stored) continue;
                const t = document.querySelector(`.ws-tab[data-g="${g}"][data-t="${stored}"]`);
                if (t) {
                    document.querySelectorAll(`.ws-tab[data-g="${g}"]`).forEach(x => x.classList.remove('active'));
                    t.classList.add('active');
                }
            }
        })();

        function sendFromUI(text) { el('chatIn').value = text; el('sendBtn').click(); }

        // === Session management ===
        function _relativeTime(ts) {
            const diff = Date.now() / 1000 - ts;
            if (diff < 60) return 'just now';
            if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
            if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
            if (diff < 86400 * 7) return Math.floor(diff / 86400) + 'd ago';
            return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        }

        function _setSessionTitle(title) {
            sessionTitle = title;
            el('sessionTitle').textContent = title || 'New Session';
            el('sessionTitle').title = title || 'New Session';
        }

        async function initSession() {
            const stored = localStorage.getItem('pathclaw_active_session');
            if (stored) {
                try {
                    const d = await fetch(`${API}/api/chat/sessions/${stored}/resume`, { method: 'POST' }).then(r => {
                        if (!r.ok) throw new Error('not found');
                        return r.json();
                    });
                    sessionId = stored;
                    _setSessionTitle(d.title || 'New Session');
                    el('msgs').innerHTML = '';
                    for (const m of (d.messages || [])) {
                        if (m.role === 'user') addMsg(m.content, 'user');
                        else if (m.role === 'assistant' && m.content) addMsg(m.content, 'agent');
                    }
                    if (!d.messages || !d.messages.length) _showWelcome();
                    await loadSessionList();
                    return;
                } catch(e) { /* fall through to create new */ }
            }
            await createNewSession();
        }

        function _resetWorkspacePanel() {
            // Stop any active polling so the previous session's job doesn't bleed into this one
            try { if (typeof jobPollInterval !== 'undefined' && jobPollInterval) { clearInterval(jobPollInterval); jobPollInterval = null; } } catch {}
            try { activeJobId = null; sessionStorage.removeItem('activeJobId'); } catch {}
            try { activeExperimentId = null; } catch {}
            // Reset Data-group back to Overview; clear Workspace-group active state.
            document.querySelectorAll('.ws-tab[data-g="browser"]').forEach(t => t.classList.toggle('active', t.dataset.t === 'overview'));
            document.querySelectorAll('.ws-tab[data-g="workspace"]').forEach(t => t.classList.remove('active'));
            try { sessionStorage.setItem('ws_active_browser', 'overview'); sessionStorage.removeItem('ws_active_workspace'); } catch {}
            // Blank out the center panel
            if (wsEl) {
                wsEl.className = 'ws-content';
                wsEl.style.padding = '';
                wsEl.innerHTML = `<div class="ws-empty"><h3>New session</h3>
                    <p>Ask something in chat, open a slide in the Explorer, or click Editor to write code.</p></div>`;
            }
        }

        async function createNewSession() {
            // Common reset — runs on both success and failure paths so the prior session's
            // chat, tool outputs, and workspace state never bleed into the new one.
            const _resetSessionUI = () => {
                el('msgs').innerHTML = '';
                _showWelcome();
                _resetWorkspacePanel();
                // Clear any stray per-session UI: composer input, attachment strip, active-job card
                try { const ci = el('chatIn'); if (ci) { ci.value = ''; autoResize(ci); } } catch {}
                try { const a = el('attachList'); if (a) a.innerHTML = ''; } catch {}
                try { const j = el('liveJobCard'); if (j) j.remove(); } catch {}
                // Force sidebar re-render (workspace tree + experiments) so its state reflects no-active-session
                try { renderFileTree(true); } catch {}
            };
            try {
                const d = await fetch(`${API}/api/chat/sessions`, { method: 'POST' }).then(r => r.json());
                sessionId = d.session_id;
                _setSessionTitle('New Session');
                localStorage.setItem('pathclaw_active_session', sessionId);
                _resetSessionUI();
            } catch(e) {
                sessionId = '';
                _setSessionTitle('New Session');
                _resetSessionUI();
            }
            await loadSessionList();
        }

        function _showWelcome() {
            if (el('msgs').children.length === 0) {
                el('msgs').innerHTML = `<div class="msg agent"><div class="who">PathClaw</div>
                    <strong>What would you like to do?</strong><br><br>
                    Try: <em>"Review the SOTA on MSI prediction from H&amp;E, last 3 years, with PubMed links"</em><br>
                    Or: <em>"Train TransMIL on tcga-ucec UNI features for MSI, then write up the results section"</em>
                </div>`;
            }
        }

        async function switchSession(sid) {
            if (sid === sessionId) { toggleSessionDrawer(); return; }
            try {
                const d = await fetch(`${API}/api/chat/sessions/${sid}/resume`, { method: 'POST' }).then(r => {
                    if (!r.ok) throw new Error('not found');
                    return r.json();
                });
                sessionId = sid;
                _setSessionTitle(d.title || 'New Session');
                localStorage.setItem('pathclaw_active_session', sessionId);
                el('msgs').innerHTML = '';
                for (const m of (d.messages || [])) {
                    if (m.role === 'user') addMsg(m.content, 'user');
                    else if (m.role === 'assistant' && m.content) addMsg(m.content, 'agent');
                }
                if (!d.messages || !d.messages.length) _showWelcome();
                _resetWorkspacePanel();
                renderSessionList();
                renderFileTree();
                refreshTaskPlan();
                refreshJobsPanel();
                toggleSessionDrawer();
            } catch(e) {
                showError('Failed to load session.');
            }
        }

        async function deleteSession(sid, e) {
            e.stopPropagation();
            // Find session title for confirmation dialog
            const titleEl = document.querySelector(`.session-item[data-sid="${sid}"] .session-item-title`);
            const label = titleEl ? titleEl.textContent.trim() : 'this session';
            if (!confirm(`Delete "${label}"?\n\nThis will permanently remove the chat history from the server.`)) return;
            await fetch(`${API}/api/chat/history/${sid}`, { method: 'DELETE' });
            if (sid === sessionId) {
                const remaining = await fetch(`${API}/api/chat/history`).then(r => r.json()).then(d => d.chats || []);
                if (remaining.length) await switchSession(remaining[0].session_id);
                else await createNewSession();
            } else {
                await loadSessionList();
            }
        }

        async function loadSessionList() {
            try {
                const d = await fetch(`${API}/api/chat/history`).then(r => r.json());
                window._sessionListCache = d.chats || [];
                renderSessionList();
            } catch(e) {}
        }

        function renderSessionList() {
            const chats = window._sessionListCache || [];
            const listEl = el('sessionList');
            if (!listEl) return;
            if (!chats.length) {
                listEl.innerHTML = '<p style="padding:8px 10px;font-size:11px;color:var(--text-3)">No sessions yet.</p>';
                return;
            }
            listEl.innerHTML = chats.map(c => {
                const active = c.session_id === sessionId ? ' active' : '';
                const t = c.title.length > 45 ? c.title.slice(0, 45) + '…' : c.title;
                const meta = _relativeTime(c.updated_at || c.created_at) + (c.message_count ? ` · ${c.message_count} msgs` : '');
                return `<div class="session-item${active}" data-sid="${c.session_id}" onclick="switchSession('${c.session_id}')">
                    <div class="session-item-body">
                        <div class="session-item-title">${t}</div>
                        <div class="session-item-meta">${meta}</div>
                    </div>
                    <button class="session-del" onclick="deleteSession('${c.session_id}', event)" title="Delete">×</button>
                </div>`;
            }).join('');
        }

        function toggleSessionDrawer() {
            sessionDrawerOpen = !sessionDrawerOpen;
            el('sessionDrawer').style.display = sessionDrawerOpen ? 'block' : 'none';
            if (sessionDrawerOpen) renderSessionList();
        }

        async function startRenameSession() {
            const current = sessionTitle || 'New Session';
            const newTitle = prompt('Rename session:', current);
            if (!newTitle || newTitle === current) return;
            _setSessionTitle(newTitle);
            await fetch(`${API}/api/chat/history/${sessionId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });
            await loadSessionList();
        }

        // === Overview (command centre) ===
        async function renderOverview() {
            wsEl.className = 'ws-content fill';
            wsEl.innerHTML = '<div style="width:100%"><p style="color:var(--text-3);font-size:13px">Loading...</p></div>';

            const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            const [dsRes, expRes] = await Promise.all([
                fetch(`${API}/api/datasets${sidQ}`).then(r => r.json()).catch(() => ({ datasets: [] })),
                fetch(`${API}/api/artifacts${sidQ}`).then(r => r.json()).catch(() => ({ artifacts: [] }))
            ]);
            const ds = dsRes.datasets || [], exps = expRes.artifacts || [];
            const modelCount = exps.filter(a => a.has_model).length;

            // Fetch status for all experiments to find running ones
            const statuses = await Promise.all(exps.map(a =>
                fetch(`${API}/api/training/${a.experiment_id}`).then(r => r.ok ? r.json() : {}).catch(() => ({}))
            ));
            const runningJobs = statuses.filter(s => s.status === 'running' || s.status === 'queued');

            // Build active job section
            let activeJobData = null;
            if (activeJobId) {
                activeJobData = statuses.find(s => s.job_id === activeJobId);
                if (!activeJobData) {
                    try { const r = await fetch(`${API}/api/training/${activeJobId}`); activeJobData = r.ok ? await r.json() : null; } catch { }
                }
            }
            if (!activeJobData && runningJobs.length) activeJobData = runningJobs[0];

            // Start/resume polling if there's a running job
            if (activeJobData && (activeJobData.status === 'running' || activeJobData.status === 'queued') && !jobPollInterval) {
                startJobPolling(activeJobData.job_id);
            }

            let jobHtml = '';
            if (activeJobData) {
                const j = activeJobData, m = j.metrics || {};
                const pct = Math.round((j.progress || 0) * 100);
                const statusColors = { running: 'var(--blue)', completed: 'var(--green)', failed: 'var(--red)', queued: 'var(--amber)' };
                jobHtml = `
                <div id="liveJobCard" class="job-card ${j.status}">
                    <div class="job-card-head">
                        <span class="job-card-id">${j.job_id || activeJobId}</span>
                        <span class="job-badge ${j.status}">${j.status}</span>
                    </div>
                    <div class="job-meta">
                        ${j.config?.mil_method ? `<span class="job-tag">${j.config.mil_method}</span>` : ''}
                        ${j.config?.feature_backbone ? `<span class="job-tag">${j.config.feature_backbone}</span>` : ''}
                        ${j.config?.task ? `<span class="job-tag">${j.config.task}</span>` : ''}
                        ${j.config?.mammoth?.enabled ? `<span class="job-tag" style="color:var(--violet)">MAMMOTH</span>` : ''}
                    </div>
                    <div class="progress-wrap"><div class="progress-fill" style="width:${pct}%"></div></div>
                    <div class="job-metrics">
                        <div class="jm"><div class="jm-lbl">Epoch</div><div class="jm-val jep">${j.epoch || 0} / ${j.total_epochs || '?'}</div></div>
                        <div class="jm"><div class="jm-lbl">Progress</div><div class="jm-val jpct">${pct}%</div></div>
                        <div class="jm"><div class="jm-lbl">Elapsed</div><div class="jm-val">${j.elapsed_human || '—'}</div></div>
                        <div class="jm"><div class="jm-lbl">ETA</div><div class="jm-val">${j.eta_human || (j.status === 'running' ? '…' : '—')}</div></div>
                        <div class="jm"><div class="jm-lbl">Train loss</div><div class="jm-val jtl">${m.train_loss ? m.train_loss.toFixed(4) : '—'}</div></div>
                        <div class="jm"><div class="jm-lbl">Val loss</div><div class="jm-val jvl">${m.val_loss ? m.val_loss.toFixed(4) : '—'}</div></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:10px;align-items:center">
                        <div class="jm"><div class="jm-lbl">Best accuracy</div><div class="jm-val jacc" style="color:${m.best_val_accuracy > 0.8 ? 'var(--green)' : 'var(--text-1)'}">${m.best_val_accuracy ? (m.best_val_accuracy * 100).toFixed(1) + '%' : '—'}</div></div>
                        <button class="qa-btn" style="font-size:11px" onclick="document.querySelector('.ws-tab[data-t=\\"logs\\"]').click()">View logs</button>
                    </div>
                    ${(j.errors || []).length ? `<div class="job-errors" style="margin-top:8px;font-size:11px;color:var(--red);font-family:var(--mono)">${j.errors.slice(-1)[0]}</div>` : '<div class="job-errors"></div>'}
                </div>`;
            }

            // Recent experiments table
            let expTableHtml = '';
            if (statuses.length) {
                const rows = statuses.slice().reverse().slice(0, 8).map(j => {
                    if (!j.job_id) return '';
                    const acc = j.metrics?.best_val_accuracy;
                    const accStr = acc !== undefined ? `${(acc * 100).toFixed(1)}%` : '—';
                    const accClass = acc > 0.8 ? 'acc-good' : acc > 0.65 ? 'acc-ok' : 'acc-bad';
                    const dotCol = { completed: 'var(--green)', failed: 'var(--red)', running: 'var(--blue)', queued: 'var(--amber)' }[j.status] || 'var(--text-3)';
                    return `<tr onclick="viewExperiment('${j.job_id}')">
                        <td class="mono"><span class="sdot" style="background:${dotCol}"></span>${j.job_id}</td>
                        <td>${j.config?.mil_method || '—'}</td>
                        <td>${j.config?.feature_backbone || '—'}</td>
                        <td>${j.config?.mammoth?.enabled ? 'yes' : 'no'}</td>
                        <td><span class="${accClass}">${accStr}</span></td>
                        <td>${j.status}</td>
                    </tr>`;
                }).join('');
                expTableHtml = `<p class="sec-title" style="margin-top:20px">Experiments</p>
                <table class="exp-tbl">
                    <thead><tr><th>Job ID</th><th>Method</th><th>Backbone</th><th>MAMMOTH</th><th>Best Acc</th><th>Status</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>`;
            }

            wsEl.innerHTML = `<div style="width:100%">
                <div class="stat-bar">
                    <div class="stat-item"><div class="sv">${ds.length}</div><div class="sl">Datasets</div></div>
                    <div class="stat-item"><div class="sv">${exps.length}</div><div class="sl">Experiments</div></div>
                    <div class="stat-item"><div class="sv">${modelCount}</div><div class="sl">Models</div></div>
                    <div class="stat-item ${runningJobs.length ? 'live' : ''}"><div class="sv">${runningJobs.length}</div><div class="sl">Running</div></div>
                </div>
                ${activeJobData ? `<p class="sec-title">Active Job</p>${jobHtml}` : ''}
                <p class="sec-title" ${activeJobData ? 'style="margin-top:16px"' : ''}>Quick Actions</p>
                <div class="qa-row" style="margin-bottom:16px">
                    <button class="qa-btn" onclick="sendFromUI('Scan my data directory and register any new datasets')">Scan datasets</button>
                    <button class="qa-btn" onclick="sendFromUI('Search TCGA for BRCA whole slide images')">TCGA-BRCA search</button>
                    <button class="qa-btn" onclick="sendFromUI('What is the current system status?')">System status</button>
                    <button class="qa-btn" onclick="sendFromUI('Show me the recommended training config for subtyping with 200 slides')">Config advice</button>
                </div>
                ${expTableHtml}
            </div>`;
        }

        // === Configure ===
        let cfgMode = 'beginner';
        let cfgData = {};

        async function renderConfig() {
            wsEl.className = 'ws-content fill';
            wsEl.innerHTML = '<div class="cfg-form"><p style="color:var(--text-3);font-size:13px">Loading config space...</p></div>';
            const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            const [dsRes, milRes, bbRes, defRes] = await Promise.all([
                fetch(`${API}/api/datasets${sidQ}`).then(r => r.json()).catch(() => ({ datasets: [] })),
                fetch(`${API}/api/config-space/mil-methods`).then(r => r.json()).catch(() => ({ methods: [] })),
                fetch(`${API}/api/config-space/backbones`).then(r => r.json()).catch(() => ({ backbones: [] })),
                fetch(`${API}/api/config-space/defaults?mode=advanced`).then(r => r.json()).catch(() => ({ config: {} }))
            ]);
            cfgData = { datasets: dsRes.datasets || [], milMethods: milRes.methods || [], backbones: bbRes.backbones || [], defaults: defRes.config || {} };
            _paintConfig();
        }

        function _paintConfig() {
            const d = cfgData.defaults, ds = cfgData.datasets, mils = cfgData.milMethods, bbs = cfgData.backbones;
            const tr = d.training || {}, mm = d.mammoth || {}, ev = d.evaluation || {};
            const dsOpts = ds.length ? ds.map(x => `<option value="${x.id || x.name}">${x.name} (${x.feature_count || x.slide_count} ${x.feature_count ? 'features' : 'slides'})</option>`).join('') : '<option value="">— No datasets —</option>';
            const milOpts = mils.map(m => `<option value="${m.id}" ${m.id === (d.mil_method || 'abmil') ? 'selected' : ''}>${m.name}</option>`).join('');
            const bbOpts = bbs.map(b => `<option value="${b.id}" ${b.id === (d.feature_backbone || 'uni') ? 'selected' : ''}>${b.id} · dim ${b.dim}${b.gated ? ' [gated]' : ''}</option>`).join('');
            const optStr = (pairs, sel) => pairs.map(p => `<option value="${p[0]}" ${p[0] === sel ? 'selected' : ''}>${p[1]}</option>`).join('');

            if (cfgMode === 'beginner') {
                wsEl.innerHTML = `<div class="cfg-form">
                    <div class="cfg-mode-bar">
                        <span style="font-size:14px;font-weight:700;color:var(--text-1)">Training Configuration</span>
                        <div style="margin-left:auto;display:flex;gap:6px">
                            <button class="cfg-mode-btn active" onclick="setCfgMode('beginner')">Beginner</button>
                            <button class="cfg-mode-btn" onclick="setCfgMode('advanced')">Advanced</button>
                        </div>
                    </div>
                    <div class="bf">
                        <div class="bfg"><label>Dataset</label>
                            <select class="cfg-sel" id="bfDs">${dsOpts}</select>
                            ${!ds.length ? '<span style="font-size:11px;color:var(--amber)">No datasets yet — ask the agent to scan your data directory.</span>' : ''}</div>
                        <div class="bfg"><label>Task name</label>
                            <input class="cfg-txt" type="text" id="bfTask" placeholder="e.g. BRCA_subtyping">
                            <span style="font-size:11px;color:var(--text-3)">Short label for this experiment.</span></div>
                        <div class="bfg"><label>MIL method</label>
                            <select class="cfg-sel" id="bfMil">${milOpts}</select></div>
                        <div class="bfg"><label>Feature backbone</label>
                            <select class="cfg-sel" id="bfBb">${bbOpts}</select></div>
                        <div class="bfg">
                            <div class="tog-row">
                                <label class="tog"><input type="checkbox" id="bfMam" ${mm.enabled !== false ? 'checked' : ''}><span class="tog-track"></span></label>
                                <span class="tog-lbl">Enable MAMMOTH</span>
                                <span style="font-size:11px;color:var(--text-3);margin-left:6px">Mixture-of-Mini-Experts · avg +3.8% accuracy</span>
                            </div></div>
                        <div class="bfg"><label>Epochs</label>
                            <div class="srow">
                                <input type="range" class="cfg-slider" id="bfEp" min="10" max="500" step="10" value="${tr.epochs || 100}" oninput="el('bfEpV').textContent=this.value">
                                <span class="sval" id="bfEpV">${tr.epochs || 100}</span>
                            </div></div>
                        <button class="train-btn" id="bfBtn" onclick="launchTraining('beginner')">Train with Smart Defaults</button>
                        <div class="adv-lnk" onclick="setCfgMode('advanced')">Show all settings →</div>
                    </div>
                </div>`;
            } else {
                wsEl.innerHTML = `<div class="cfg-form">
                    <div class="cfg-mode-bar">
                        <span style="font-size:14px;font-weight:700;color:var(--text-1)">Training Configuration</span>
                        <div style="margin-left:auto;display:flex;gap:6px">
                            <button class="cfg-mode-btn" onclick="setCfgMode('beginner')">Beginner</button>
                            <button class="cfg-mode-btn active" onclick="setCfgMode('advanced')">Advanced</button>
                        </div>
                    </div>
                    ${_acc('data', 'Data & Features', true, `
                        ${_row('Dataset', 'Registered dataset to train on', `<select class="cfg-sel" id="afDs">${dsOpts}</select>`)}
                        ${_row('Task name', 'Short label for this experiment', `<input class="cfg-txt" type="text" id="afTask" placeholder="BRCA_subtyping">`)}
                        ${_row('Backbone', 'Foundation model for patch embeddings', `<select class="cfg-sel" id="afBb" onchange="onBbChange()">${bbOpts}</select>`)}
                        ${_row('Feature dim', 'Output dim (auto-filled by backbone)', `<input class="cfg-num" type="number" id="afDim" value="${d.feature_dim || 1024}">`)}
                        ${_row('Embed dim', 'Internal MIL embedding dimension', `<input class="cfg-num" type="number" id="afEmb" value="${d.embed_dim || 512}">`)}
                        ${_row('Num classes', 'Number of output classes', `<input class="cfg-num" type="number" id="afCls" value="${d.num_classes || 2}" min="2" max="20">`)}
                    `)}
                    ${_acc('model', 'Model', true, `
                        ${_row('MIL method', 'Bag aggregation architecture', `<select class="cfg-sel" id="afMil" onchange="_updMilDesc()">${milOpts}</select><div id="afMilDesc" style="font-size:11px;color:var(--text-3);margin-top:4px"></div>`)}
                        ${_row('Attention dim', 'Hidden dim for gated attention', `<input class="cfg-num" type="number" id="afAttn" value="128" min="32" max="512">`)}
                    `)}
                    ${_acc('mam', 'MAMMOTH', true, `
                        ${_row('Enabled', 'Replace linear layer with MoE module', `<div class="tog-row"><label class="tog"><input type="checkbox" id="afMam" ${mm.enabled !== false ? 'checked' : ''}><span class="tog-track"></span></label><span class="tog-lbl">${mm.enabled !== false ? 'Active (recommended)' : 'Disabled'}</span></div>`)}
                        ${_row('Num experts', 'Low-rank expert matrices (5–100)', `<div class="srow"><input type="range" class="cfg-slider" id="afExp" min="5" max="100" step="5" value="${mm.num_experts || 30}" oninput="el('afExpV').textContent=this.value"><span class="sval" id="afExpV">${mm.num_experts || 30}</span></div>`)}
                        ${_row('Num slots', 'Routing slots per expert (1–30)', `<div class="srow"><input type="range" class="cfg-slider" id="afSlots" min="1" max="30" step="1" value="${mm.num_slots || 10}" oninput="el('afSlotsV').textContent=this.value"><span class="sval" id="afSlotsV">${mm.num_slots || 10}</span></div>`)}
                        ${_row('Num heads', 'Attention heads for expert routing', `<div class="srow"><input type="range" class="cfg-slider" id="afHds" min="1" max="32" step="1" value="${mm.num_heads || 16}" oninput="el('afHdsV').textContent=this.value"><span class="sval" id="afHdsV">${mm.num_heads || 16}</span></div>`)}
                        ${_row('Share LoRA', 'Share first projection across experts', `<div class="tog-row"><label class="tog"><input type="checkbox" id="afSL" ${mm.share_lora_weights !== false ? 'checked' : ''}><span class="tog-track"></span></label></div>`)}
                        ${_row('Auto rank', 'Compute LoRA rank from dimensions', `<div class="tog-row"><label class="tog"><input type="checkbox" id="afAR" ${mm.auto_rank !== false ? 'checked' : ''}><span class="tog-track"></span></label></div>`)}
                        ${_row('LoRA rank', 'Manual rank (0=auto). Typical: 8–64', `<input class="cfg-num" type="number" id="afRank" value="${mm.rank || 0}" min="0" max="256">`)}
                        ${_row('Dropout', 'Within MAMMOTH module (0–0.5)', `<div class="srow"><input type="range" class="cfg-slider" id="afDrop" min="0" max="0.5" step="0.05" value="${mm.dropout || 0.1}" oninput="el('afDropV').textContent=this.value"><span class="sval" id="afDropV">${mm.dropout || 0.1}</span></div>`)}
                        ${_row('Temperature', 'Routing softmax temperature', `<div class="srow"><input type="range" class="cfg-slider" id="afTemp" min="0.1" max="5" step="0.1" value="${mm.temperature || 1.0}" oninput="el('afTempV').textContent=this.value"><span class="sval" id="afTempV">${mm.temperature || 1.0}</span></div>`)}
                    `)}
                    ${_acc('train', 'Training', true, `
                        ${_row('Epochs', 'Training epochs (10–500)', `<div class="srow"><input type="range" class="cfg-slider" id="afEp" min="10" max="500" step="10" value="${tr.epochs || 100}" oninput="el('afEpV').textContent=this.value"><span class="sval" id="afEpV">${tr.epochs || 100}</span></div>`)}
                        ${_row('Learning rate', 'Initial LR', `<input class="cfg-num" type="number" id="afLr" value="${tr.lr || 0.0001}" step="0.00001">`)}
                        ${_row('Weight decay', 'L2 regularization', `<input class="cfg-num" type="number" id="afWd" value="${tr.weight_decay || 0.00001}" step="0.000001">`)}
                        ${_row('Optimizer', 'Gradient descent optimizer', `<select class="cfg-sel" id="afOpt">${optStr([['adam', 'Adam (default)'], ['adamw', 'AdamW'], ['sgd', 'SGD + momentum'], ['radam', 'RAdam']], tr.optimizer || 'adam')}</select>`)}
                        ${_row('Scheduler', 'LR schedule', `<select class="cfg-sel" id="afSch">${optStr([['cosine', 'Cosine Annealing'], ['step', 'Step LR (1/3 intervals)'], ['plateau', 'Reduce on Plateau'], ['none', 'None (constant)']], tr.scheduler || 'cosine')}</select>`)}
                        ${_row('Early stopping', 'Patience epochs (0=disabled)', `<input class="cfg-num" type="number" id="afPat" value="${tr.early_stopping_patience || 0}" min="0" max="100">`)}
                    `)}
                    ${_acc('eval', 'Evaluation', false, `
                        ${_row('Strategy', 'Evaluation methodology', `<select class="cfg-sel" id="afEval">${optStr([['holdout', 'Holdout (80/20 stratified)'], ['5-fold-cv', '5-fold cross-validation'], ['3-fold-cv', '3-fold cross-validation'], ['10-fold-cv', '10-fold cross-validation']], ev.strategy || 'holdout')}</select>`)}
                    `)}
                    <div id="jsonBox" style="display:none"></div>
                    <div class="cfg-acts">
                        <button class="prev-btn" onclick="previewCfg()">Preview JSON</button>
                        <button class="train-btn" id="afBtn" onclick="launchTraining('advanced')">Launch Training</button>
                    </div>
                </div>`;
                _updMilDesc();
            }
        }

        function _acc(id, title, open, body) {
            return `<div class="cfg-section">
                <div class="cfg-sect-head ${open ? 'open' : ''}" onclick="togAcc('${id}')">
                    <span class="cht">${title}</span><span class="chv">▼</span>
                </div>
                <div class="cfg-sect-body ${open ? 'open' : ''}" id="acc_${id}">${body}</div>
            </div>`;
        }
        function _row(lbl, hint, ctrl) {
            return `<div class="cfg-row">
                <div class="cfg-lbl">${lbl}<div class="cfg-hint">${hint}</div></div>
                <div class="cfg-ctrl">${ctrl}</div>
            </div>`;
        }
        function togAcc(id) { const b = el(`acc_${id}`), h = b.previousElementSibling; b.classList.toggle('open'); h.classList.toggle('open'); }
        function _updMilDesc() {
            const v = el('afMil')?.value, d = el('afMilDesc');
            const m = (cfgData.milMethods || []).find(x => x.id === v);
            if (d && m) d.textContent = m.description + (m.best_for ? ` Best for: ${m.best_for}` : '');
        }
        function onBbChange() {
            const v = el('afBb')?.value, b = (cfgData.backbones || []).find(x => x.id === v);
            if (b && el('afDim')) el('afDim').value = b.dim;
        }
        function setCfgMode(m) { cfgMode = m; _paintConfig(); }

        function _gatherCfg(mode) {
            if (mode === 'beginner') {
                const bb = el('bfBb')?.value || 'uni', b = (cfgData.backbones || []).find(x => x.id === bb) || { dim: 1024 };
                const mam = el('bfMam')?.checked !== false;
                return {
                    dataset_id: el('bfDs')?.value || '', task: el('bfTask')?.value.trim() || 'experiment',
                    mil_method: el('bfMil')?.value || 'abmil', feature_backbone: bb, feature_dim: b.dim,
                    embed_dim: 512, num_classes: 2,
                    mammoth: { enabled: mam, num_experts: mam ? 30 : 10, num_slots: 10, num_heads: 16, share_lora_weights: true, auto_rank: true, dropout: 0.1, rank: 0, temperature: 1.0 },
                    training: { epochs: parseInt(el('bfEp')?.value || '100'), lr: 1e-4, weight_decay: 1e-5, optimizer: 'adam', scheduler: 'cosine', early_stopping_patience: 0 },
                    evaluation: { strategy: 'holdout' }
                };
            }
            const g = id => el(id);
            const bb = g('afBb')?.value || 'uni', b = (cfgData.backbones || []).find(x => x.id === bb) || { dim: 1024 };
            return {
                dataset_id: g('afDs')?.value || '', task: g('afTask')?.value.trim() || 'experiment',
                mil_method: g('afMil')?.value || 'abmil', feature_backbone: bb,
                feature_dim: parseInt(g('afDim')?.value || b.dim), embed_dim: parseInt(g('afEmb')?.value || '512'),
                num_classes: parseInt(g('afCls')?.value || '2'), attn_dim: parseInt(g('afAttn')?.value || '128'),
                mammoth: {
                    enabled: g('afMam')?.checked || false, num_experts: parseInt(g('afExp')?.value || '30'),
                    num_slots: parseInt(g('afSlots')?.value || '10'), num_heads: parseInt(g('afHds')?.value || '16'),
                    share_lora_weights: g('afSL')?.checked !== false, auto_rank: g('afAR')?.checked !== false,
                    rank: parseInt(g('afRank')?.value || '0'), dropout: parseFloat(g('afDrop')?.value || '0.1'),
                    temperature: parseFloat(g('afTemp')?.value || '1.0')
                },
                training: {
                    epochs: parseInt(g('afEp')?.value || '100'), lr: parseFloat(g('afLr')?.value || '0.0001'),
                    weight_decay: parseFloat(g('afWd')?.value || '0.00001'), optimizer: g('afOpt')?.value || 'adam',
                    scheduler: g('afSch')?.value || 'cosine', early_stopping_patience: parseInt(g('afPat')?.value || '0')
                },
                evaluation: { strategy: g('afEval')?.value || 'holdout' }
            };
        }

        function previewCfg() {
            const box = el('jsonBox'); if (!box) return;
            if (box.style.display !== 'none') { box.style.display = 'none'; return; }
            const cfg = _gatherCfg('advanced');
            box.innerHTML = `<div class="json-pre">${JSON.stringify(cfg, null, 2).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`;
            box.style.display = 'block';
        }

        async function launchTraining(mode) {
            const cfg = _gatherCfg(mode);
            if (!cfg.dataset_id) { showError('Please select a dataset.'); return; }
            const btnId = mode === 'beginner' ? 'bfBtn' : 'afBtn', btn = el(btnId);
            if (btn) { btn.disabled = true; btn.textContent = 'Launching...'; }
            try {
                const resp = await fetch(`${API}/api/training/start`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg)
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || JSON.stringify(data));
                // Switch to overview and start live polling
                document.querySelector('.ws-tab[data-t="overview"]').click();
                startJobPolling(data.job_id);
                addMsg(`Training launched.\n\n**Job ID:** \`${data.job_id}\`\n**Status:** ${data.status}\n\nMonitoring live in Overview. Ask me for analysis when done.`, 'agent');
            } catch (e) {
                addMsg(`Launch failed: ${e.message}`, 'agent');
            } finally {
                if (btn) { btn.disabled = false; btn.textContent = mode === 'beginner' ? 'Train with Smart Defaults' : 'Launch Training'; }
            }
        }

        // === Plots (with SVG training curve fallback) ===
        async function renderPlots() {
            wsEl.className = 'ws-content fill';
            wsEl.innerHTML = '<div style="width:100%"><p style="color:var(--text-3);font-size:13px">Loading...</p></div>';
            const r = await fetch(`${API}/api/artifacts`).catch(() => null);
            const arts = r?.ok ? (await r.json()).artifacts || [] : [];
            if (!arts.length) {
                wsEl.innerHTML = '<div class="ws-empty"><h3>No experiments yet</h3><p>Train a model to see evaluation results here.</p></div>';
                wsEl.className = 'ws-content'; return;
            }
            // If activeJobId is set, show only that experiment; otherwise show all
            const toShow = activeJobId
                ? arts.filter(a => a.experiment_id === activeJobId)
                : arts.slice().reverse();
            if (!toShow.length && activeJobId) {
                // activeJobId not in artifacts yet — show all
                arts.slice().reverse().forEach(a => toShow.push(a));
            }

            let html = '<div style="width:100%">';
            for (const a of toShow) {
                // Get status + history
                const sr = await fetch(`${API}/api/training/${a.experiment_id}`).catch(() => null);
                const status = sr?.ok ? await sr.json() : {};
                const history = _extractHistory(status);

                // Try training plots first, then eval plots
                const tpr = await fetch(`${API}/api/training/${a.experiment_id}/plots`).catch(() => null);
                const trainingPlots = tpr?.ok ? (await tpr.json()).plots || [] : [];
                const epr = await fetch(`${API}/api/eval/${a.experiment_id}/plots`).catch(() => null);
                const evalPlots = epr?.ok ? (await epr.json()).plots || [] : [];
                const plots = trainingPlots.length ? trainingPlots : evalPlots;
                const plotBase = trainingPlots.length
                    ? `${API}/api/training/${a.experiment_id}/plots`
                    : `${API}/api/eval/${a.experiment_id}/plots`;

                html += `<div class="sec-block">
                    <p class="sec-title" style="display:flex;justify-content:space-between">
                        <span>${a.experiment_id}</span>
                        <span style="font-size:10px;font-weight:400;color:var(--text-3)">
                            ${status.config?.mil_method || ''} · ${status.config?.feature_backbone || ''}
                            ${status.metrics?.best_val_accuracy ? ' · best acc ' + (status.metrics.best_val_accuracy * 100).toFixed(1) + '%' : ''}
                        </span>
                    </p>`;

                if (plots.length) {
                    html += `<div class="plot-grid">${plots.map(p => `
                        <div class="thumb-card">
                            <img src="${plotBase}/${p.name}" loading="lazy" alt="${p.name}" onclick="_showLightbox(this.src)">
                            <div class="thumb-lbl">${p.name}</div>
                            <button class="btn-tiny" onclick="sendPlotToManuscript('${a.experiment_id}','${p.name}', this)">→ Manuscript</button>
                        </div>`).join('')}</div>`;
                } else if (history.train_loss?.length) {
                    // SVG training curve fallback
                    html += `<div class="chart-wrap">
                        ${_svgCurve(history)}
                        <div class="chart-legend">
                            <span><i style="background:var(--blue)"></i>train loss</span>
                            <span><i style="background:var(--amber)"></i>val loss</span>
                            <span><i style="background:var(--green)"></i>val accuracy</span>
                        </div>
                    </div>`;
                    // Metrics table
                    if (history.train_loss.length) {
                        html += `<table class="metrics-tbl"><thead><tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Val Acc</th></tr></thead><tbody>`;
                        html += history.train_loss.map((tl, i) =>
                            `<tr><td>${i + 1}</td><td>${tl.toFixed(4)}</td><td>${(history.val_loss[i] || 0).toFixed(4)}</td><td>${((history.val_acc[i] || 0) * 100).toFixed(1)}%</td></tr>`
                        ).join('');
                        html += '</tbody></table>';
                    }
                } else {
                    html += `<p class="empty-state">No plots or history data yet</p>`;
                }
                html += '</div>';
            }
            wsEl.innerHTML = html + '</div>';
        }

        function _extractHistory(status) {
            // Try direct history fields on status, or empty
            if (status.history) return status.history;
            // Build from epoch-by-epoch if available
            return { train_loss: [], val_loss: [], val_acc: [] };
        }

        function _svgCurve(history) {
            const tl = history.train_loss || [], vl = history.val_loss || [], va = history.val_acc || [];
            const n = Math.max(tl.length, vl.length, va.length);
            if (!n) return '';
            const W = 560, H = 160, pl = 42, pt = 10, pb = 24, pr = 12;
            const iW = W - pl - pr, iH = H - pt - pb;
            const xS = i => pl + (n > 1 ? (i / (n - 1)) : 0.5) * iW;

            // Dual Y axes: loss (left) and acc (right)
            const lossVals = [...tl, ...vl].filter(v => v !== undefined);
            const minL = Math.min(...lossVals), maxL = Math.max(...lossVals);
            const rangeL = maxL - minL || 0.001;
            const yL = v => pt + iH - ((v - minL) / rangeL) * iH;

            const minA = 0, maxA = 1;
            const yA = v => pt + iH - ((v - minA) / (maxA - minA)) * iH;

            const poly = (pts) => pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
            const tlPts = tl.map((v, i) => [xS(i), yL(v)]);
            const vlPts = vl.map((v, i) => [xS(i), yL(v)]);
            const vaPts = va.map((v, i) => [xS(i), yA(v)]);

            // Y ticks
            const lTicks = [minL, (minL + maxL) / 2, maxL].map(v =>
                `<text x="${pl - 5}" y="${yL(v) + 3}" fill="var(--text-3)" font-size="8" text-anchor="end">${v.toFixed(3)}</text>`).join('');
            // X ticks
            const step = Math.max(1, Math.floor(n / 5));
            const xTicks = Array.from({ length: n }, (_, i) => i).filter(i => i % step === 0 || i === n - 1)
                .map(i => `<text x="${xS(i).toFixed(1)}" y="${H - 5}" fill="var(--text-3)" font-size="8" text-anchor="middle">${i + 1}</text>`).join('');

            return `<svg width="100%" viewBox="0 0 ${W} ${H}" style="overflow:visible">
                <line x1="${pl}" y1="${pt}" x2="${pl}" y2="${pt + iH}" stroke="var(--border)" stroke-width="0.5"/>
                <line x1="${pl}" y1="${pt + iH}" x2="${pl + iW}" y2="${pt + iH}" stroke="var(--border)" stroke-width="0.5"/>
                ${lTicks}${xTicks}
                ${tlPts.length > 1 ? `<polyline points="${poly(tlPts)}" fill="none" stroke="var(--blue)" stroke-width="1.5" stroke-linejoin="round"/>` : ''}
                ${vlPts.length > 1 ? `<polyline points="${poly(vlPts)}" fill="none" stroke="var(--amber)" stroke-width="1.5" stroke-linejoin="round"/>` : ''}
                ${vaPts.length > 1 ? `<polyline points="${poly(vaPts)}" fill="none" stroke="var(--green)" stroke-width="1.5" stroke-linejoin="round" stroke-dasharray="3,2"/>` : ''}
            </svg>`;
        }

        // === Slides ===
        async function renderSlides() {
            wsEl.className = 'ws-content fill';
            wsEl.innerHTML = '<div style="width:100%"><p style="color:var(--text-3);font-size:13px">Loading...</p></div>';
            const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            const r = await fetch(`${API}/api/datasets${sidQ}`).catch(() => null);
            const datasets = r?.ok ? (await r.json()).datasets || [] : [];
            if (!datasets.length) {
                wsEl.innerHTML = '<div class="ws-empty"><h3>No datasets yet</h3><p>Register a dataset and preprocess slides to see tissue previews here.</p></div>';
                wsEl.className = 'ws-content'; return;
            }
            let html = '<div style="width:100%">';
            for (const ds of datasets) {
                html += `<div class="sec-block"><p class="sec-title">${ds.name} — ${ds.slide_count} slides${ds.feature_count ? ` · ${ds.feature_count} features` : ''}</p>`;
                const pr = await fetch(`${API}/api/preprocess/preview/${ds.id}`).catch(() => null);
                const prev = pr?.ok ? (await pr.json()).previews || [] : [];
                html += prev.length
                    ? `<div class="slide-grid">${prev.slice(0, 12).map(p => `<div class="thumb-card"><img class="lb" src="${p.url}" loading="lazy" alt="${p.slide}" title="Click to enlarge" style="cursor:zoom-in"><div class="thumb-lbl">${p.slide}</div></div>`).join('')}</div>`
                    : `<p class="empty-state">Not preprocessed yet — features available for training.</p>`;
                html += '</div>';
            }
            wsEl.innerHTML = html + '</div>';
        }

        // === Logs ===
        async function renderLogs() {
            wsEl.className = 'ws-content logs-mode';
            // Get artifacts to build selector
            const r = await fetch(`${API}/api/artifacts`).catch(() => null);
            const arts = r?.ok ? (await r.json()).artifacts || [] : [];
            if (!arts.length) {
                wsEl.innerHTML = '<div class="ws-empty"><h3>No experiments yet</h3><p>Train a model to see logs here.</p></div>';
                wsEl.className = 'ws-content'; return;
            }
            const opts = arts.slice().reverse().map(a =>
                `<option value="${a.experiment_id}" ${a.experiment_id === activeJobId ? 'selected' : ''}>${a.experiment_id}</option>`
            ).join('');
            wsEl.innerHTML = `<div class="log-wrap">
                <div class="log-sel-bar">
                    <label>Experiment</label>
                    <select class="cfg-sel" id="logJobSel" style="width:auto;min-width:200px" onchange="loadLog(this.value)">${opts}</select>
                    <button class="qa-btn" onclick="loadLog(el('logJobSel').value)" style="margin-left:4px">Refresh</button>
                </div>
                <div id="logContent" class="log-viewer">Loading...</div>
            </div>`;
            const selectedId = activeJobId && arts.find(a => a.experiment_id === activeJobId)
                ? activeJobId
                : arts[arts.length - 1].experiment_id;
            loadLog(selectedId);
        }

        async function loadLog(jobId) {
            const logEl = el('logContent');
            if (!logEl) return;
            // Reflect the selected job in the dropdown (in case loadLog was called from an external trigger)
            const sel = el('logJobSel');
            if (sel && sel.value !== jobId) sel.value = jobId;
            logEl.textContent = 'Loading...';
            try {
                const [lr, sr] = await Promise.all([
                    fetch(`${API}/api/training/${jobId}/logs`),
                    fetch(`${API}/api/training/${jobId}`).catch(() => null),
                ]);
                const d = lr.ok ? await lr.json() : { logs: '' };
                const status = sr?.ok ? await sr.json() : null;

                let header = '';
                if (status) {
                    const pct = Math.round((status.progress || 0) * 100);
                    const cfg = status.config || {};
                    const metrics = status.metrics || {};
                    const best = metrics.best_val_accuracy != null ? ` · best val acc ${(metrics.best_val_accuracy*100).toFixed(1)}%` : '';
                    header = `<div class="log-status-hdr">
                        <span class="log-pill ${status.status}">${status.status || 'unknown'}</span>
                        <span>${cfg.mil_method || ''} · ${cfg.feature_backbone || ''}</span>
                        <span>epoch ${status.epoch || 0} / ${status.total_epochs || '?'} · ${pct}%${best}</span>
                    </div>`;
                }

                const raw = d.logs || '';
                const colored = raw.split('\n').map(line => {
                    if (/error|failed|exception/i.test(line)) return `<span class="log-err">${line}</span>`;
                    if (/epoch \d+.*complete|saved|best|success/i.test(line)) return `<span class="log-ok">${line}</span>`;
                    if (/^#|loading|initializ/i.test(line)) return `<span class="log-dim">${line}</span>`;
                    return line;
                }).join('\n');

                let body = colored;
                if (!raw.trim()) {
                    if (status && status.status === 'running') {
                        body = '<span class="log-dim">Training is running — logs will appear at the first epoch boundary.\n'
                             + 'For in-flight progress use the status header above, or watch the Plots tab for per-epoch curves.</span>';
                    } else {
                        body = '<span class="log-dim">No log content yet.</span>';
                    }
                }

                logEl.innerHTML = header + body;
                logEl.scrollTop = logEl.scrollHeight;
            } catch (e) {
                logEl.textContent = `Could not load logs: ${e.message}`;
            }
        }

        // === Image lightbox ===
        function _showLightbox(src) {
            const lb = document.createElement('div');
            lb.className = 'img-lightbox';
            lb.innerHTML = `<img src="${src}" alt="plot">`;
            lb.onclick = () => lb.remove();
            document.body.appendChild(lb);
        }

        async function sendPlotToManuscript(jobId, filename, btn) {
            if (!sessionId) { showError('Start or open a chat session first — manuscript is per-session.'); return; }
            const origText = btn.textContent;
            btn.disabled = true; btn.textContent = '…';
            try {
                const r = await fetch(`${API}/api/chat/manuscript/${sessionId}/attach-figure`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        job_id: jobId,
                        job_type: 'training',
                        filename,
                        caption: filename.replace(/\.(png|jpg|jpeg|pdf|svg)$/i, '').replace(/_/g, ' '),
                        insert_in_tex: true,
                    }),
                });
                // Read once as text so a non-JSON body (e.g. raw "Internal Server Error")
                // doesn't explode the frontend with a JSON parse error.
                const raw = await r.text();
                let d = null;
                try { d = raw ? JSON.parse(raw) : null; } catch { /* non-JSON */ }
                if (r.ok && d && d.status === 'ok') {
                    btn.textContent = '✓ Added';
                    btn.style.background = 'var(--green, #16a34a)';
                    btn.style.color = 'white';
                } else {
                    btn.textContent = 'Failed';
                    const detail = (d && (d.detail || d.message)) || raw || `HTTP ${r.status}`;
                    showError('Attach failed: ' + detail);
                }
            } catch (e) {
                btn.textContent = 'Failed';
                showError('Request error: ' + e.message);
            } finally {
                setTimeout(() => { btn.disabled = false; btn.textContent = origText; btn.style.background = ''; btn.style.color = ''; }, 2500);
            }
        }

        // === Active Jobs Panel ===
        // Only show genuinely active jobs (running or queued). Completed / failed / cancelled
        // are visible under the experiment tree, so keeping them here turns the sidebar into
        // noise across sessions.
        // === One-button upload (auto-routes by extension) ===
        async function onUploadAny(input) {
            const files = Array.from(input.files || []);
            if (!files.length) return;
            const fd = new FormData();
            fd.append('session_id', sessionId || '');
            for (const f of files) fd.append('files', f, f.name);

            // Toast: starting
            const totalMB = (files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(1);
            try { showError(`Uploading ${files.length} file(s) (${totalMB} MB)…`); } catch {}

            try {
                const r = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
                const txt = await r.text();
                let d = null; try { d = JSON.parse(txt); } catch {}
                if (!r.ok) {
                    showError('Upload failed: ' + ((d && d.detail) || txt));
                    return;
                }
                const lines = (d.uploaded || []).map(u => `  • ${u.filename} → ${u.category} (${(u.size_bytes / 1e6).toFixed(1)} MB)`);
                const next = (d.next_steps || []).join('\n');
                const summary = `Uploaded ${d.uploaded.length} file(s):\n${lines.join('\n')}${next ? '\n\nNext: ' + next : ''}`;
                showError(summary);
                // Refresh tree + editor so newly-routed files show up immediately.
                try { renderFileTree(true); } catch {}
                try { editorRefreshTree(); } catch {}
            } catch (e) {
                showError('Upload error: ' + e.message);
            } finally {
                input.value = '';
            }
        }

        async function refreshJobsPanel() {
            try {
                const sidQ = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
                const d = await fetch(`${API}/api/jobs/all${sidQ}`).then(r => r.json());
                const jobs = (d.jobs || []).filter(j => j.status === 'running' || j.status === 'queued');
                const panel = el('jobsPanel');
                if (!jobs.length) { panel.style.display = 'none'; panel.innerHTML = ''; return; }
                const statusCls = { running: 'running', completed: 'done', failed: 'err', partial: 'warn', queued: 'queued' };
                const typeLabel = { training: 'Training', eval: 'Evaluation', preprocess: 'Preprocess', features: 'Features', download: 'Download' };
                panel.style.display = '';
                panel.innerHTML = `<div class="jobs-panel-title">Active Jobs</div>` + jobs.slice(0, 10).map(j => {
                    const pct = Math.round((j.progress || 0) * 100);
                    const label = typeLabel[j.type] || j.type;
                    const dotCls = statusCls[j.status] || 'queued';

                    let detail = '';
                    if (j.type === 'download') {
                        const filesStr = j.total ? `${j.done || 0} / ${j.total} files` : '';
                        const gbStr = j.bytes_done ? ` · ${(j.bytes_done / 1e9).toFixed(2)} GB` : '';
                        const dir = j.output_dir ? j.output_dir.replace(/^\/home\/[^/]+/, '~') : '';
                        detail = `<div class="job-detail">${filesStr}${gbStr}${dir ? `<br>${dir}` : ''}</div>`;
                    } else if (pct) {
                        detail = `<div class="job-detail">${pct}%</div>`;
                    }

                    return `<div class="job-row">
                        <div class="job-dot ${dotCls}"></div>
                        <div class="job-row-main">
                            <div class="job-row-header">
                                <span class="job-row-type">${label}</span>
                                <span class="job-row-id">${j.job_id}</span>
                                <span class="job-row-status ${dotCls}">${j.status}</span>
                            </div>
                            ${detail}
                            <div class="job-mini-bar"><div class="job-mini-fill ${dotCls}" style="width:${pct}%"></div></div>
                        </div>
                    </div>`;
                }).join('');
            } catch (e) { /* silent */ }
        }

        // === Task-plan panel (live agent checklist) ===
        function renderTaskPlan(plan) {
            const panel = el('tasksPanel');
            if (!panel) return;
            const tasks = (plan && plan.tasks) || [];
            if (!tasks.length) { panel.style.display = 'none'; panel.innerHTML = ''; return; }
            panel.style.display = '';
            const marks = { completed: '[x]', in_progress: '[~]', pending: '[ ]', skipped: '[-]' };
            panel.innerHTML = `<div class="tasks-panel-title"><span>Task Plan</span><span class="tp-clear" onclick="clearTaskPlan()" title="Clear plan">Clear</span></div>` +
                tasks.map(t => {
                    const st = t.status || 'pending';
                    const mark = marks[st] || '[ ]';
                    const title = (t.title || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    const pause = t.pause_after ? '<span class="task-pause">pause</span>' : '';
                    return `<div class="task-row ${st}"><span class="task-mark ${st}">${mark}</span><span class="task-title">${t.id}. ${title}</span>${pause}</div>`;
                }).join('');
        }
        async function refreshTaskPlan() {
            if (!sessionId) { renderTaskPlan(null); return; }
            try {
                const r = await fetch(`${API}/api/task-plan/tasks?session_id=${encodeURIComponent(sessionId)}`);
                if (!r.ok) return;
                const plan = await r.json();
                renderTaskPlan(plan);
            } catch (e) { /* silent */ }
        }
        async function clearTaskPlan() {
            if (!sessionId) return;
            try {
                await fetch(`${API}/api/task-plan/tasks?session_id=${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
                renderTaskPlan(null);
            } catch (e) { /* silent */ }
        }
        window.clearTaskPlan = clearTaskPlan;

        // === Init ===
        checkStatus(); checkOnboard(); checkOllama(); renderFileTree(); refreshJobsPanel(); refreshTaskPlan();
        setInterval(checkStatus, 10000);
        setInterval(checkOllama, 15000);
        setInterval(renderFileTree, 30000);
        setInterval(refreshJobsPanel, 5000);
        setInterval(refreshTaskPlan, 10000);

        // Resume polling if there was an active job from a previous page load
        if (activeJobId) {
            fetch(`${API}/api/training/${activeJobId}`)
                .then(r => r.ok ? r.json() : null)
                .then(j => {
                    if (j && (j.status === 'running' || j.status === 'queued')) {
                        startJobPolling(activeJobId);
                    }
                })
                .catch(e => console.warn('Resume poll failed:', e.message));
        }

        // === Telegram connect modal ===
        async function openTelegramModal() {
            try {
                const r = await fetch(`${API}/api/telegram/status`); const d = await r.json();
                el('tgStatus').innerHTML = d.running
                    ? `<span style="color:var(--green)">● Bot running</span> (pid ${d.pid})${d.token_set ? ' — token saved' : ''}`
                    : (d.token_set ? '<span style="color:var(--text-2)">○ Stopped — token saved, click Start bot</span>' : '<span style="color:var(--text-2)">○ Not configured</span>');
                el('tgAllowed').value = d.allowed_usernames || '';
                el('tgPasscode').value = d.passcode_set ? '' : '';
                el('tgPasscode').placeholder = d.passcode_set ? '(passcode is set — leave blank to keep)' : 'e.g. pathclaw2026';
            } catch (e) { el('tgStatus').textContent = 'Status unavailable'; }
            el('tgModal').classList.remove('hidden');
        }

        async function startTelegram() {
            const token = el('tgToken').value.trim();
            const body = {
                token,
                allowed_usernames: el('tgAllowed').value.trim(),
                passcode: el('tgPasscode').value.trim(),
            };
            try {
                const r = await fetch(`${API}/api/telegram/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                const d = await r.json();
                if (r.ok) { el('tgStatus').innerHTML = `<span style="color:var(--green)">● Running</span> (pid ${d.pid})`; el('tgToken').value = ''; }
                else { el('tgStatus').innerHTML = `<span style="color:var(--red)">Error: ${d.detail || 'failed'}</span>`; }
            } catch (e) { el('tgStatus').innerHTML = `<span style="color:var(--red)">Error: ${e.message}</span>`; }
        }

        async function stopTelegram() {
            try {
                await fetch(`${API}/api/telegram/stop`, { method: 'POST' });
                el('tgStatus').innerHTML = '<span style="color:var(--text-2)">○ Stopped</span>';
            } catch (e) { }
        }

        // === Lightbox for slide thumbnails ===
        function openLightbox(src) {
            el('lightboxImg').src = src;
            el('lightbox').classList.remove('hidden');
        }
        function closeLightbox(ev) {
            if (ev && ev.target && ev.target.id && ev.target.id === 'lightboxImg') return;
            el('lightbox').classList.add('hidden');
            el('lightboxImg').src = '';
        }
        // Auto-wire thumbnails: any <img> inside a .ws-content with class 'lb' opens the lightbox
        document.addEventListener('click', (ev) => {
            const t = ev.target;
            if (t && t.tagName === 'IMG' && t.classList.contains('lb')) {
                openLightbox(t.src);
            }
        });

        // === Notebook tab renderer ===
        async function renderNotebook() {
            const ws = el('wsContent');
            ws.className = 'ws-content fill';
            if (!sessionId) { ws.innerHTML = '<div class="ws-empty">No session</div>'; return; }
            let notes = '';
            try {
                const r = await fetch(`${API}/api/chat/notes/${sessionId}`);
                const d = await r.json();
                notes = d.notes || '';
            } catch (e) { }
            const body = notes
                ? `<pre style="white-space:pre-wrap; font-family:var(--mono); font-size:12.5px; line-height:1.65; background:var(--bg-2); padding:18px; border:1px solid var(--border); border-radius:var(--r); max-height:70vh; overflow:auto">${_escape(notes)}</pre>`
                : `<div class="fld-empty-state"><div class="fld-empty-icon">📓</div><h3>Notebook is empty</h3><p>As the agent works, it will append findings, decisions, and job IDs here. Notes are injected into every round of the agent's context, so they survive message trimming.</p></div>`;
            ws.innerHTML = `
                <div class="nb-wrap">
                    <div class="nb-header">
                        <div>
                            <h2>Session Notebook</h2>
                            <p>Per-session memory — injected into every round of the agent's context. Survives message trimming.</p>
                        </div>
                        <div style="display:flex; gap:6px">
                            <button class="btn-ghost" onclick="renderNotebook()">Refresh</button>
                            ${notes ? '<button class="btn-ghost" onclick="clearNotebook()" style="color:var(--red)">Clear</button>' : ''}
                        </div>
                    </div>
                    ${body}
                </div>
            `;
        }
        async function clearNotebook() {
            if (!sessionId || !confirm('Clear notebook for this session?')) return;
            await fetch(`${API}/api/chat/notes/${sessionId}`, { method: 'DELETE' });
            renderNotebook();
        }
        function _escape(s) { return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

        // === Folders tab renderer ===
        async function renderFolders() {
            const ws = el('wsContent');
            ws.className = 'ws-content fill';
            let folders = [];
            try {
                const r = await fetch(`${API}/api/folders`); const d = await r.json();
                folders = d.folders || [];
            } catch (e) { }
            const attached = sessionId ? await fetch(`${API}/api/folders/session/${sessionId}`).then(r => r.json()).then(d => new Set((d.folders || []).map(f => f.id))).catch(() => new Set()) : new Set();
            const listHtml = folders.length
                ? `<div class="fld-list">${folders.map(f => `
                    <div class="fld-card ${attached.has(f.id) ? 'fld-attached' : ''}">
                        <div class="fld-head">
                            <div class="fld-name">
                                <span class="fld-icon">📁</span>
                                <strong>${_escape(f.name)}</strong>
                                <span class="fld-count">${f.file_count || 0} PDF${f.file_count === 1 ? '' : 's'}</span>
                                ${attached.has(f.id) ? '<span class="fld-badge">Attached</span>' : ''}
                            </div>
                            <div class="fld-actions">
                                ${sessionId ? `<button class="btn-ghost" onclick="toggleFolderAttach('${f.id}', ${attached.has(f.id)})">${attached.has(f.id) ? 'Detach' : 'Attach to session'}</button>` : ''}
                                <label class="btn-ghost"><span>Upload PDF</span><input type="file" accept="application/pdf" style="display:none" onchange="uploadPdf('${f.id}', this)"></label>
                                <button class="btn-ghost fld-del" onclick="deleteFolder('${f.id}')">Delete</button>
                            </div>
                        </div>
                        ${(f.files || []).length ? `<div class="fld-files">${f.files.map(x => `<div class="fld-file">📄 ${_escape(x.filename || x)}</div>`).join('')}</div>` : '<div class="fld-empty">No PDFs uploaded yet.</div>'}
                    </div>
                `).join('')}</div>`
                : `<div class="fld-empty-state">
                        <div class="fld-empty-icon">📁</div>
                        <h3>No folders yet</h3>
                        <p>Folders hold PDFs you want the agent to read — papers, protocols, prior drafts. Attach a folder to a chat and the agent automatically sees what's inside.</p>
                        <button class="btn-primary" onclick="createFolder()">+ Create your first folder</button>
                   </div>`;
            ws.innerHTML = `
                <div class="fld-wrap">
                    <div class="fld-header">
                        <div>
                            <h2>Folders</h2>
                            <p>Upload PDFs the agent can read. Attach a folder to the current session to auto-expose its contents.</p>
                        </div>
                        ${folders.length ? '<button class="btn-primary" onclick="createFolder()">+ New Folder</button>' : ''}
                    </div>
                    ${listHtml}
                </div>
            `;
        }
        async function createFolder() {
            const name = prompt('Folder name:');
            if (!name) return;
            await fetch(`${API}/api/folders`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
            renderFolders();
        }
        async function deleteFolder(id) {
            if (!confirm('Delete this folder and its PDFs?')) return;
            await fetch(`${API}/api/folders/${id}`, { method: 'DELETE' });
            renderFolders();
        }
        async function uploadPdf(folderId, input) {
            const file = input.files[0]; if (!file) return;
            const fd = new FormData(); fd.append('file', file);
            try {
                const r = await fetch(`${API}/api/folders/${folderId}/upload`, { method: 'POST', body: fd });
                if (!r.ok) { const e = await r.json().catch(() => ({})); showError(`Upload failed: ${e.detail || r.status}`); }
            } catch (e) { showError(`Upload failed: ${e.message}`); }
            renderFolders();
        }
        async function toggleFolderAttach(folderId, isAttached) {
            const action = isAttached ? 'detach' : 'attach';
            await fetch(`${API}/api/folders/${folderId}/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId }) });
            renderFolders();
        }

        // === Manuscript (LaTeX) tab renderer ===
        let _msActiveFile = '';
        let _msPdfUrl = '';
        async function renderManuscript() {
            const ws = el('wsContent');
            ws.className = 'ws-content fill';
            if (!sessionId) { ws.innerHTML = '<div class="ws-empty">No session — start a chat first.</div>'; return; }
            let files = [];
            try {
                const r = await fetch(`${API}/api/chat/manuscript/${sessionId}`);
                const d = await r.json();
                files = d.files || [];
            } catch (e) { }
            if (!_msActiveFile && files.length) _msActiveFile = (files.find(f => f.name === 'main.tex') || files[0]).name;
            const fileList = files.length
                ? files.map(f => `<div class="ms-file ${f.name === _msActiveFile ? 'active' : ''}" onclick="msOpenFile('${f.name.replace(/'/g, "\\'")}')">${_escape(f.name)} <span style="color:var(--text-3); font-size:10px">${f.size}b</span></div>`).join('')
                : '<div style="color:var(--text-3); font-size:12px; padding:6px">No files yet</div>';
            let content = '';
            if (_msActiveFile) {
                try {
                    const r = await fetch(`${API}/api/chat/manuscript/${sessionId}/file/${encodeURIComponent(_msActiveFile)}`);
                    content = r.ok ? await r.text() : '';
                } catch (e) { }
            }
            const pdfPane = _msPdfUrl
                ? `<iframe src="${_msPdfUrl}" style="width:100%; height:100%; border:none; background:#fff"></iframe>`
                : '<div class="ws-empty" style="font-size:12px">Compile to preview PDF here.</div>';
            ws.innerHTML = `
                <div style="display:flex; flex-direction:column; height:100%; width:100%; max-width:1200px">
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-bottom:1px solid var(--border)">
                        <h3 style="margin:0; font-size:14px">Manuscript (LaTeX)</h3>
                        <div style="display:flex; gap:6px">
                            <button class="btn-ghost" style="font-size:11px" onclick="msNewFile()">+ New file</button>
                            <button class="btn-ghost" style="font-size:11px" onclick="msStartTemplate()">Insert template</button>
                            <button class="btn-ghost" style="font-size:11px" onclick="renderManuscript()">Refresh</button>
                            <button class="btn-ghost" style="font-size:11px" onclick="msExportZip()">Export .zip</button>
                            <button class="btn-primary" style="font-size:11px" onclick="msCompile()">Compile PDF</button>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns:180px 1fr 1fr; gap:0; flex:1; min-height:0">
                        <div style="border-right:1px solid var(--border); overflow:auto; padding:8px">
                            <div style="font-size:10px; color:var(--text-3); text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px">Files</div>
                            ${fileList}
                        </div>
                        <div style="display:flex; flex-direction:column; border-right:1px solid var(--border); min-width:0">
                            <div style="padding:6px 10px; font-size:11px; color:var(--text-2); border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center">
                                <span>${_msActiveFile ? _escape(_msActiveFile) : '(no file)'}</span>
                                ${_msActiveFile ? `<div style="display:flex; gap:4px"><button class="btn-ghost" style="font-size:10px" onclick="msSaveFile()">Save</button><button class="btn-ghost" style="font-size:10px; color:var(--red)" onclick="msDeleteFile()">Delete</button></div>` : ''}
                            </div>
                            <textarea id="msEditor" style="flex:1; background:var(--bg-1); color:var(--text-0); border:none; padding:10px; font-family:var(--mono); font-size:12px; resize:none; outline:none; min-height:0">${_escape(content)}</textarea>
                        </div>
                        <div id="msPdfPane" style="background:#222">${pdfPane}</div>
                    </div>
                    <div id="msLog" style="border-top:1px solid var(--border); padding:8px 12px; font-size:11px; color:var(--text-2); background:var(--bg-2); max-height:120px; overflow:auto; display:none"></div>
                </div>
            `;
            // Inject a small style for the file rows (inline to keep patch contained)
            if (!el('msStyleTag')) {
                const s = document.createElement('style');
                s.id = 'msStyleTag';
                s.textContent = `.ms-file{padding:4px 6px;border-radius:4px;cursor:pointer;font-size:12px;color:var(--text-1);margin-bottom:2px}.ms-file:hover{background:var(--bg-hover)}.ms-file.active{background:var(--blue-dim);color:var(--blue)}`;
                document.head.appendChild(s);
            }
        }
        async function msOpenFile(name) { _msActiveFile = name; await renderManuscript(); }
        async function msSaveFile() {
            if (!_msActiveFile) return;
            const content = el('msEditor').value;
            const r = await fetch(`${API}/api/chat/manuscript/${sessionId}/file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: _msActiveFile, content, mode: 'write' }),
            });
            if (!r.ok) { showError('Save failed'); return; }
            _msShowLog(`Saved ${_msActiveFile}`);
        }
        async function msDeleteFile() {
            if (!_msActiveFile || !confirm(`Delete ${_msActiveFile}?`)) return;
            await fetch(`${API}/api/chat/manuscript/${sessionId}/file/${encodeURIComponent(_msActiveFile)}`, { method: 'DELETE' });
            _msActiveFile = '';
            renderManuscript();
        }
        async function msNewFile() {
            const name = prompt('New file name (e.g. main.tex, sections/intro.tex, refs.bib):');
            if (!name) return;
            await fetch(`${API}/api/chat/manuscript/${sessionId}/file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: name, content: '', mode: 'write' }),
            });
            _msActiveFile = name;
            renderManuscript();
        }
        async function msStartTemplate() {
            const tpl = `\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{booktabs}\n\n\\title{Working Title}\n\\author{Your Name}\n\\date{\\today}\n\n\\begin{document}\n\\maketitle\n\n\\begin{abstract}\nShort abstract here.\n\\end{abstract}\n\n\\section{Introduction}\nWrite the motivation and prior work.\n\n\\section{Methods}\nDatasets, preprocessing, model, training.\n\n\\section{Results}\nMetrics, figures, tables.\n\n\\section{Discussion}\nInterpretation and limitations.\n\n\\bibliographystyle{plain}\n\\bibliography{refs}\n\n\\end{document}\n`;
            await fetch(`${API}/api/chat/manuscript/${sessionId}/file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: 'main.tex', content: tpl, mode: 'write' }),
            });
            _msActiveFile = 'main.tex';
            renderManuscript();
        }
        function msExportZip() {
            window.location.href = `${API}/api/chat/manuscript/${sessionId}/export`;
        }
        async function msCompile() {
            if (_msActiveFile) await msSaveFile();  // auto-save current edits
            // Make sure main.tex exists — if the project is empty, scaffold the template.
            let files = [];
            try {
                const r = await fetch(`${API}/api/chat/manuscript/${sessionId}`);
                files = (await r.json()).files || [];
            } catch (e) { }
            const hasMain = files.some(f => f.name === 'main.tex');
            if (!hasMain) {
                if (files.length === 0) {
                    _msShowLog('Manuscript empty — scaffolding main.tex from the default template...');
                    await msStartTemplate();
                } else {
                    _msShowLog(`No main.tex found. Your files: ${files.map(f => f.name).join(', ')}. Rename your top-level file to main.tex, or use "Insert template" to create one.`);
                    return;
                }
            }
            _msShowLog('Compiling...');
            try {
                const r = await fetch(`${API}/api/chat/manuscript/${sessionId}/compile?main_file=main.tex`, { method: 'POST' });
                const d = await r.json();
                if (d.status === 'ok' && d.pdf_url) {
                    _msPdfUrl = d.pdf_url + '?t=' + Date.now();
                    _msShowLog(`Compiled with ${d.compiler} → PDF ready.`);
                    const pane = el('msPdfPane');
                    if (pane) pane.innerHTML = `<iframe src="${_msPdfUrl}" style="width:100%; height:100%; border:none; background:#fff"></iframe>`;
                } else {
                    _msShowLog('Compile failed:\n' + (d.log || 'unknown error'));
                }
            } catch (e) { _msShowLog('Compile error: ' + e.message); }
        }
        function _msShowLog(text) {
            const lg = el('msLog');
            if (!lg) return;
            lg.style.display = 'block';
            lg.textContent = text;
        }

