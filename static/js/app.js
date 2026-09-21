/**
 * NEURAL HUD // REAL-TIME PREDICTION & SPELLCHECK CONTROLLER (FASTAPI V2.1)
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const inputText = document.getElementById('inputText');
    const ghostPredicted = document.getElementById('ghostPredicted');
    const candidatesContainer = document.getElementById('candidatesContainer');
    const latencyVal = document.getElementById('latencyVal');
    const tokenCount = document.getElementById('tokenCount');
    const activeModelTag = document.getElementById('activeModelTag');
    const modelSelector = document.getElementById('modelSelector');
    
    // Sliders
    const tempSlider = document.getElementById('tempSlider');
    const tempVal = document.getElementById('tempVal');
    const topkSlider = document.getElementById('topkSlider');
    const topkVal = document.getElementById('topkVal');
    const repSlider = document.getElementById('repSlider');
    const repVal = document.getElementById('repVal');
    const toppSlider = document.getElementById('toppSlider');
    const toppVal = document.getElementById('toppVal');
    const genWordsSlider = document.getElementById('genWordsSlider');
    const genWordsVal = document.getElementById('genWordsVal');
    
    // Spellcheck elements
    const spellcheckBanner = document.getElementById('spellcheckBanner');
    const spellcheckItems = document.getElementById('spellcheckItems');

    // Action buttons
    const btnGenerate = document.getElementById('btnGenerate');
    const btnClear = document.getElementById('btnClear');
    const genResultBox = document.getElementById('genResultBox');
    const genResultContent = document.getElementById('genResultContent');
    const specArch = document.getElementById('specArch');
    const specSeqLen = document.getElementById('specSeqLen');
    const specVocab = document.getElementById('specVocab');

    let debounceTimer = null;
    let currentTopWord = '';
    let availableModels = {};

    // 1. Initialize Neural Particles Canvas
    initNeuralCanvas();

    // 2. Fetch Models and Populate Selector
    fetchModels();

    // 3. Event Listeners for Typing & Autocomplete
    inputText.addEventListener('input', () => {
        updateTokenCount();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(triggerPrediction, 40); // 40ms debounce
    });

    // Keyboard Hotkeys: Tab or ArrowRight to accept ghost suggestion
    inputText.addEventListener('keydown', (e) => {
        if ((e.key === 'Tab' || e.key === 'ArrowRight') && currentTopWord) {
            if (inputText.selectionStart === inputText.value.length) {
                e.preventDefault();
                acceptPrediction(currentTopWord);
            }
        }
    });

    // Preset Prompt Buttons
    document.querySelectorAll('.prompt-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const prompt = pill.getAttribute('data-prompt');
            inputText.value = prompt;
            inputText.focus();
            inputText.setSelectionRange(prompt.length, prompt.length);
            updateTokenCount();
            triggerPrediction();
        });
    });

    // Slider Listeners
    tempSlider.addEventListener('input', (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(2);
        triggerPrediction();
    });

    topkSlider.addEventListener('input', (e) => {
        topkVal.textContent = e.target.value;
        triggerPrediction();
    });

    repSlider.addEventListener('input', (e) => {
        repVal.textContent = parseFloat(e.target.value).toFixed(2);
        triggerPrediction();
    });

    toppSlider.addEventListener('input', (e) => {
        toppVal.textContent = parseFloat(e.target.value).toFixed(2);
        triggerPrediction();
    });

    genWordsSlider.addEventListener('input', (e) => {
        genWordsVal.textContent = e.target.value;
    });

    // Model Selector Change
    modelSelector.addEventListener('change', async (e) => {
        const modelId = e.target.value;
        await switchModel(modelId);
        triggerPrediction();
    });

    // Generate Sequence Button (Streaming or Async)
    btnGenerate.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) return;

        btnGenerate.disabled = true;
        btnGenerate.innerHTML = `<span>⚡</span> Synthesizing...`;

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    num_words: parseInt(genWordsSlider.value),
                    temperature: parseFloat(tempSlider.value),
                    top_p: parseFloat(toppSlider.value),
                    repetition_penalty: parseFloat(repSlider.value),
                })
            });
            const data = await response.json();
            if (data.status === 'success') {
                displayGenerationResult(data.data);
            }
        } catch (err) {
            console.error('Generation failed:', err);
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = `<span>⚡</span> Complete Sentence`;
        }
    });

    // Clear Button
    btnClear.addEventListener('click', () => {
        inputText.value = '';
        ghostPredicted.textContent = '...';
        currentTopWord = '';
        candidatesContainer.innerHTML = '';
        genResultBox.classList.remove('active');
        spellcheckBanner.classList.remove('active');
        spellcheckItems.innerHTML = '';
        updateTokenCount();
        latencyVal.textContent = '0.0 ms';
    });

    // Functions
    function updateTokenCount() {
        const words = inputText.value.trim().split(/\s+/).filter(Boolean);
        tokenCount.textContent = words.length;
    }

    async function triggerPrediction() {
        const text = inputText.value;
        if (!text.trim()) {
            ghostPredicted.textContent = '...';
            currentTopWord = '';
            candidatesContainer.innerHTML = '';
            spellcheckBanner.classList.remove('active');
            latencyVal.textContent = '0.0 ms';
            return;
        }

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    top_k: parseInt(topkSlider.value),
                    temperature: parseFloat(tempSlider.value),
                    top_p: parseFloat(toppSlider.value),
                    repetition_penalty: parseFloat(repSlider.value),
                })
            });
            const json = await res.json();
            if (json.status === 'success') {
                renderPrediction(json.data);
                renderSpellcheck(json.data.misspelled_words || []);
            }
        } catch (err) {
            console.error('Prediction request error:', err);
        }
    }

    function renderSpellcheck(misspelled) {
        if (!misspelled || misspelled.length === 0) {
            spellcheckBanner.classList.remove('active');
            spellcheckItems.innerHTML = '';
            return;
        }

        spellcheckBanner.classList.add('active');
        spellcheckItems.innerHTML = '';

        misspelled.forEach(item => {
            const span = document.createElement('span');
            span.style.display = 'inline-flex';
            span.style.alignItems = 'center';
            span.style.gap = '6px';
            span.innerHTML = `
                <span class="wiggly-underline">"${escapeHtml(item.word)}"</span>
                ${item.suggestion ? `<button type="button" class="spell-fix-btn" data-old="${escapeHtml(item.word)}" data-new="${escapeHtml(item.suggestion)}">Fix: ${escapeHtml(item.suggestion)}</button>` : ''}
            `;
            spellcheckItems.appendChild(span);
        });

        // Add click listeners to fix buttons
        document.querySelectorAll('.spell-fix-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const oldWord = e.target.getAttribute('data-old');
                const newWord = e.target.getAttribute('data-new');
                if (oldWord && newWord) {
                    const regex = new RegExp(`\\b${oldWord}\\b`, 'i');
                    inputText.value = inputText.value.replace(regex, newWord);
                    inputText.focus();
                    triggerPrediction();
                }
            });
        });
    }

    function renderPrediction(data) {
        // Latency
        latencyVal.textContent = `${data.latency_ms} ms ${data.cached ? '⚡ cached' : ''}`;

        // Top word ghost
        currentTopWord = data.top_word || '';
        ghostPredicted.textContent = currentTopWord || '...';

        // Render Top-K candidate cards
        candidatesContainer.innerHTML = '';
        const candidates = data.candidates || [];

        if (candidates.length === 0) {
            candidatesContainer.innerHTML = `
                <div style="grid-column: 1/-1; color: var(--text-dim); font-size: 13px; text-align: center; padding: 12px;">
                    No candidates found for current context.
                </div>
            `;
            return;
        }

        candidates.forEach((cand, idx) => {
            const card = document.createElement('div');
            card.className = 'candidate-card';
            card.innerHTML = `
                <div class="candidate-top">
                    <div>
                        <span class="candidate-rank">#${idx + 1}</span>
                        <span class="candidate-word">${escapeHtml(cand.word)}</span>
                    </div>
                    <span class="candidate-pct">${cand.confidence_pct}%</span>
                </div>
                <div class="probability-track">
                    <div class="probability-fill" style="width: ${Math.max(3, cand.confidence_pct)}%;"></div>
                </div>
            `;
            card.addEventListener('click', () => {
                acceptPrediction(cand.word);
            });
            candidatesContainer.appendChild(card);
        });
    }

    function acceptPrediction(word) {
        if (!word) return;
        const current = inputText.value;
        const needsSpace = current.length > 0 && !current.endsWith(' ');
        inputText.value = (needsSpace ? current + ' ' : current) + word + ' ';
        inputText.focus();
        inputText.setSelectionRange(inputText.value.length, inputText.value.length);
        updateTokenCount();
        triggerPrediction();
    }

    function displayGenerationResult(data) {
        genResultBox.classList.add('active');
        const words = data.generated_words || [];
        const wordsSpan = words.map(w => `<span class="gen-word-highlight">${escapeHtml(w)}</span>`).join(' ');
        genResultContent.innerHTML = `
            <div><strong>Seed:</strong> ${escapeHtml(data.seed_text)}</div>
            <div style="margin-top: 6px;"><strong>Synthesized:</strong> ${escapeHtml(data.seed_text)} ${wordsSpan}</div>
            <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim);">
                Inference Latency: ${data.latency_ms} ms (${words.length} tokens) &bull; Top-p: ${toppSlider.value} &bull; Rep-Penalty: ${repSlider.value}
            </div>
        `;
    }

    async function fetchModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.status === 'success') {
                availableModels = data.models;
                modelSelector.innerHTML = '';
                Object.values(availableModels).forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = `${m.name} [${m.arch.toUpperCase()}]`;
                    if (m.id === data.active_model) {
                        opt.selected = true;
                    }
                    modelSelector.appendChild(opt);
                });
                updateModelSpecs(data.active_model);
            }
        } catch (err) {
            console.error('Failed to fetch models:', err);
        }
    }

    async function switchModel(modelId) {
        try {
            const res = await fetch('/api/model/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            });
            const data = await res.json();
            if (data.status === 'success') {
                updateModelSpecs(modelId);
            }
        } catch (err) {
            console.error('Failed to switch model:', err);
        }
    }

    function updateModelSpecs(modelId) {
        const m = availableModels[modelId];
        if (!m) return;
        activeModelTag.textContent = m.name;
        specArch.textContent = m.arch.toUpperCase();
        specSeqLen.textContent = `${m.input_length} tokens`;
        specVocab.textContent = `${m.vocab_size.toLocaleString()} words`;
    }

    function escapeHtml(str) {
        return str.replace(/[&<>'"]/g, tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag));
    }

    // Subtle Neural Particle Network Animation
    function initNeuralCanvas() {
        const canvas = document.getElementById('neuralCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        const particles = [];
        const numParticles = Math.min(45, Math.floor(width / 35));

        for (let i = 0; i < numParticles; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                radius: Math.random() * 1.8 + 0.8,
            });
        }

        function render() {
            ctx.clearRect(0, 0, width, height);

            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 140) {
                        const alpha = (1 - dist / 140) * 0.15;
                        ctx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
                        ctx.lineWidth = 0.8;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            for (let p of particles) {
                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0) p.x = width;
                if (p.x > width) p.x = 0;
                if (p.y < 0) p.y = height;
                if (p.y > height) p.y = 0;

                ctx.fillStyle = 'rgba(0, 240, 255, 0.4)';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
            }

            requestAnimationFrame(render);
        }

        render();
    }
});
