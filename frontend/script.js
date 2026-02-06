/**
 * SFSD AI Frontend
 * Final Version (Hybrid RAG + Sources + Pseudocode blocks)
 */

// ============================================
// DOM Elements
// ============================================
const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const loadingIndicator = document.getElementById('loadingIndicator');
const answerContainer = document.getElementById('answerContainer');
const answerContent = document.getElementById('answerContent');

// ============================================
// Configuration
// ============================================
// إذا راح تنشر أونلاين، بدّل SERVER_BASE لـ رابط السيرفر تاعك
const CONFIG = {
  SERVER_BASE: 'https://sfsd-ai.onrender.com',
  TIMEOUT: 60000
};

// ============================================
// State
// ============================================
const appState = {
  isLoading: false,
  chatHistory: []
};

// ============================================
// Helpers
// ============================================
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

function validateQuestion(question) {
  if (!question) return false;
  if (question.trim().length < 3) return false;
  if (question.length > 2000) return false;
  return true;
}

function updateLoadingText(text) {
  const el = document.querySelector('.loading-text');
  if (el) el.textContent = text;
}

function setLoadingState(isLoading) {
  appState.isLoading = isLoading;
  askButton.disabled = isLoading;
  loadingIndicator.classList.toggle('active', isLoading);

  // نخبي الإجابة فقط كي نبدأ التحميل
  if (isLoading) {
    answerContainer.classList.remove('active');
  }
}

/**
 * Render answer while preserving Markdown code blocks:
 * ```pseudo
 * ...
 * ```
 */
function renderAnswerPreserveCode(answer) {
  const raw = answer || '';

  // Extract code blocks first
  const codeBlocks = [];
  const placeholder = (i) => `__CODE_BLOCK_${i}__`;

  let text = raw.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push({
      lang: (lang || 'text').trim(),
      code: code
    });
    return placeholder(idx);
  });

  // Escape rest HTML
  text = escapeHtml(text);

  // Keep line breaks (simple, safe formatting)
  text = text
    .split('\n')
    .map(line => line.replace(/\s+$/g, ''))
    .join('<br>');

  // Restore code blocks
  codeBlocks.forEach((b, i) => {
    const safeCode = escapeHtml(b.code);
    const blockHtml =
      `<pre style="margin-top:12px; padding:12px; border-radius:10px; overflow:auto;">
         <code class="lang-${escapeHtml(b.lang)}">${safeCode}</code>
       </pre>`;
    text = text.replaceAll(placeholder(i), blockHtml);
  });

  return text;
}

function displayAnswer(answer, historyCount, grounded = false, sources = []) {
  const hasSources = Array.isArray(sources) && sources.length > 0;

  const statusLine = grounded
    ? "✅ Réponse basée sur vos PDFs"
    : "⚠️ Réponse générale (pas trouvée dans vos PDFs)";

  const sourcesHtml = hasSources
    ? `<details style="margin-top:12px;">
         <summary>📚 Sources (${sources.length})</summary>
         <ul style="margin-top:10px; padding-left:18px;">
           ${sources.map(s => `
             <li style="margin-bottom:6px;">
               <strong>${escapeHtml(s.file)}</strong> — page ${escapeHtml(String(s.page))}
               ${typeof s.score === 'number' ? `(score: ${s.score.toFixed(3)})` : ''}
             </li>
           `).join('')}
         </ul>
       </details>`
    : '';

  answerContent.innerHTML = `
    <div class="answer-text">${renderAnswerPreserveCode(answer)}</div>
    <div class="sources-info">${statusLine} • Question #${historyCount}</div>
    ${sourcesHtml}
  `;

  answerContainer.classList.add('active');

  setTimeout(() => {
    answerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 80);
}

function showError(message) {
  answerContent.innerHTML = `
    <div style="color:#dc2626; text-align:center; padding:24px;">
      <h3 style="margin-bottom:10px;">Error</h3>
      <p style="white-space:pre-wrap; color:#64748b;">${escapeHtml(message)}</p>
    </div>
  `;
  answerContainer.classList.add('active');
  loadingIndicator.classList.remove('active');
}

// ============================================
// API
// ============================================
async function askQuestion(question) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), CONFIG.TIMEOUT);

  try {
    const res = await fetch(`${CONFIG.SERVER_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode: 'hybrid' }),
      signal: controller.signal
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    if (!data.answer) throw new Error('Invalid response format');
    return data;
  } finally {
    clearTimeout(t);
  }
}

async function checkHealth() {
  try {
    const res = await fetch(`${CONFIG.SERVER_BASE}/health`);
    const data = await res.json();
    console.log('Health:', data);
  } catch (e) {
    console.warn('Health check failed:', e);
  }
}

// ============================================
// Handlers
// ============================================
async function handleAskQuestion() {
  const question = questionInput.value.trim();

  if (!validateQuestion(question)) {
    showError('⚠️ اكتب سؤال صحيح (3 أحرف على الأقل).');
    questionInput.focus();
    return;
  }

  setLoadingState(true);
  updateLoadingText('🔄 Recherche dans vos PDFs...');

  try {
    updateLoadingText('🤖 Génération de la réponse...');
    const result = await askQuestion(question);

    appState.chatHistory.unshift({
      question,
      answer: result.answer,
      grounded: result.grounded,
      sources: result.sources || [],
      ts: new Date().toLocaleTimeString()
    });

    displayAnswer(
      result.answer,
      result.history_count ?? appState.chatHistory.length,
      result.grounded ?? false,
      result.sources ?? []
    );

    questionInput.value = '';
    questionInput.style.height = 'auto';
  } catch (e) {
    console.error(e);
    showError(`❌ ${e.message}\n\n- تأكد أن Flask خدام\n- SERVER_BASE صحيح\n- CORS مفعل`);
  } finally {
    setLoadingState(false);
  }
}

// ============================================
// Listeners
// ============================================
askButton.addEventListener('click', handleAskQuestion);

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleAskQuestion();
  }
});

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = questionInput.scrollHeight + 'px';
});

document.addEventListener('DOMContentLoaded', async () => {
  console.log('✅ SFSD AI Front loaded');
  questionInput.focus();
  await checkHealth();
});
