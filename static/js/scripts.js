let currentChart = null;
let currentAnalysisId = null;

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('dataForm');
  const darkToggle = document.getElementById('darkToggle');
  const body = document.body;

  // Load dark mode from localStorage
  if (localStorage.getItem('theme') === 'dark') {
    body.classList.add('dark-mode');
  }

  darkToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    localStorage.setItem('theme', body.classList.contains('dark-mode') ? 'dark' : 'light');
  });

  document.getElementById('satisfactionSlider').addEventListener('input', (e) => {
    document.getElementById('satisfactionValue').textContent = e.target.value;
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const jdeData = document.getElementById('jdeData').value;
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const loadingStatus = document.getElementById('loadingStatus');

    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    updateAgentStatus('dataValidatorStatus', 'working');
    loadingStatus.textContent = 'Analyzing data structure...';

    try {
      const response = await fetch('/analyze-agentic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jde_data: JSON.parse(jdeData) })
      });

      const result = await response.json();
      currentAnalysisId = result.analysis_id;

      ['dataValidatorStatus', 'chartAnalyzerStatus', 'validationAgentStatus', 'mcpConnectorStatus']
        .forEach(id => updateAgentStatus(id, 'completed'));

      loadingDiv.style.display = 'none';
      displayAgenticResults(result);
      resultsDiv.style.display = 'block';
    } catch (err) {
      loadingDiv.style.display = 'none';
      alert(err.message);
    }
  });
});

function updateAgentStatus(id, status) {
  const el = document.getElementById(id);
  el.style.background = status === 'working' ? '#ffc107' :
                        status === 'completed' ? '#28a745' : '#dc3545';
}

function displayAgenticResults(result) {
  const analysisResults = document.getElementById('analysisResults');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceDetails = document.getElementById('confidenceDetails');

  analysisResults.innerHTML = `
    <p><strong>Chart Type:</strong> ${result.chart_type}</p>
    <p><strong>Validation:</strong> ${result.validation_status}</p>
    <p><strong>Data Quality:</strong> ${(result.data_quality_score * 100).toFixed(1)}%</p>
    <p><strong>Reasoning:</strong> ${result.reasoning}</p>
  `;

  confidenceFill.style.width = (result.confidence_score * 100).toFixed(1) + '%';
  confidenceDetails.innerHTML = `
    <p><strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%</p>
    <p><strong>Suggestions:</strong> ${result.suggestions.join(', ')}</p>
  `;

  createChart(result);
}

function createChart(config) {
  const ctx = document.getElementById('dynamicChart').getContext('2d');
  if (currentChart) currentChart.destroy();

  currentChart = new Chart(ctx, {
    type: config.chart_type,
    data: config.chart_data,
    options: { responsive: true, maintainAspectRatio: false, ...config.chart_config.options }
  });
}

function submitFeedback() {
  if (!currentAnalysisId) return alert('No analysis ID found.');
  const val = document.getElementById('satisfactionSlider').value;

  fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ analysis_id: currentAnalysisId, satisfaction: parseInt(val) })
  }).then(() => alert('✅ Feedback submitted!'))
    .catch(err => alert('Error: ' + err.message));
}
