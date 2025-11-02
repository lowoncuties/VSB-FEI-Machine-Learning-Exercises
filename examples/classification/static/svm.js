const qs = (id) => document.getElementById(id);

let seed = 42;

function exp10(v) {
  return Math.pow(10, v);
}

function updateVisibility() {
  const kernel = qs('kernel').value;
  qs('gammaGroup').style.display = kernel === 'linear' ? 'none' : 'block';
  qs('degreeGroup').style.display = kernel === 'poly' ? 'block' : 'none';
}

function updateLabels() {
  qs('samplesValue').textContent = qs('samples').value;
  qs('noiseValue').textContent = parseFloat(qs('noise').value).toFixed(2);
  const cPow = parseFloat(qs('C').value);
  const cVal = exp10(cPow);
  qs('CValue').textContent = cVal.toFixed(2);
  const gPow = parseFloat(qs('gamma').value);
  const gVal = exp10(gPow);
  qs('gammaValue').textContent = gVal.toFixed(2);
  qs('degreeValue').textContent = qs('degree').value;
}

async function fetchSVM() {
  const params = {
    dataset: qs('dataset').value,
    n_samples: parseInt(qs('samples').value, 10),
    noise: parseFloat(qs('noise').value),
    random_state: seed,
    kernel: qs('kernel').value,
    C: exp10(parseFloat(qs('C').value)),
    gamma_mode: 'value',
    gamma: exp10(parseFloat(qs('gamma').value)),
    degree: parseInt(qs('degree').value, 10)
  };

  qs('update').disabled = true;
  try {
    const res = await fetch('/api/svm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    qs('img2d').src = 'data:image/png;base64,' + data.image_2d;
    qs('acc').textContent = (data.accuracy * 100).toFixed(1) + '%';
    qs('nsv').textContent = data.n_support_vectors ?? '—';
    qs('params').textContent = `${data.params.kernel}, C=${data.params.C}, gamma=${data.params.gamma}`;

    // interactive 3D surface
    await fetchSurface(params);
  } catch (e) {
    console.error(e);
    alert('Error: ' + e.message);
  } finally {
    qs('update').disabled = false;
  }
}

async function fetchSurface(params) {
  const res = await fetch('/api/svm-surface', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  const surf = await res.json();
  if (surf.error) throw new Error(surf.error);

  const colors = ['#1f77b4', '#d62728'];
  const surface = {
    type: 'surface',
    x: surf.x,
    y: surf.y,
    z: surf.z,
    colorscale: 'Viridis',
    opacity: 0.9,
    showscale: true,
    contours: { z: { show: true, start: 0, end: 0, size: 0.01, color: 'black' } }
  };

  const pts = {
    type: 'scatter3d',
    mode: 'markers',
    x: surf.pointsX,
    y: surf.pointsY,
    z: surf.pointsZ,
    marker: { size: 3, color: surf.classes.map(c => colors[c]), line: { color: 'black', width: 0.5 } },
    name: 'Training points'
  };

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: {
      xaxis: { title: 'x1' },
      yaxis: { title: 'x2' },
      zaxis: { title: 'decision_function(x)' },
    }
  };

  Plotly.newPlot('plot3d', [surface, pts], layout, { responsive: true, displaylogo: false });
}

function wire() {
  updateVisibility();
  updateLabels();

  ['dataset','samples','noise','kernel','C','gamma','degree'].forEach(id => {
    qs(id).addEventListener('input', () => { updateLabels(); updateVisibility(); });
    qs(id).addEventListener('change', fetchSVM);
  });

  qs('update').addEventListener('click', fetchSVM);
  qs('randomize').addEventListener('click', () => { seed = Math.floor(Math.random() * 100000); fetchSVM(); });

  // initial load
  fetchSVM();
}

document.addEventListener('DOMContentLoaded', wire);


