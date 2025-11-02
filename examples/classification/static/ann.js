const q = (id) => document.getElementById(id);
let seedANN = 42;
let lastAGrids = null; // activation grids
let lastZGrids = null; // pre-activation grids
let lastActivationsMeta = { x: null, y: null, num: 0, activation: 'relu' };
let colorMode = 'z';
let lastPoint = null; // {x,y,decision,neurons:[]}

function updateLabelsANN() {
  q('samplesValue').textContent = q('samples').value;
  q('noiseValue').textContent = parseFloat(q('noise').value).toFixed(2);
  q('iterValue').textContent = q('max_iter').value;
}

function parseHidden() {
  const text = q('hidden').value.trim();
  if (!text) return '5';
  return text;
}

function buildParamsANN() {
  return {
    dataset: q('dataset').value,
    n_samples: parseInt(q('samples').value, 10),
    noise: parseFloat(q('noise').value),
    random_state: seedANN,
    hidden_layer_sizes: parseHidden(),
    activation: q('activation').value,
    solver: q('solver').value,
    max_iter: parseInt(q('max_iter').value, 10)
  };
}

async function updateANN() {
  const params = buildParamsANN();
  q('update').disabled = true;
  try {
    const res = await fetch('/api/ann', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    q('img2d').src = 'data:image/png;base64,' + data.image_2d;
    q('acc').textContent = (data.accuracy * 100).toFixed(1) + '%';
    q('loss').textContent = isFinite(data.loss) ? data.loss.toFixed(4) : '—';
    q('iters').textContent = data.n_iter ?? '—';

    // 3D surface
    await updateSurfaceANN(params);
    // Activations
    await updateActivations(params);
  } catch (e) {
    console.error(e);
    alert('Error: ' + e.message);
  } finally {
    q('update').disabled = false;
  }
}

async function updateSurfaceANN(params) {
  const res = await fetch('/api/ann-surface', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params)
  });
  const surf = await res.json();
  if (surf.error) throw new Error(surf.error);
  const colors = ['#1f77b4', '#d62728'];
  const surface = { type: 'surface', x: surf.x, y: surf.y, z: surf.z, colorscale: 'Viridis', opacity: 0.9, showscale: true,
    contours: { z: { show: true, start: 0, end: 0, size: 0.01, color: 'black' } } };
  const pts = { type: 'scatter3d', mode: 'markers', x: surf.pointsX, y: surf.pointsY, z: surf.pointsZ,
    marker: { size: 3, color: surf.classes.map(c => colors[c]), line: { color: 'black', width: 0.5 } }, name: 'Training points' };
  const layout = { margin: { l:0, r:0, b:0, t:0 }, scene: { xaxis:{title:'x1'}, yaxis:{title:'x2'}, zaxis:{title:'decision_function(x)'} } };
  Plotly.newPlot('plot3d-ann', [surface, pts], layout, { responsive: true, displaylogo: false });
}

async function updateActivations(params) {
  const res = await fetch('/api/ann-activations', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params)
  });
  const act = await res.json();
  if (act.error) throw new Error(act.error);
  lastAGrids = act.activations;
  lastZGrids = act.pre_activations;
  lastActivationsMeta = { x: act.x, y: act.y, num: act.num_neurons, activation: act.activation };
  const container = q('activations');
  container.innerHTML = '';
  for (let i = 0; i < act.num_neurons; i++) {
    const holder = document.createElement('div');
    holder.className = 'act-card';
    const div = document.createElement('div');
    div.id = 'act_' + i;
    div.style.width = '100%';
    div.style.height = '160px';
    holder.appendChild(div);
    const cap = document.createElement('div');
    cap.className = 'small';
    cap.textContent = 'Neuron ' + (i+1);
    holder.appendChild(cap);
    container.appendChild(holder);

    const trace = heatmapForIndex(i);
    Plotly.newPlot(div.id, [trace],
      { margin:{l:20,r:10,t:10,b:20}, xaxis:{title:'x1', ticks:''}, yaxis:{title:'x2', ticks:''} },
      { displayModeBar: false, responsive: true });
  }

  // If we already have a point, overlay it now
  if (lastPoint) overlayPointOnHeatmaps(lastPoint);
}

function wireANN() {
  updateLabelsANN();
  ['dataset','samples','noise','hidden','activation','solver','max_iter'].forEach(id => {
    q(id).addEventListener('input', updateLabelsANN);
    q(id).addEventListener('change', updateANN);
  });
  q('update').addEventListener('click', updateANN);
  q('randomize').addEventListener('click', () => { seedANN = Math.floor(Math.random()*100000); updateANN(); });
  updateANN();
}

document.addEventListener('DOMContentLoaded', wireANN);

async function evaluatePoint() {
  const params = buildParamsANN();
  const px = parseFloat(q('px').value);
  const py = parseFloat(q('py').value);
  if (Number.isNaN(px) || Number.isNaN(py)) {
    alert('Please enter numeric x and y.');
    return;
  }
  const body = Object.assign({}, params, { point: [px, py] });
  try {
    const res = await fetch('/api/ann-point', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    const info = await res.json();
    if (info.error) throw new Error(info.error);
    lastPoint = info;

    // Update 3D plot with point marker
    const pointTrace = { type:'scatter3d', mode:'markers', x:[info.point.x], y:[info.point.y], z:[info.decision],
      marker:{ size:6, color:'#ef4444', symbol:'diamond' }, name:'Evaluated point' };
    const gd = document.getElementById('plot3d-ann');
    const data = gd.data ? gd.data.slice() : [];
    // Remove previous evaluated point trace if present (named 'Evaluated point')
    let idx = data.findIndex(t => t.name === 'Evaluated point');
    if (idx !== -1) { data.splice(idx, 1); }
    data.push(pointTrace);
    Plotly.react('plot3d-ann', data, gd.layout, { responsive:true, displaylogo:false });

    // Overlay on heatmaps
    overlayPointOnHeatmaps(info);

    // Show explainer with equations
    const box = q('point-explainer');
    const xs = info.point.x_scaled.toFixed(3);
    const ys = info.point.y_scaled.toFixed(3);
    let html = `<div><b>Point</b>: (x=${info.point.x.toFixed(3)}, y=${info.point.y.toFixed(3)}) → standardized (x'=${xs}, y'=${ys})</div>`;
    html += `<div style="margin-top:6px;"><b>First layer</b> (activation: ${info.activation})</div>`;
    html += '<ol style="margin:6px 0 0 18px; padding:0;">';
    info.neurons.forEach((n, i) => {
      const z = n.z.toFixed(3); const a = n.a.toFixed(3);
      const w1 = n.w1.toFixed(3); const w2 = n.w2.toFixed(3); const b = n.b.toFixed(3);
      html += `<li>z${i+1} = ${w1}·x' + ${w2}·y' + ${b} = <b>${z}</b>; a${i+1} = f(z${i+1}) = <b>${a}</b></li>`;
    });
    html += '</ol>';
    box.innerHTML = html;
    box.style.display = 'block';
  } catch (e) {
    console.error(e); alert('Error: '+e.message);
  }
}

function overlayPointOnHeatmaps(info) {
  if ((!lastAGrids && !lastZGrids) || !lastActivationsMeta || lastActivationsMeta.num === 0) return;
  for (let i = 0; i < lastActivationsMeta.num; i++) {
    const divId = 'act_' + i;
    const gd = document.getElementById(divId);
    if (!gd) continue;
    const data = gd.data ? gd.data.slice(0,1) : []; // keep heatmap only
    const aVal = info.neurons[i] ? info.neurons[i].a : null;
    const zVal = info.neurons[i] ? info.neurons[i].z : null;
    let on = false;
    if (colorMode === 'z') on = (zVal !== null && zVal > 0);
    else on = (aVal !== null && aVal > 0);
    const color = on ? '#ef4444' : '#3b82f6';
    const trace = { type:'scatter', mode:'markers', x:[info.point.x], y:[info.point.y], marker:{ size:8, color: color, line:{ color:'#111', width:1 } }, name:'Point' };
    data.push(trace);
    Plotly.react(divId, data, gd.layout, { displayModeBar:false, responsive:true });
  }
}

function heatmapForIndex(i) {
  const x = lastActivationsMeta.x; const y = lastActivationsMeta.y;
  if (colorMode === 'z') {
    return { type:'heatmap', x, y, z: lastZGrids[i], colorscale:'RdBu', zmid:0 };
  }
  // activation mode: choose sensible mid
  const act = lastActivationsMeta.activation;
  if (act === 'relu') {
    return { type:'heatmap', x, y, z: lastAGrids[i], colorscale:'Blues', zmin:0 };
  }
  if (act === 'logistic') {
    return { type:'heatmap', x, y, z: lastAGrids[i], colorscale:'RdBu', zmid:0.5 };
  }
  // tanh or identity defaults
  return { type:'heatmap', x, y, z: lastAGrids[i], colorscale:'RdBu', zmid:0 };
}

// wire evaluate button
document.addEventListener('DOMContentLoaded', () => {
  const btn = q('evalPoint');
  if (btn) btn.addEventListener('click', evaluatePoint);
  const cm = q('colorMode');
  if (cm) cm.addEventListener('change', () => {
    colorMode = cm.value;
    // re-render heatmaps
    if (!lastActivationsMeta || lastActivationsMeta.num === 0) return;
    for (let i = 0; i < lastActivationsMeta.num; i++) {
      const divId = 'act_' + i;
      const gd = document.getElementById(divId);
      if (!gd) continue;
      const trace = heatmapForIndex(i);
      Plotly.react(divId, [trace], gd.layout, { displayModeBar:false, responsive:true });
    }
    if (lastPoint) overlayPointOnHeatmaps(lastPoint);
  });
});


