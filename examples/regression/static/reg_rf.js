const qr = (id) => document.getElementById(id);
let seedR = 42;

function updateLabelsR() {
  qr('samplesValue').textContent = qr('samples').value;
  qr('noiseValue').textContent = parseFloat(qr('noise').value).toFixed(2);
  qr('treesValue').textContent = qr('n_estimators').value;
  const md = parseInt(qr('max_depth').value, 10);
  qr('depthValue').textContent = md >= 20 ? 'None' : md.toString();
  qr('leafValue').textContent = qr('min_samples_leaf').value;
}

function paramsR() {
  const md = parseInt(qr('max_depth').value, 10);
  return {
    dataset: qr('dataset').value,
    n_samples: parseInt(qr('samples').value, 10),
    noise: parseFloat(qr('noise').value),
    random_state: seedR,
    n_estimators: parseInt(qr('n_estimators').value, 10),
    max_depth: md >= 20 ? null : md,
    min_samples_leaf: parseInt(qr('min_samples_leaf').value, 10)
  };
}

async function updateRF() {
  const p = paramsR();
  qr('update').disabled = true;
  try {
    const res = await fetch('/api/rf', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p) });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    qr('r2').textContent = data.r2.toFixed(3);
    if (data.metrics) {
      qr('metrics').textContent = `MAE=${data.metrics.mae.toFixed(4)}, MSE=${data.metrics.mse.toFixed(4)}, RMSE=${data.metrics.rmse.toFixed(4)}, MAPE=${(data.metrics.mape*100).toFixed(2)}%, R²=${data.metrics.r2.toFixed(3)}`;
    }
    if (data.feature_importances) {
      const imp = data.feature_importances.map(v => (v*100).toFixed(1)+'%');
      const label = (qr('dataset').value === 'hills_2d') ? 'x1, x2' : 'x';
      qr('importances').textContent = `Feature importances (${label}): [${imp.join(', ')}]`;
    }
    if (data.is_1d) {
      qr('img1d').src = 'data:image/png;base64,' + data.image_1d;
      qr('img1d').style.display = 'block';
      qr('plot3d-rf').style.display = 'none';
    } else {
      qr('img1d').style.display = 'none';
      qr('plot3d-rf').style.display = 'block';
      const surface = { type:'surface', x:data.x, y:data.y, z:data.z, colorscale:'Viridis', showscale:true, opacity:0.95 };
      const pts = { type:'scatter3d', mode:'markers', x:data.pointsX, y:data.pointsY, z:data.pointsZ, marker:{ size:3, color:'#111827' }, name:'Data' };
      const layout = { margin:{l:0,r:0,b:0,t:0}, scene:{ xaxis:{title:'x1'}, yaxis:{title:'x2'}, zaxis:{title:'y'} } };
      Plotly.newPlot('plot3d-rf', [surface, pts], layout, { responsive:true, displaylogo:false });
    }
  } catch (e) {
    console.error(e); alert('Error: ' + e.message);
  } finally {
    qr('update').disabled = false;
  }
}

function wireRF() {
  updateLabelsR();
  ['dataset','samples','noise','n_estimators','max_depth','min_samples_leaf'].forEach(id => {
    qr(id).addEventListener('input', updateLabelsR);
    qr(id).addEventListener('change', updateRF);
  });
  qr('update').addEventListener('click', updateRF);
  qr('randomize').addEventListener('click', () => { seedR = Math.floor(Math.random()*100000); updateRF(); });
  updateRF();
}

document.addEventListener('DOMContentLoaded', wireRF);


