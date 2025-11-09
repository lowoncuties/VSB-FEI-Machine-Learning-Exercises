const ql = (id) => document.getElementById(id);
let seedL = 42;

function updateLabelsL() {
  ql('samplesValue').textContent = ql('samples').value;
  ql('noiseValue').textContent = parseFloat(ql('noise').value).toFixed(2);
  ql('degreeValue').textContent = ql('degree').value;
}

function paramsL() {
  return {
    dataset: ql('dataset').value,
    n_samples: parseInt(ql('samples').value, 10),
    noise: parseFloat(ql('noise').value),
    random_state: seedL,
    degree: parseInt(ql('degree').value, 10)
  };
}

async function updateLinear() {
  const p = paramsL();
  ql('update').disabled = true;
  try {
    const res = await fetch('/api/linear', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p) });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    ql('r2').textContent = data.r2.toFixed(3);
    // equation and metrics
    if (data.equation) {
      ql('eq').textContent = data.equation;
      if (data.scaler && data.scaler.mean && data.scaler.scale) {
        const m = Array.isArray(data.scaler.mean) ? data.scaler.mean.map(v => v.toFixed(3)).join(', ') : data.scaler.mean;
        const s = Array.isArray(data.scaler.scale) ? data.scaler.scale.map(v => v.toFixed(3)).join(', ') : data.scaler.scale;
        ql('scalerInfo').textContent = `Standardization: x' = (x - mean) / scale; mean = [${m}], scale = [${s}]`;
      }
    } else {
      ql('eq').textContent = '';
      ql('scalerInfo').textContent = '';
    }
    if (data.metrics) {
      ql('metrics').textContent = `MAE=${data.metrics.mae.toFixed(4)}, MSE=${data.metrics.mse.toFixed(4)}, RMSE=${data.metrics.rmse.toFixed(4)}, MAPE=${(data.metrics.mape*100).toFixed(2)}%, R²=${data.metrics.r2.toFixed(3)}`;
    }
    if (data.is_1d) {
      ql('img1d').src = 'data:image/png;base64,' + data.image_1d;
      ql('img1d').style.display = 'block';
      ql('plot3d-lin').style.display = 'none';
    } else {
      ql('img1d').style.display = 'none';
      ql('plot3d-lin').style.display = 'block';
      const surface = { type:'surface', x:data.x, y:data.y, z:data.z, colorscale:'Viridis', showscale:true, opacity:0.95 };
      const pts = { type:'scatter3d', mode:'markers', x:data.pointsX, y:data.pointsY, z:data.pointsZ, marker:{ size:3, color:'#111827' }, name:'Data' };
      const layout = { margin:{l:0,r:0,b:0,t:0}, scene:{ xaxis:{title:'x1'}, yaxis:{title:'x2'}, zaxis:{title:'y'} } };
      Plotly.newPlot('plot3d-lin', [surface, pts], layout, { responsive:true, displaylogo:false });
    }
  } catch (e) {
    console.error(e); alert('Error: ' + e.message);
  } finally {
    ql('update').disabled = false;
  }
}

function wireLinear() {
  updateLabelsL();
  ['dataset','samples','noise','degree'].forEach(id => {
    ql(id).addEventListener('input', updateLabelsL);
    ql(id).addEventListener('change', updateLinear);
  });
  ql('update').addEventListener('click', updateLinear);
  ql('randomize').addEventListener('click', () => { seedL = Math.floor(Math.random()*100000); updateLinear(); });
  updateLinear();
}

document.addEventListener('DOMContentLoaded', wireLinear);


