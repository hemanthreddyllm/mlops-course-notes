// Shared behaviour: hover tooltips, theme toggle, simple line-chart crosshair.
(function () {
  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.hidden = true;
  document.body.appendChild(tip);

  function place(x, y) {
    const pad = 14;
    const r = tip.getBoundingClientRect();
    let left = x + pad, top = y + pad;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function show(html, x, y) { tip.innerHTML = html; tip.hidden = false; place(x, y); }
  function hide() { tip.hidden = true; }
  window.NotesTip = { show, hide };

  // Any element with data-tip gets a tooltip on hover / focus.
  document.addEventListener('pointerover', e => {
    const el = e.target.closest('[data-tip]');
    if (el) show(el.getAttribute('data-tip'), e.clientX, e.clientY);
  });
  document.addEventListener('pointermove', e => {
    if (!tip.hidden && e.target.closest('[data-tip]')) place(e.clientX, e.clientY);
  });
  document.addEventListener('pointerout', e => {
    if (e.target.closest('[data-tip]')) hide();
  });
  document.addEventListener('focusin', e => {
    const el = e.target.closest('[data-tip]');
    if (el) { const r = el.getBoundingClientRect(); show(el.getAttribute('data-tip'), r.right, r.top); }
  });
  document.addEventListener('focusout', hide);

  // Theme toggle (remembered per browser when storage is available)
  const btn = document.createElement('button');
  btn.className = 'theme-btn';
  btn.type = 'button';
  const root = document.documentElement;
  function label() {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    btn.textContent = dark ? '☀ Light' : '☾ Dark';
  }
  try { const t = localStorage.getItem('mlops-notes-theme'); if (t) root.dataset.theme = t; } catch (_) {}
  btn.addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('mlops-notes-theme', root.dataset.theme); } catch (_) {}
    label();
  });
  label();
  document.body.appendChild(btn);

  // Line chart with crosshair: <svg data-linechart='{"x":[..],"series":[{"name","color","values"}],"xlabel","ylabel","ymax","marks":[{i,label}]}'>
  document.querySelectorAll('svg[data-linechart]').forEach(svg => {
    const cfg = JSON.parse(svg.getAttribute('data-linechart'));
    const W = 760, H = 320, m = { l: 52, r: 20, t: 16, b: 44 };
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const NS = 'http://www.w3.org/2000/svg';
    const el = (n, a, p = svg) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
    const n = cfg.x.length;
    const ymax = cfg.ymax;
    const X = i => m.l + (i / (n - 1)) * (W - m.l - m.r);
    const Y = v => H - m.b - (v / ymax) * (H - m.t - m.b);
    // grid + y ticks
    const ticks = 5;
    for (let k = 0; k <= ticks; k++) {
      const v = ymax * k / ticks, y = Y(v);
      el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: k === 0 ? 'axis' : 'grid-line' });
      el('text', { x: m.l - 8, y: y + 4, 'text-anchor': 'end', class: 't-xs' }).textContent = Math.round(v);
    }
    // x labels (every step)
    const every = cfg.xEvery || 1;
    cfg.x.forEach((lab, i) => {
      if (i % every) return;
      el('text', { x: X(i), y: H - m.b + 18, 'text-anchor': 'middle', class: 't-xs' }).textContent = lab;
    });
    if (cfg.xlabel) el('text', { x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle', class: 't-xs' }).textContent = cfg.xlabel;
    if (cfg.ylabel) el('text', { x: 14, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', class: 't-xs', transform: `rotate(-90 14 ${(H - m.b + m.t) / 2})` }).textContent = cfg.ylabel;
    // annotation bands
    (cfg.bands || []).forEach(b => {
      el('rect', { x: X(b.from), y: m.t, width: X(b.to) - X(b.from), height: H - m.t - m.b, fill: 'var(--surface-2)' });
      el('text', { x: (X(b.from) + X(b.to)) / 2, y: m.t + 14, 'text-anchor': 'middle', class: 't-xs' }).textContent = b.label;
    });
    // series
    cfg.series.forEach(s => {
      const d = s.values.map((v, i) => v == null ? null : `${X(i)},${Y(v)}`).filter(Boolean);
      el('polyline', { points: d.join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-dasharray': s.dash || '' });
    });
    // direct marks
    (cfg.marks || []).forEach(mk => {
      const s = cfg.series[mk.s || 0];
      const v = s.values[mk.i];
      el('circle', { cx: X(mk.i), cy: Y(v), r: 4.5, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2 });
      el('text', { x: X(mk.i) + (mk.dx || 0), y: Y(v) + (mk.dy || -12), 'text-anchor': mk.anchor || 'middle', class: 't-xs' }).textContent = mk.label;
    });
    // crosshair
    const cross = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '3 3', visibility: 'hidden' });
    const dots = cfg.series.map(s => el('circle', { r: 4.5, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' }));
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' });
    hit.addEventListener('pointermove', e => {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const i = Math.max(0, Math.min(n - 1, Math.round((p.x - m.l) / (W - m.l - m.r) * (n - 1))));
      cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('visibility', 'visible');
      let html = `<b>${cfg.x[i]}</b>`;
      cfg.series.forEach((s, k) => {
        const v = s.values[i];
        if (v == null) { dots[k].setAttribute('visibility', 'hidden'); return; }
        dots[k].setAttribute('cx', X(i)); dots[k].setAttribute('cy', Y(v)); dots[k].setAttribute('visibility', 'visible');
        html += `<br>${s.name}: ${v}${cfg.unit || ''}`;
      });
      if (cfg.notes && cfg.notes[i]) html += `<br><i>${cfg.notes[i]}</i>`;
      show(html, e.clientX, e.clientY);
    });
    hit.addEventListener('pointerleave', () => {
      cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); hide();
    });
  });
})();
