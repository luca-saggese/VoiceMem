import { useEffect, useRef } from 'react';

const IMG_AR = 1672 / 941;
const BOX = { x0: .245, x1: .755, y0: .11, y1: .855 };
const HEMI = { L: { cx: .378, cy: .485, rx: .090, ry: .295 }, R: { cx: .622, cy: .485, rx: .090, ry: .295 } };
const LEFT_SLOTS = ['work', 'health', 'relationships', 'finance', 'goals', 'daily_life', 'knowledge'];
const RIGHT_SLOTS = ['emotion', 'personality', 'preference'];
const COLORS = { entity: '#6ba7f0', emotion: '#ef7bab', exper: '#f0a35f', prefer: '#79d7a2', user: '#d766ad' };
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function rgba(hex, alpha) {
  const value = Number.parseInt(hex.slice(1), 16);
  return `rgba(${value >> 16},${value >> 8 & 255},${value & 255},${alpha})`;
}

function makeNode(node) {
  return { ...node, r: node.kind === 'user' ? 8 : node.kind === 'emotion' ? 5 : node.kind === 'exper' ? 4.6 : node.kind === 'prefer' ? 4.8 : 3, ph: Math.random() * 6.28, ph2: Math.random() * 6.28, sp: .28 + Math.random() * .4, ax: .5 + Math.random() * .9, ay: .5 + Math.random() * .9, iv: 0, h: 0, p: 0, hit: 0, hitT: 0, act: 0, bw: 0, bh: 0, dl: node.dl ?? ((performance.now() / 1000) + Math.random() * .3), sx: 0, sy: 0, x: 0, y: 0 };
}

export function BrainGraph({ memories = { left: [], right: [] } }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const nodesRef = useRef([]);
  const edgesRef = useRef([]);
  const selectedRef = useRef(-1);
  const pointerRef = useRef({ x: -9999, y: -9999, active: false });
  const cardRef = useRef(null);
  const cardPartsRef = useRef({ kd: null, id: null, sw: null, src: null, body: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = canvas.parentElement;
    const context = canvas.getContext('2d');
    const image = imageRef.current;
    const card = cardRef.current;
    const cardParts = card ? { kd: card.querySelector('.memory-card-kind'), id: card.querySelector('.memory-card-id'), sw: card.querySelector('.memory-card-swatch'), src: card.querySelector('.memory-card-source'), body: card.querySelector('.memory-card-content') } : {};
    if (card) card.style.opacity = '0';
    const nodes = [];
    const edges = [];
    const left = Array.isArray(memories.left) ? memories.left : [];
    const right = Array.isArray(memories.right) ? memories.right : [];
    const byCluster = new Map();
    const byEntity = new Map();
    const labelIds = new Map();
    const glow = {};
    const pulse = { t: -.6, dur: .9, trail: .3, seg: 26 };
    const hit = { pop: .42, hold: 1, fade: .45 };
    let mouseX = -9999; let mouseY = -9999; let hasPointer = false; let selected = -1; let previousSelected = -1;
    let cardX = 0; let cardY = 0; let cardTargetX = 0; let cardTargetY = 0; let cardSide = 1; let cardOn = false;
    let W = 0; let H = 0;
    const add = (node) => { const item = makeNode({ id: nodes.length, ...node }); nodes.push(item); return item.id; };
    LEFT_SLOTS.forEach((slot, seat) => labelIds.set(slot, add({ kind: 'label', side: 'L', cluster: slot, seat, w: slot })));
    RIGHT_SLOTS.forEach((slot, seat) => labelIds.set(slot, add({ kind: 'label', side: 'R', cluster: slot, seat, w: slot })));
    const userId = add({ kind: 'user', side: 'C', cluster: null, w: 'you' });
    const link = (a, b, type = 'in', weight = .4) => { if (a === b || !nodes[a] || !nodes[b] || edges.some((edge) => (edge.a === a && edge.b === b) || (edge.a === b && edge.b === a))) return; edges.push({ a, b, type, weight }); };
    [...LEFT_SLOTS, ...RIGHT_SLOTS].forEach((slot) => link(userId, labelIds.get(slot), 'cross', .5));
    left.forEach((memory) => {
      const slot = LEFT_SLOTS.includes(memory.slot) ? memory.slot : 'daily_life';
      const id = add({ kind: 'entity', side: 'L', cluster: slot, w: String(memory.text || '').slice(0, 40), detail: memory });
      link(labelIds.get(slot), id, 'in', .34);
      if (!byCluster.has(slot)) byCluster.set(slot, []); byCluster.get(slot).push(id);
      const entities = Array.isArray(memory.ents) ? memory.ents : Array.isArray(memory.entities) ? memory.entities : [];
      entities.forEach((entity) => {
        const name = String(entity).trim().toLowerCase();
        if (!name) return;
        if (!byEntity.has(name)) byEntity.set(name, []);
        byEntity.get(name).push(id);
      });
    });
    right.forEach((memory) => {
      const cluster = memory.cluster === 'emotion' ? 'emotion' : memory.cluster === 'preference' ? 'preference' : 'personality';
      const kind = cluster === 'emotion' ? 'emotion' : cluster === 'preference' ? 'prefer' : 'exper';
      const id = add({ kind, side: 'R', cluster, w: String(memory.text || '').slice(0, 40), detail: memory });
      link(labelIds.get(cluster), id, 'in', .34);
      if (!byCluster.has(cluster)) byCluster.set(cluster, []); byCluster.get(cluster).push(id);
    });
    byEntity.forEach((ids) => {
      const unique = [...new Set(ids)];
      for (let first = 0; first < unique.length; first += 1) {
        for (let second = first + 1; second < unique.length; second += 1) link(unique[first], unique[second], 'rel', .28);
      }
    });
    nodesRef.current = nodes; edgesRef.current = edges;

    let frame;
    let scale = 1;
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      W = rect.width; H = rect.height;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const boxWidth = BOX.x1 - BOX.x0; const boxHeight = BOX.y1 - BOX.y0;
      const boxAr = boxWidth * IMG_AR / boxHeight;
      let brainHeight = rect.height * .94; let brainWidth = brainHeight * boxAr;
      if (brainWidth > rect.width * .94) { brainWidth = rect.width * .94; brainHeight = brainWidth / boxAr; }
      scale = Math.max(.62, brainHeight / 620);
      const iw = brainWidth / boxWidth; const ih = brainHeight / boxHeight;
      const ox = (rect.width - brainWidth) / 2 - BOX.x0 * iw; const oy = (rect.height - brainHeight) / 2 - BOX.y0 * ih;
      if (image) { image.style.left = `${ox}px`; image.style.top = `${oy}px`; image.style.width = `${iw}px`; image.style.height = `${ih}px`; }
      const hemi = (side) => ({ cx: ox + HEMI[side].cx * iw, cy: oy + HEMI[side].cy * ih, rx: HEMI[side].rx * iw, ry: HEMI[side].ry * ih });
      const seat = (side, index) => { const count = side === 'R' ? RIGHT_SLOTS.length : LEFT_SLOTS.length; const angle = -Math.PI / 2 + index * (Math.PI * 2 / count); const radius = side === 'R' ? .56 : .63; return [Math.cos(angle) * radius, Math.sin(angle) * radius]; };
      const labelNodes = nodes.filter((node) => node.kind === 'label');
      labelNodes.forEach((node) => { const point = seat(node.side, node.seat); const h = hemi(node.side); node.sx = h.cx + point[0] * h.rx; node.sy = h.cy + point[1] * h.ry; });
      const center = { x: ox + .5 * iw, y: oy + .5 * ih };
      nodes.find((node) => node.id === userId).sx = center.x; nodes.find((node) => node.id === userId).sy = center.y;
      byCluster.forEach((ids, cluster) => {
        const first = nodes[ids[0]]; const h = hemi(first.side); const label = nodes[labelIds.get(cluster)]; const point = seat(first.side, label.seat); const maxRadius = first.side === 'R' ? .46 : .34; const radius = Math.min(maxRadius, (first.side === 'R' ? .16 : .13) * Math.sqrt(ids.length + 1));
        ids.forEach((id, index) => { const angle = index * 2.399963 + label.seat * .7; let u = point[0] + Math.cos(angle) * radius * (.5 + .5 * Math.sqrt((index + .5) / ids.length)) / 1; let v = point[1] + Math.sin(angle) * radius * (.5 + .5 * Math.sqrt((index + .5) / ids.length)) / 1; const distance = Math.hypot(u, v); if (distance > .95) { u *= .95 / distance; v *= .95 / distance; } nodes[id].sx = h.cx + u * h.rx; nodes[id].sy = h.cy + v * h.ry; });
      });
      nodes.forEach((node) => { node.x = node.sx; node.y = node.sy; });
      buildGlow();
    };
    const hexa = (hex, alpha) => { const value = Number.parseInt(hex.slice(1), 16); return `rgba(${value >> 16},${value >> 8 & 255},${value & 255},${alpha})`; };
    const buildGlow = () => {
      Object.keys(COLORS).forEach((kind) => {
        if (kind === 'label') return;
        const radius = Math.max(16, Math.round((kind === 'user' ? 90 : 44) * scale));
        const glowCanvas = document.createElement('canvas'); glowCanvas.width = radius * 2; glowCanvas.height = radius * 2;
        const glowContext = glowCanvas.getContext('2d'); const gradient = glowContext.createRadialGradient(radius, radius, 0, radius, radius, radius);
        gradient.addColorStop(0, hexa(COLORS[kind], .52)); gradient.addColorStop(.3, hexa(COLORS[kind], .18)); gradient.addColorStop(.65, hexa(COLORS[kind], .04)); gradient.addColorStop(1, hexa(COLORS[kind], 0));
        glowContext.fillStyle = gradient; glowContext.fillRect(0, 0, radius * 2, radius * 2); glow[kind] = glowCanvas;
      });
    };
    const roundRect = (x, y, width, height, radius) => { context.beginPath(); context.roundRect(x, y, width, height, radius); };
    const drawShape = (node, radius) => {
      context.beginPath();
      if (node.kind === 'emotion' || node.kind === 'prefer') context.roundRect(node.x - radius, node.y - radius, radius * 2, radius * 2, Math.min(radius * .4, 3));
      else if (node.kind === 'exper') { context.moveTo(node.x, node.y - radius * 1.1); context.lineTo(node.x + radius, node.y + radius * .72); context.lineTo(node.x - radius, node.y + radius * .72); context.closePath(); }
      else if (node.kind === 'user') {
        context.arc(node.x, node.y, radius, 0, Math.PI * 2); context.fillStyle = hexa(COLORS.user, .22); context.fill(); context.stroke();
        context.beginPath(); context.arc(node.x, node.y, radius * .6, 0, Math.PI * 2); context.strokeStyle = hexa(COLORS.user, .5); context.stroke();
        context.beginPath(); context.arc(node.x, node.y, Math.max(1.6, radius * .16), 0, Math.PI * 2); context.fillStyle = hexa(COLORS.user, .95); context.fill();
        return;
      } else context.arc(node.x, node.y, radius, 0, Math.PI * 2);
      context.fill(); context.stroke();
      context.beginPath(); context.arc(node.x, node.y, Math.max(.9, radius * .28), 0, Math.PI * 2); context.fillStyle = hexa(COLORS[node.kind] || COLORS.entity, Math.min(1, .5 + node.h * .5)); context.fill();
    };
    const updateCard = () => {
      if (!card) return;
      if (selected >= 0) {
        const node = nodes[selected];
        if (selected !== previousSelected) {
          previousSelected = selected;
          if (cardParts.sw) cardParts.sw.style.borderColor = COLORS[node.kind] || '#fff';
          if (cardParts.id) cardParts.id.textContent = `#${String(1000 + node.id * 7 % 8999).padStart(4, '0')}`;
          if (node.kind === 'label') {
            if (cardParts.kd) cardParts.kd.textContent = node.w;
            if (cardParts.body) cardParts.body.textContent = node.cluster;
            if (cardParts.src) cardParts.src.textContent = `${node.side === 'L' ? 'Sinistro' : 'Destro'} · ${nodes.filter((item) => item.cluster === node.cluster && item.kind !== 'label').length} memorie`;
          } else if (node.kind === 'user') {
            if (cardParts.kd) cardParts.kd.textContent = 'You · owner';
            if (cardParts.body) cardParts.body.textContent = 'Identity · anchor';
            if (cardParts.src) cardParts.src.textContent = 'identity · anchor';
          } else {
            if (cardParts.kd) cardParts.kd.textContent = `${node.kind} · ${node.cluster}`;
            if (cardParts.body) cardParts.body.textContent = node.detail?.desc || node.detail?.text || node.w;
            if (cardParts.src) cardParts.src.textContent = node.side === 'L' ? 'Sinistro · fatti' : 'Destro · profilo';
          }
          card.style.opacity = '1'; cardOn = true;
        }
        const width = card.offsetWidth; const height = card.offsetHeight;
        cardSide = node.x < W / 2 ? -1 : 1;
        cardTargetX = Math.min(Math.max(node.x + (cardSide > 0 ? 26 : -26 - width), 8), Math.max(8, W - width - 8));
        cardTargetY = Math.min(Math.max(node.y - height / 2, 8), Math.max(8, H - height - 8));
      } else if (cardOn) {
        cardOn = false; previousSelected = -1; card.style.opacity = '0';
      }
      if (cardOn) {
        cardX += (cardTargetX - cardX) * .22; cardY += (cardTargetY - cardY) * .22;
        card.style.transform = `translate3d(${cardX.toFixed(1)}px,${cardY.toFixed(1)}px,0)`;
        const node = nodes[selected]; const anchorX = cardSide > 0 ? cardX : cardX + card.offsetWidth; const anchorY = cardY + card.offsetHeight / 2;
        context.strokeStyle = `rgba(255,255,255,${(.4 * node.h).toFixed(3)})`; context.lineWidth = .9; context.beginPath(); context.moveTo(node.x + Math.sign(anchorX - node.x) * (node.r * scale + 5), node.y); context.lineTo(anchorX - Math.sign(anchorX - node.x) * 3, anchorY); context.stroke();
      }
    };
    const draw = (time) => {
      const rect = wrap.getBoundingClientRect(); context.clearRect(0, 0, rect.width, rect.height);
      const seconds = time / 1000;
      nodes.forEach((node) => {
        const progress = Math.min(1, Math.max(0, (seconds - node.dl) / .6));
        node.iv = 1 - (1 - progress) ** 3;
        const wobble = REDUCED ? 0 : 3.4 * scale;
        node.x = node.sx + Math.sin(seconds * node.sp + node.ph) * wobble * node.ax;
        node.y = node.sy + Math.cos(seconds * node.sp * .83 + node.ph2) * wobble * node.ay;
      });
      context.lineCap = 'round';
      let best = -1; let bestDistance = Infinity;
      if (hasPointer) nodes.forEach((node) => {
        const distance = node.kind === 'label' && node.bw
          ? Math.hypot(Math.max(Math.abs(node.x - mouseX) - node.bw / 2, 0), Math.max(Math.abs(node.y - mouseY) - node.bh / 2, 0))
          : Math.max(0, Math.hypot(node.x - mouseX, node.y - mouseY) - node.r * scale);
        if (distance < bestDistance) { bestDistance = distance; best = node.id; }
      });
      if (bestDistance > 26) best = -1;
      selected = best;
      const linked = new Set(selected >= 0 ? edges.flatMap((edge) => edge.a === selected ? [edge.b] : edge.b === selected ? [edge.a] : []) : []);
      nodes.forEach((node) => {
        const target = Math.max(node.id === selected ? 1 : linked.has(node.id) ? .4 : 0, node.act || 0);
        node.h += (target - node.h) * (target ? .22 : .10);
        const distance = Math.hypot(node.x - mouseX, node.y - mouseY);
        const proximity = hasPointer && distance < 120 * scale ? (1 - distance / (120 * scale)) ** 2 : 0;
        node.p += (proximity - node.p) * .14;
        if (node.act > 0) node.act = Math.max(0, node.act - .008);
      });
      edges.forEach((edge) => {
        const a = nodes[edge.a]; const b = nodes[edge.b];
        const active = selected === edge.a || selected === edge.b || linked.has(edge.a) || linked.has(edge.b);
        const style = edge.type === 'cross' ? ['rgba(198,214,240,', .13, 1.1] : edge.type === 'rel' ? ['rgba(170,196,230,', .13, .7] : ['rgba(150,180,220,', .17, .75];
        const lift = Math.max(a.h, b.h) * .4 + Math.max(a.p, b.p) * .15;
        const alpha = (style[1] * (edge.weight || 1) + lift) * Math.min(a.iv, b.iv);
        if (alpha <= .006) return;
        context.strokeStyle = `${style[0]}${alpha.toFixed(3)})`;
        context.lineWidth = style[2] * (1 + lift * 1.4) * Math.max(.7, scale * .8);
        context.beginPath(); context.moveTo(a.x, a.y); if (edge.type === 'cross') context.quadraticCurveTo((a.x + b.x) / 2, (a.y + b.y) * .42 + rect.height * .29, b.x, b.y); else context.lineTo(b.x, b.y); context.stroke();
      });
      context.globalCompositeOperation = 'lighter';
      nodes.forEach((node) => {
        const cached = glow[node.kind]; if (!cached) return;
        const radius = node.r * scale * 3 * (1 + node.h * 1.1 + node.p * .25);
        const idle = REDUCED ? 1 : 1 + .25 * Math.sin(seconds * .78 + node.ph);
        context.globalAlpha = Math.min(1, (.15 * idle + node.h * .6 + node.p * .2) * node.iv);
        context.drawImage(cached, node.x - radius, node.y - radius, radius * 2, radius * 2);
      });
      context.globalAlpha = 1; context.globalCompositeOperation = 'source-over';
      nodes.forEach((node) => {
        if (node.kind === 'label') return;
        const color = COLORS[node.kind] || COLORS.entity; const radius = node.r * scale * (1 + node.h * .5 + node.p * .12) * (.4 + .6 * node.iv);
        const alpha = (.75 + node.h * .25) * node.iv;
        context.strokeStyle = node.id === selected ? '#fff' : hexa(color, alpha); context.fillStyle = hexa(color, alpha * .18); context.lineWidth = Math.max(.8, 1.2 * Math.max(.8, scale * .8)); drawShape(node, radius);
        context.globalAlpha = 1;
        if (node.kind === 'user' && !REDUCED) {
          context.globalCompositeOperation = 'lighter';
          for (let ring = 0; ring < 2; ring += 1) {
            const phase = ((time / 3400) + ring * .5) % 1;
            context.strokeStyle = rgba(COLORS.user, (1 - phase) ** 2 * .35);
            context.lineWidth = 1;
            context.beginPath(); context.arc(node.x, node.y, radius * (1 + phase * 3), 0, Math.PI * 2); context.stroke();
          }
          context.globalCompositeOperation = 'source-over';
        }
      });
      context.font = `${Math.max(8.6, 9.6 * scale)}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`; context.textAlign = 'center'; context.textBaseline = 'middle';
      nodes.filter((node) => node.kind === 'label' || node.kind === 'user').forEach((node) => {
        const word = node.kind === 'user' ? 'you' : node.w;
        const width = context.measureText(word).width + 1.5 * Math.max(8.6, 9.6 * scale);
        const height = 1.9 * Math.max(8.6, 9.6 * scale);
        const centerY = node.kind === 'user' ? node.y + node.r * scale + height * .9 : node.y;
        if (node.kind === 'label' || node.kind === 'user') { node.bw = width; node.bh = height; }
        roundRect(node.x - width / 2, centerY - height / 2, width, height, 5);
        context.fillStyle = `rgba(8,8,8,${(.88 * node.iv).toFixed(3)})`; context.fill();
        context.strokeStyle = node.kind === 'user' ? `rgba(255,255,255,${(.5 * node.iv).toFixed(3)})` : `rgba(255,255,255,${(.5 * node.iv).toFixed(3)})`; context.stroke();
        context.fillStyle = `rgba(255,255,255,${(.95 * node.iv).toFixed(3)})`; context.fillText(word, node.x, centerY + Math.max(8.6, 9.6 * scale) * .06);
      });
      if (selected >= 0) {
        const node = nodes[selected];
        const radius = node.kind === 'label' ? 24 : node.r * 2.4;
        context.save(); context.translate(node.x, node.y); context.rotate(time / 1000 * .35); context.strokeStyle = 'rgba(255,255,255,.42)'; context.lineWidth = .9; context.setLineDash([2.4, 5.6]); context.beginPath(); context.arc(0, 0, radius, 0, Math.PI * 2); context.stroke(); context.restore(); context.setLineDash([]);
      }
      updateCard();
      frame = requestAnimationFrame(draw);
    };
    const onPointerMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left; const y = event.clientY - rect.top;
      mouseX = x; mouseY = y; hasPointer = true;
    };
    const onPointerLeave = () => { hasPointer = false; mouseX = -9999; mouseY = -9999; };
    canvas.addEventListener('pointermove', onPointerMove, { passive: true }); canvas.addEventListener('pointerleave', onPointerLeave, { passive: true });
    const observer = new ResizeObserver(resize); observer.observe(wrap); image?.addEventListener('load', resize); resize(); frame = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); image?.removeEventListener('load', resize); canvas.removeEventListener('pointermove', onPointerMove); canvas.removeEventListener('pointerleave', onPointerLeave); };
  }, [memories]);

  return <div className="brainwrap"><img ref={imageRef} className="brain" src="/images/background.webp" alt="" /><canvas ref={canvasRef} className="fx" /><div ref={cardRef} className="memory-card"><div className="memory-card-head"><i className="memory-card-swatch" /><span className="memory-card-kind">entity</span><span className="memory-card-id">#0000</span></div><div className="memory-card-content">...</div><div className="memory-card-foot"><span className="memory-card-source">-</span></div></div><div className="legend"><span><i className="left-dot" />Sinistro · fatti</span><span><i className="right-dot" />Destro · profilo</span></div></div>;
}
