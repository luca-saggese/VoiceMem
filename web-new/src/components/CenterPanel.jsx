import { useEffect, useRef, useState } from 'react';

function Recall({ title, items, side }) { return <div className={`rc ${side}`}><div className="rc-h">{title}</div><ul>{(items.length ? items : ['Nessuna corrispondenza']).map((item, index) => <li key={`${item}-${index}`}><span>{typeof item === 'string' ? item : item.text}</span>{typeof item !== 'string' && <span className="sc">{item.score?.toFixed?.(2) || ''}</span>}</li>)}</ul></div>; }
export function CenterPanel({ liveInput, reply, recall, status, audioLevel, onSend, onStart, onPause }) {
  const [text, setText] = useState('');
  const waveRef = useRef(null);
  const orbRef = useRef(null);
  const audioLevelRef = useRef(audioLevel);
  audioLevelRef.current = audioLevel;
  useEffect(() => {
    const canvas = orbRef.current;
    if (!canvas) return undefined;
    const context = canvas.getContext('2d');
    const glow = document.createElement('canvas');
    const glowContext = glow.getContext('2d');
    const strands = 3; const lines = 7; const points = 88; const spacing = .06;
    const base = .27; const shrink = .45; const idleAmp = .14; const liveAmp = 1;
    const idleSpeed = .10; const liveSpeed = .5; const rotation = .5; const liveRotation = 1.2;
    const spread = .3; const lineWidth = 1.1; const squish = .88;
    let orbLevel = 0; let orbPhase = 0; let orbRotation = 0; let last = performance.now(); let accumulator = 0;
    const orbPoint = (angle, phase, strand, radius, amount, verticalScale) => {
      const strandPhase = (strand / strands) * Math.PI * 2;
      const currentRadius = radius * (1 + amount * (
        .4 * Math.sin(3 * angle + strandPhase + phase)
        + .24 * Math.sin(5 * angle - strandPhase * 2 + phase * .6)
        + .16 * Math.sin(2 * angle + strandPhase * 1.5 - phase * 1.4)));
      const wobble = .09 * radius * amount * Math.sin(4 * angle - phase * 1.8 + strandPhase);
      return {
        x: currentRadius * Math.cos(angle) + wobble * Math.cos(angle + Math.PI / 2),
        y: currentRadius * Math.sin(angle) * verticalScale + wobble * Math.sin(angle + Math.PI / 2),
      };
    };
    const draw = (time) => {
      const canvas = waveRef.current;
      if (canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = canvas.clientWidth || 96; const height = canvas.clientHeight || 24;
        canvas.width = width * ratio; canvas.height = height * ratio;
        const context = canvas.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
        context.beginPath();
        for (let x = 0; x <= width; x += 1) { const envelope = Math.sin((x / width) * Math.PI); const y = height / 2 + Math.sin(x * .14 - time / 170) * envelope * height * .4 * (status === 'Risposta in corso' ? .8 : .12); x ? context.lineTo(x, y) : context.moveTo(x, y); }
        context.strokeStyle = 'rgba(121,215,162,.85)'; context.lineWidth = 1.5; context.stroke();
      }
      const orb = orbRef.current;
      if (orb) {
        const ratio = Math.min(2, window.devicePixelRatio || 1);
        const width = orb.clientWidth || 264; const height = orb.clientHeight || width;
        if (orb.width !== width * ratio || orb.height !== height * ratio) {
          orb.width = width * ratio; orb.height = height * ratio;
          glow.width = width * ratio; glow.height = height * ratio;
        }
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        glowContext.setTransform(ratio, 0, 0, ratio, 0, 0);
        const dt = Math.min((time - last) / 1000, .1); last = time; accumulator += dt;
        if (orbLevel >= .03 || accumulator >= .05) {
          accumulator = 0;
          const userSpeaking = status === 'In ascolto' && audioLevelRef.current > .02;
          const want = status === 'Risposta in corso' ? .55 : userSpeaking ? .85 : 0;
          orbLevel += (want - orbLevel) * (want > orbLevel ? .2 : .05);
          const amount = idleAmp + (liveAmp - idleAmp) * orbLevel;
          const verticalScale = 1 - (1 - squish) * orbLevel;
          orbPhase += dt * (idleSpeed + (liveSpeed - idleSpeed) * orbLevel);
          orbRotation += dt * (rotation + (liveRotation - rotation) * orbLevel);
          const radius = Math.min(width, height) * base * (1 - shrink * orbLevel);
          glowContext.clearRect(0, 0, width, height); glowContext.save();
          glowContext.translate(width / 2, height / 2); glowContext.rotate(orbRotation);
          const gradient = glowContext.createRadialGradient(0, 0, 0, 0, 0, radius * 2.1);
          gradient.addColorStop(0, 'rgba(58,116,220,1)'); gradient.addColorStop(.55, 'rgba(58,116,220,1)'); gradient.addColorStop(1, 'rgba(150,205,245,1)');
          const half = (lines - 1) / 2; const lineSpacing = radius * spacing;
          glowContext.strokeStyle = gradient; glowContext.lineWidth = lineWidth; glowContext.lineJoin = 'round'; glowContext.lineCap = 'round';
          for (let strand = 0; strand < strands; strand += 1) {
            const spin = orbRotation * spread * strand; const cos = Math.cos(spin); const sin = Math.sin(spin);
            for (let line = 0; line < lines; line += 1) {
              const lineRadius = radius + (line - half) * lineSpacing; const edge = 1 - Math.abs(line - half) / (half + .001);
              glowContext.beginPath();
              for (let point = 0; point <= points; point += 1) {
                const angle = (point / points) * Math.PI * 2; const value = orbPoint(angle, orbPhase, strand, lineRadius, amount, verticalScale);
                const x = value.x * cos - value.y * sin; const y = value.x * sin + value.y * cos;
                point ? glowContext.lineTo(x, y) : glowContext.moveTo(x, y);
              }
              glowContext.globalAlpha = .5 + .5 * edge; glowContext.stroke();
            }
          }
          glowContext.globalAlpha = 1; glowContext.restore();
          context.clearRect(0, 0, width, height); context.save(); context.translate(width / 2, height / 2);
          const background = context.createRadialGradient(0, 0, 0, 0, 0, radius * 2.1);
          background.addColorStop(0, `rgba(60,130,255,${.2 + .28 * orbLevel})`); background.addColorStop(.5, `rgba(30,70,190,${.09 + .13 * orbLevel})`); background.addColorStop(1, 'rgba(0,0,0,0)');
          context.fillStyle = background; context.beginPath(); context.arc(0, 0, radius * 2.1, 0, Math.PI * 2); context.fill(); context.restore();
          context.save(); context.globalCompositeOperation = 'lighter'; context.filter = 'blur(5px)'; context.globalAlpha = .34; context.drawImage(glow, 0, 0, width, height); context.filter = 'blur(2px)'; context.globalAlpha = .42; context.drawImage(glow, 0, 0, width, height); context.filter = 'none'; context.globalAlpha = 1; context.drawImage(glow, 0, 0, width, height); context.restore();
        }
      }
      frame = requestAnimationFrame(draw);
    };
    let frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [status]);
  const submit = (event) => { event.preventDefault(); onSend(text); setText(''); };
  return <section className="center"><canvas ref={orbRef} className="orb" aria-hidden="true" /><div className="panel p-in"><div className="panel-h">Input streaming</div><div className="spk"><span className="av">Tu</span><span>Tu</span><span className="id">speaker 0 · voce riconosciuta</span><span className="levels">▂▅▇▅▂</span></div><div className={`live-text ${liveInput ? '' : 'idle'}`}>{liveInput || 'In attesa della tua voce…'}</div><form onSubmit={submit}><input className="textin" value={text} onChange={(event) => setText(event.target.value)} placeholder="Oppure scrivi e premi Invio" /></form><div className="tags"><span className="tag ent"><b>ENT</b> VoiceMem</span><span className="tag emo"><b>EMO</b> curiosità</span></div></div><div className="panel p-recall"><div className="panel-h">Recupero Top-K</div><div className="recall-cols"><Recall title="Sinistro · fatti" items={recall.left || []} side="l" /><Recall title="Destro · profilo" items={recall.right || []} side="r" /></div></div><div className="panel p-out"><div className="panel-h">Risposta AI</div><div className={`reply ${reply.startsWith('Nessuna') ? 'idle' : ''}`}>{reply}</div><div className="out-foot"><div className="voice"><canvas ref={waveRef} className="wave" /><span className="lab">{status}</span></div><div className="btns"><button className="btn" onClick={onPause} disabled={status === 'Inattivo'}>{status === 'In pausa' ? 'Riprendi' : 'Pausa'}</button><button className="btn primary" onClick={onStart}>{status === 'In ascolto' ? 'Termina sessione' : 'Inizia a parlare'}</button></div></div></div></section>;
}
