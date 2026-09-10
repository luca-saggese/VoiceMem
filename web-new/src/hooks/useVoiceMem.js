import { useCallback, useEffect, useRef, useState } from 'react';

const emptySession = { id: 'local', title: 'Nuova conversazione', turns: [] };
const SAMPLE_RATE = 24000;

export function useVoiceMem() {
  const [spaces, setSpaces] = useState([{ id: 'demo', name: 'Demo', language: 'it', left: [], right: [] }]);
  const [spaceId, setSpaceId] = useState('demo');
  const [sessions, setSessions] = useState([emptySession]);
  const [sessionId, setSessionId] = useState('local');
  const [liveInput, setLiveInput] = useState('');
  const [reply, setReply] = useState('Nessuna risposta ancora.');
  const [recall, setRecall] = useState({ left: [], right: [] });
  const [status, setStatus] = useState('Inattivo');
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [activeMemoryIds, setActiveMemoryIds] = useState([]);
  const sessionsLoaded = useRef(false);
  const socket = useRef(null);
  const mic = useRef({ stream: null, context: null, processor: null, source: null });
  const playback = useRef({ context: null, node: null, ready: null, pending: [], scheduled: [], nextTime: 0, prebuffer: .08, streaming: false, drainPending: false, drainWaiters: [], outputId: '', sampleRate: SAMPLE_RATE, paused: false, workletFailed: false });
  const sessionIdRef = useRef(sessionId);
  const pausedRef = useRef(paused);
  const replyRef = useRef('');
  sessionIdRef.current = sessionId;
  pausedRef.current = paused;

  const loadSpaces = useCallback(async () => {
    try {
      const response = await fetch('/api/spaces');
      if (!response.ok) return;
      const data = await response.json();
      const list = Array.isArray(data) ? data : data.spaces;
      if (list?.length) setSpaces((current) => list.map((item) => ({ ...item, left: item.left || [], right: item.right || [] })));
    } catch { /* backend opzionale durante lo sviluppo UI */ }
  }, []);

  useEffect(() => { loadSpaces(); }, [loadSpaces]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/chat-sessions', { credentials: 'include' })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (cancelled) return;
        const loaded = Array.isArray(data?.sessions) ? data.sessions : [];
        setSessions(loaded.length ? loaded : [emptySession]);
        setSessionId(loaded[0]?.id || emptySession.id);
        sessionsLoaded.current = true;
      })
      .catch(() => { sessionsLoaded.current = true; });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!sessionsLoaded.current) return;
    sessions.forEach((session) => {
      fetch(`/api/chat-sessions/${encodeURIComponent(session.id)}`, {
        method: 'PUT', credentials: 'include', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(session),
      }).catch(() => { /* la conversazione resta disponibile localmente */ });
    });
  }, [sessions]);

  const loadMemories = useCallback(async (id = spaceId) => {
    try {
      const response = await fetch(`/api/memories?space=${encodeURIComponent(id)}`);
      if (!response.ok) return;
      const data = await response.json();
      setSpaces((current) => current.map((space) => space.id === id
        ? { ...space, left: data.left || [], right: data.right || [] }
        : space));
    } catch { /* permette alla UI di partire anche senza backend */ }
  }, [spaceId]);

  useEffect(() => { loadMemories(spaceId); }, [loadMemories, spaceId]);

  const stopMic = useCallback(() => {
    const current = mic.current;
    if (current.processor) current.processor.disconnect();
    if (current.source) current.source.disconnect();
    current.stream?.getTracks().forEach((track) => track.stop());
    mic.current = { stream: null, context: current.context, processor: null, source: null };
    setAudioLevel(0);
  }, []);

  const downsample = (samples, fromRate, toRate) => {
    if (fromRate === toRate) return samples;
    const ratio = fromRate / toRate;
    const output = new Float32Array(Math.floor(samples.length / ratio));
    for (let index = 0; index < output.length; index += 1) output[index] = samples[Math.floor(index * ratio)];
    return output;
  };

  const resample = (samples, fromRate, toRate) => {
    if (fromRate === toRate) return samples;
    const ratio = fromRate / toRate;
    const output = new Float32Array(Math.max(1, Math.round(samples.length / ratio)));
    for (let index = 0; index < output.length; index += 1) {
      const position = index * ratio; const first = Math.floor(position); const second = Math.min(first + 1, samples.length - 1); const fraction = position - first;
      output[index] = samples[first] * (1 - fraction) + samples[second] * fraction;
    }
    return output;
  };

  const reportCheckpoint = useCallback((message) => {
    const state = playback.current;
    if (!state.outputId || socket.current?.readyState !== WebSocket.OPEN) return;
    try {
      socket.current.send(JSON.stringify({
        type: 'playback_checkpoint', output_id: state.outputId,
        rendered_samples: Number(message.renderedSamples || 0),
        sample_rate: Number(message.sampleRate || SAMPLE_RATE),
        buffered_samples: Math.round(Number(message.bufferedMs || 0) * Number(message.sampleRate || SAMPLE_RATE) / 1000),
        state: { started: 'playing', buffer: 'playing', resumed: 'playing', paused: 'paused', underflow: 'stalled', drained: 'drained', interrupted: 'interrupted' }[message.type] || message.type,
      }));
    } catch { /* socket chiuso durante il teardown */ }
  }, []);

  const resolveDrain = useCallback(() => {
    const state = playback.current;
    state.streaming = false;
    state.drainWaiters.splice(0).forEach((resolve) => resolve());
  }, []);

  const initPlayback = useCallback(async () => {
    const state = playback.current;
    if (state.ready) return state.ready;
    state.ready = state.context.audioWorklet.addModule('/pcm-player-worklet.js').then(() => {
      state.node = new AudioWorkletNode(state.context, 'voicemem-pcm-player', { outputChannelCount: [1] });
      state.node.connect(state.context.destination);
      state.node.port.onmessage = (event) => {
        const message = event.data || {};
        reportCheckpoint(message);
        if (message.type === 'drained') resolveDrain();
      };
      state.node.port.postMessage({ type: 'config', prebuffer: state.prebuffer });
      if (state.outputId) state.node.port.postMessage({ type: 'start', outputId: state.outputId, sampleRate: state.sampleRate });
      if (state.paused) state.node.port.postMessage({ type: 'pause' });
      state.pending.splice(0).forEach((item) => state.node.port.postMessage({ type: 'audio', samples: item.samples, sourceFrames: item.sourceFrames }, [item.samples.buffer]));
      if (state.drainPending) { state.drainPending = false; state.node.port.postMessage({ type: 'drain' }); }
    }).catch(() => { state.ready = null; state.workletFailed = true; state.pending.splice(0).forEach((item) => legacyPlayPcm(item.samples)); });
    return state.ready;
  }, [reportCheckpoint, resolveDrain]);

  const legacyPlayPcm = useCallback((samples) => {
    const state = playback.current;
    if (!state.context || !samples.length) return;
    const buffer = state.context.createBuffer(1, samples.length, state.context.sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = state.context.createBufferSource();
    source.buffer = buffer; source.connect(state.context.destination);
    const now = state.context.currentTime;
    const at = Math.max(now, state.nextTime || now + state.prebuffer);
    source.start(at); state.nextTime = at + buffer.duration; state.scheduled.push(source);
    source.onended = () => { state.scheduled = state.scheduled.filter((item) => item !== source); };
  }, []);

  const ensurePlayback = useCallback(async () => {
    const state = playback.current;
    if (!state.context) state.context = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    if (state.context.state === 'suspended') await state.context.resume();
    await initPlayback();
    return state;
  }, [initPlayback]);

  const playPcm = useCallback(async (buffer) => {
    if (buffer.byteLength < 2) return;
    const state = await ensurePlayback();
    const bytes = new Int16Array(buffer, 0, Math.floor(buffer.byteLength / 2));
    const native = new Float32Array(bytes.length);
    for (let index = 0; index < bytes.length; index += 1) native[index] = bytes[index] / 32768;
    const samples = resample(native, SAMPLE_RATE, state.context.sampleRate);
    state.streaming = true;
    if (state.workletFailed) legacyPlayPcm(samples);
    else if (state.node) state.node.port.postMessage({ type: 'audio', samples, sourceFrames: bytes.length }, [samples.buffer]);
    else state.pending.push({ samples, sourceFrames: bytes.length });
  }, [ensurePlayback, legacyPlayPcm]);

  const stopPlayback = useCallback((reason = 'reset') => {
    const state = playback.current;
    state.pending.length = 0; state.drainPending = false; state.outputId = ''; state.paused = false;
    if (state.node) state.node.port.postMessage({ type: 'clear', reason });
    state.scheduled.splice(0).forEach((source) => { try { source.stop(); } catch { /* già terminato */ } });
    state.nextTime = state.context?.currentTime || 0;
    resolveDrain();
  }, [resolveDrain]);

  const finishPlayback = useCallback(() => {
    const state = playback.current;
    if (state.workletFailed) { resolveDrain(); return; }
    if (state.node) state.node.port.postMessage({ type: 'drain' });
    else state.drainPending = true;
  }, [resolveDrain]);

  const close = useCallback(() => {
    stopMic(); stopPlayback();
    if (socket.current) { socket.current.onclose = null; socket.current.close(); socket.current = null; }
    setLive(false); setPaused(false); setStatus('Inattivo');
  }, [stopMic]);

  const start = useCallback(async () => {
    if (live) { close(); return; }
    if (!navigator.mediaDevices?.getUserMedia) { setStatus('Microfono non disponibile'); return; }
    try {
      await ensurePlayback();
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
      const connection = new WebSocket(`${protocol}://${location.host}/ws?space=${encodeURIComponent(spaceId)}`);
      connection.binaryType = 'arraybuffer';
      socket.current = connection;
      connection.onopen = async () => {
        setLive(true); setStatus('In ascolto');
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
          const context = new (window.AudioContext || window.webkitAudioContext)();
          await context.resume();
          const source = context.createMediaStreamSource(stream);
          const processor = context.createScriptProcessor(2048, 1, 1);
          processor.onaudioprocess = (event) => {
            if (pausedRef.current || socket.current?.readyState !== WebSocket.OPEN) return;
            const input = event.inputBuffer.getChannelData(0);
            const samples = downsample(input, context.sampleRate, SAMPLE_RATE);
            let peak = 0;
            for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
            setAudioLevel(Math.min(1, peak * 2.5));
            const pcm = new Int16Array(samples.length);
            for (let index = 0; index < samples.length; index += 1) pcm[index] = Math.max(-1, Math.min(1, samples[index])) * 0x7fff;
            socket.current.send(pcm.buffer);
          };
          source.connect(processor); processor.connect(context.destination);
          mic.current = { stream, context, processor, source };
        } catch (error) { setStatus(`Microfono non disponibile: ${error.name || 'errore'}`); close(); }
      };
      connection.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) { playPcm(event.data); return; }
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === 'session_ready') setStatus('In ascolto');
        if (message.type === 'partial_transcript') setLiveInput(message.text || '');
        if (message.type === 'user_transcript') {
          setLiveInput(message.text || '');
          setSessions((current) => current.map((session) => session.id === sessionIdRef.current
            ? { ...session, turns: [...session.turns, { role: 'user', text: message.text || '' }] } : session));
        }
        if (message.type === 'memory_hits') {
          const usedIds = [...new Set([
            ...(message.left_brain || []).map((item) => item.memory_id).filter(Boolean),
            ...(message.right_brain_hits || []).map((item) => item.memory_id).filter(Boolean),
          ].map(String))];
          setActiveMemoryIds(usedIds);
          setRecall({ left: (message.left_brain || []).map((item) => ({ text: item.text, score: Number(item.score || 0) })), right: (message.right_brain_hits || []).filter((item) => !item.internal).map((item) => ({ text: item.content, score: Number(item.priority || 0) })) });
          loadMemories(spaceId);
        }
        if (message.type === 'answer_start') {
          const state = playback.current;
          stopPlayback('reset');
          state.outputId = String(message.output_id || ''); state.sampleRate = Number(message.sample_rate || SAMPLE_RATE); state.paused = false;
          if (state.node) state.node.port.postMessage({ type: 'start', outputId: state.outputId, sampleRate: state.sampleRate });
          setStatus('Risposta in corso'); replyRef.current = ''; setReply('');
        }
        if (message.type === 'answer_delta') {
          replyRef.current += message.text || '';
          setReply(replyRef.current);
        }
        if (message.type === 'answer_pause') { playback.current.paused = true; playback.current.node?.port.postMessage({ type: 'pause' }); }
        if (message.type === 'answer_resume') { playback.current.paused = false; playback.current.node?.port.postMessage({ type: 'resume' }); }
        if (message.type === 'answer_interrupt') { stopPlayback('interrupted'); setStatus('In ascolto'); }
        if (message.type === 'answer_done') {
          const text = replyRef.current.trim();
          if (text) setSessions((current) => current.map((session) => session.id === sessionIdRef.current
            ? { ...session, turns: [...session.turns, { role: 'assistant', text }] } : session));
          finishPlayback();
          window.setTimeout(() => loadMemories(spaceId), 400);
          setStatus('In ascolto');
        }
      };
      connection.onerror = () => setStatus('Backend disconnesso');
      connection.onclose = () => { stopMic(); setLive(false); setStatus('Inattivo'); };
    } catch { setStatus('Backend disconnesso'); }
  }, [close, finishPlayback, live, loadMemories, playPcm, sessionId, spaceId, stopMic, stopPlayback]);

  const togglePause = useCallback(() => {
    setPaused((value) => { const next = !value; pausedRef.current = next; if (next) stopPlayback('pause'); return next; });
    setStatus((value) => value === 'In pausa' ? 'In ascolto' : 'In pausa');
  }, [stopPlayback]);

  const sendText = useCallback((text) => {
    const value = text.trim();
    if (!value) return;
    setLiveInput(value);
    setStatus('Risposta in corso');
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: 'user_text', text: value }));
    } else {
      setReply('Backend non connesso. Premi “Inizia a parlare” per collegarti.');
      setStatus('Inattivo');
    }
  }, [sessionId]);

  useEffect(() => () => { stopMic(); stopPlayback(); socket.current?.close(); }, [stopMic, stopPlayback]);

  const createSession = useCallback(() => {
    const id = `session-${Date.now()}`;
    setSessions((current) => [...current, { id, title: 'Nuova conversazione', turns: [] }]);
    setSessionId(id);
    setLiveInput('');
    setReply('Nessuna risposta ancora.');
  }, []);

  return { spaces, spaceId, setSpaceId, sessions, sessionId, setSessionId, liveInput, reply, recall, status, live, paused, audioLevel, activeMemoryIds, sendText, start, togglePause, createSession, loadMemories };
}
