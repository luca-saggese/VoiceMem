import { StrictMode, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AuthScreen } from './components/AuthScreen';
import { Sidebar } from './components/Sidebar';
import { CenterPanel } from './components/CenterPanel';
import { RightPane } from './components/RightPane';
import { useVoiceMem } from './hooks/useVoiceMem';
import './styles/voicemem.css';
import './styles/overrides.css';
import './styles/brain-card.css';
import './styles/responsive-fix.css';
import './styles/auth.css';

function VoiceApp({ user, onLogout }) {
  const [collapsed, setCollapsed] = useState(false);
  const [language, setLanguage] = useState(() => localStorage.getItem('vm-lang') || 'it');
  const vm = useVoiceMem();
  const activeSession = vm.sessions.find((session) => session.id === vm.sessionId);
  const activeSpace = vm.spaces.find((space) => space.id === vm.spaceId);
  const graphMemories = useMemo(() => ({ left: activeSpace?.left || [], right: activeSpace?.right || [] }), [activeSpace?.left, activeSpace?.right]);

  const changeSpace = (id) => {
    vm.setSpaceId(id);
  };

  const createSpace = async () => {
    const name = window.prompt('Nome della nuova Memory Space');
    if (!name?.trim()) return;
    const response = await fetch('/api/spaces', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim(), language: 'it' }) });
    if (!response.ok) return;
    const created = await response.json();
    vm.setSpaceId(created.id);
    await vm.loadMemories(created.id);
  };

  const changeLanguage = async (value) => {
    const nextLanguage = value === 'en' ? 'en' : 'it';
    setLanguage(nextLanguage);
    localStorage.setItem('vm-lang', nextLanguage);
    try { await fetch('/api/lang', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lang: nextLanguage }) }); } catch { /* backend opzionale durante lo sviluppo UI */ }
  };

  const download = (scope) => {
    const space = activeSpace || { id: 'demo', name: 'Demo', left: [], right: [] };
    let content; let filename;
    if (scope === 'chat') {
      if (!activeSession?.turns?.length) return;
      content = `# ${activeSession.title}\n\n> Memory Space: ${space.name}\n\n${activeSession.turns.map((turn) => `**${turn.role === 'user' ? 'Tu' : 'VoiceMem'}**\n\n${turn.text}`).join('\n\n---\n\n')}`;
      filename = `voicemem-${space.id}-chat.md`;
    } else {
      const output = { exported_at: new Date().toISOString(), space: space.id, space_name: space.name };
      if (scope !== 'right') output.left_brain = space.left || [];
      if (scope !== 'left') output.right_brain = space.right || [];
      content = JSON.stringify(output, null, 2);
      filename = `voicemem-${space.id}-${scope}.json`;
    }
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([content], { type: scope === 'chat' ? 'text/markdown' : 'application/json' })); link.download = filename; link.click(); URL.revokeObjectURL(link.href);
  };

  return <main className={`app ${collapsed ? 'rail-collapsed' : ''}`}>
    <Sidebar collapsed={collapsed} onCollapse={() => setCollapsed((value) => !value)} sessions={vm.sessions} activeId={vm.sessionId} onSelect={vm.setSessionId} onNew={vm.createSession} />
    <CenterPanel liveInput={vm.liveInput} reply={vm.reply} recall={vm.recall} status={vm.status} audioLevel={vm.audioLevel} onSend={vm.sendText} onStart={vm.start} onPause={vm.togglePause} />
    <RightPane session={activeSession} memories={graphMemories} spaces={vm.spaces} spaceId={vm.spaceId} language={language} user={user} onLogout={onLogout} onLanguageChange={changeLanguage} onDownload={download} onSpaceChange={changeSpace} onNewSpace={createSpace} />
  </main>;
}

function AuthGate() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    fetch('/api/auth/me', { credentials: 'include' })
      .then((response) => response.ok ? response.json() : null)
      .then(setUser)
      .finally(() => setLoading(false));
  }, []);
  if (loading) return <div className="auth-loading">VoiceMem</div>;
  if (!user) return <AuthScreen onAuthenticated={setUser} />;
  const logout = async () => { await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' }); setUser(null); };
  return <VoiceApp user={user} onLogout={logout} />;
}

createRoot(document.getElementById('root')).render(<StrictMode><AuthGate /></StrictMode>);
