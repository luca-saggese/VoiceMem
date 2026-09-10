import { Search, SquarePen, PanelLeft, Plus, LogOut, X, Cpu } from 'lucide-react';
import { useEffect, useState } from 'react';

export function Sidebar({ collapsed, onCollapse, sessions, activeId, onSelect, onNew, user, onLogout }) {
  const [accountOpen, setAccountOpen] = useState(false);
  const [devices, setDevices] = useState([]);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const [deviceId, setDeviceId] = useState('');
  const [deviceName, setDeviceName] = useState('');
  const [deviceError, setDeviceError] = useState('');
  const [deviceBusy, setDeviceBusy] = useState(false);
  const initials = (user?.display_name || user?.email || 'U').slice(0, 1).toUpperCase();
  useEffect(() => {
    fetch('/api/devices', { credentials: 'include' })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => setDevices(Array.isArray(data?.devices) ? data.devices : []))
      .catch(() => setDevices([]));
  }, []);
  const openDeviceModal = () => { setDeviceId(''); setDeviceName(''); setDeviceError(''); setDeviceModalOpen(true); };
  const addDevice = async (event) => {
    event.preventDefault();
    if (!/^\d{6}$/.test(deviceId)) { setDeviceError('Inserisci esattamente 6 numeri.'); return; }
    if (!deviceName.trim()) { setDeviceError('Inserisci un nome per il device.'); return; }
    setDeviceBusy(true); setDeviceError('');
    try {
      const response = await fetch('/api/devices', { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device_id: deviceId, name: deviceName.trim() }) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Impossibile registrare il device');
      setDevices((current) => [...current, data]); setDeviceModalOpen(false);
    } catch (error) { setDeviceError(error.message); }
    finally { setDeviceBusy(false); }
  };
  return <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
    <div className="side-head"><button className="ico" onClick={onCollapse} title="Comprimi barra laterale"><PanelLeft /></button><span className="grow" /><button className="ico"><Search /></button><button className="ico" onClick={onNew} title="Nuova conversazione"><SquarePen /></button></div>
    {!collapsed && <><div className="side-search"><input placeholder="Cerca conversazioni" /></div><div className="side-scroll"><div className="side-label device-label">Devices<button className="section-add" onClick={openDeviceModal} title="Aggiungi device Xiaozhi"><Plus size={15} /></button></div><ul className="device-list">{devices.map((device) => <li key={device.id}><span className="device-icon"><Cpu size={14} /></span><span className="t">{device.name}</span><span className="device-id">{device.device_id}</span></li>)}{!devices.length && <li className="device-empty">Nessun device registrato</li>}</ul><div className="side-label">Conversazioni<span className="count">{sessions.length}</span></div><ul className="sess">{sessions.map((session) => <li className={session.id === activeId ? 'on' : ''} key={session.id} onClick={() => onSelect(session.id)}><span className="t">{session.title}</span><span className="n">{session.turns.length}</span></li>)}</ul></div><div className="side-account"><button className="account-trigger" onClick={() => setAccountOpen((open) => !open)} aria-expanded={accountOpen}><span className="account-avatar">{initials}</span><span className="account-copy"><strong>{user?.display_name || user?.email || 'Account'}</strong><small>{user?.display_name ? user.email : ''}</small></span><span className="account-chevron">⌄</span></button>{accountOpen && <div className="account-menu"><div className="account-menu-head">{user?.email}</div><button className="account-logout" onClick={onLogout}><LogOut size={15} /> Esci</button></div>}</div>{deviceModalOpen && <div className="device-modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setDeviceModalOpen(false); }}><form className="device-modal" onSubmit={addDevice}><div className="device-modal-head"><div><span className="auth-mark">XIAOZHI</span><h2>Aggiungi device</h2></div><button type="button" className="modal-close" onClick={() => setDeviceModalOpen(false)} title="Chiudi"><X size={17} /></button></div><p className="device-modal-sub">Registra il dispositivo associato al tuo account.</p>{deviceError && <div className="device-form-error">{deviceError}</div>}<label>Numero identificativo<input value={deviceId} onChange={(event) => setDeviceId(event.target.value.replace(/\D/g, '').slice(0, 6))} inputMode="numeric" pattern="[0-9]{6}" maxLength="6" placeholder="000000" autoFocus /></label><label>Nome del device<input value={deviceName} onChange={(event) => setDeviceName(event.target.value)} maxLength="80" placeholder="es. Xiaozhi studio" /></label><div className="device-modal-actions"><button type="button" className="btn" onClick={() => setDeviceModalOpen(false)}>Annulla</button><button type="submit" className="btn primary" disabled={deviceBusy}>{deviceBusy ? 'Salvataggio...' : 'Aggiungi device'}</button></div></form></div>}</>}
  </aside>;
}

export function SpaceControls({ spaces, spaceId, onSpaceChange, onNewSpace }) {
  return <div className="export"><span className="pill space">◈ <span>{spaces.find((space) => space.id === spaceId)?.name || 'Demo'}</span></span><select className="sel" value={spaceId} onChange={(event) => onSpaceChange(event.target.value)}>{spaces.map((space) => <option value={space.id} key={space.id}>{space.name}</option>)}</select><button className="btn sm" onClick={onNewSpace} title="Crea una nuova Memory Space"><Plus size={15} /></button><select className="sel"><option>Italiano</option><option>EN</option></select><button className="btn sm">Scarica</button></div>;
}
