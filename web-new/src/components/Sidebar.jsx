import { Search, SquarePen, PanelLeft, Plus } from 'lucide-react';

export function Sidebar({ collapsed, onCollapse, sessions, activeId, onSelect, onNew }) {
  return <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
    <div className="side-head"><button className="ico" onClick={onCollapse} title="Comprimi barra laterale"><PanelLeft /></button><span className="grow" /><button className="ico"><Search /></button><button className="ico" onClick={onNew} title="Nuova conversazione"><SquarePen /></button></div>
    {!collapsed && <><div className="side-search"><input placeholder="Cerca conversazioni" /></div><div className="side-scroll"><div className="side-label">Conversazioni<span className="count">{sessions.length}</span></div><ul className="sess">{sessions.map((session) => <li className={session.id === activeId ? 'on' : ''} key={session.id} onClick={() => onSelect(session.id)}><span className="t">{session.title}</span><span className="n">{session.turns.length}</span></li>)}</ul></div><div className="side-foot">VOICEMEM</div></>}
  </aside>;
}

export function SpaceControls({ spaces, spaceId, onSpaceChange, onNewSpace }) {
  return <div className="export"><span className="pill space">◈ <span>{spaces.find((space) => space.id === spaceId)?.name || 'Demo'}</span></span><select className="sel" value={spaceId} onChange={(event) => onSpaceChange(event.target.value)}>{spaces.map((space) => <option value={space.id} key={space.id}>{space.name}</option>)}</select><button className="btn sm" onClick={onNewSpace} title="Crea una nuova Memory Space"><Plus size={15} /></button><select className="sel"><option>Italiano</option><option>EN</option></select><button className="btn sm">Scarica</button></div>;
}
