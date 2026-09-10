import { Settings, X } from "lucide-react";
import { useState } from "react";
import { BrainGraph } from "./BrainGraph";

export function RightPane({
  session,
  memories,
  activeMemoryIds,
  spaces,
  spaceId,
  language,
  user,
  onLogout,
  onSpaceChange,
  onLanguageChange,
  onDownload,
  onNewSpace,
  onSystemPromptChange,
}) {
  const [view, setView] = useState("chat");
  const [downloadScope, setDownloadScope] = useState("all");
  const [configOpen, setConfigOpen] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(
    session?.system_prompt || "",
  );
  const turns = session?.turns || [];
  const spaceName =
    spaces.find((space) => space.id === spaceId)?.name || "Demo";
  const openConfig = () => {
    setSystemPrompt(session?.system_prompt || "");
    setConfigOpen(true);
  };
  const saveConfig = () => {
    onSystemPromptChange(systemPrompt);
    setConfigOpen(false);
  };
  return (
    <section className="rightpane">
      <div className="topbar">
        <div className={`seg ${view === "mem" ? "mem" : ""}`}>
          <span className="seg-thumb" />
          <button
            className={view === "chat" ? "on" : ""}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            className={view === "mem" ? "on" : ""}
            onClick={() => setView("mem")}
          >
            Memory Space
          </button>
        </div>
        <div className="export">
          <span className="pill account" title={user?.email}>
            {user?.display_name || user?.email}
          </span>
          <button className="btn sm" onClick={onLogout}>
            Esci
          </button>
          <span
            className="pill space"
            title="Memory Space corrente - cambiala per cambiare anche il grafo"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <rect x="5" y="11" width="14" height="9" rx="2" />
              <path d="M8 11V8a4 4 0 0 1 8 0v3" />
            </svg>
            <span>{spaceName}</span>
          </span>
          <select
            className="sel"
            value={spaceId}
            onChange={(event) => onSpaceChange(event.target.value)}
            title="Cambia Memory Space"
          >
            {spaces.map((space) => (
              <option value={space.id} key={space.id}>
                {space.name}
              </option>
            ))}
          </select>
          <button
            className="btn sm"
            onClick={onNewSpace}
            title="Crea una nuova Memory Space"
          >
            +
          </button>
          <select
            className="sel"
            value={language}
            onChange={(event) => onLanguageChange(event.target.value)}
            title="Lingua dell'interfaccia e dell'assistente"
          >
            <option value="it">Italiano</option>
            <option value="en">EN</option>
          </select>
          <select
            className="sel"
            value={downloadScope}
            onChange={(event) => setDownloadScope(event.target.value)}
            title="Seleziona l'ambito di esportazione"
          >
            <option value="all">Tutte le memorie dello spazio</option>
            <option value="left">Sinistro · fatti</option>
            <option value="right">Destro · profilo</option>
            <option value="chat">Conversazione corrente</option>
          </select>
          <button className="btn sm" onClick={() => onDownload(downloadScope)}>
            Scarica
          </button>
          <button className="btn sm icon-btn" onClick={openConfig} title="Configura la sessione" aria-label="Configura la sessione">
            <Settings size={15} />
          </button>
        </div>
      </div>
      <div className="viewport">
        {view === "chat" ? (
          <div className="chat-pane">
            <div className="chat-head">
              {session?.title || "Nuova conversazione"}
            </div>
            <div className="chat">
              {turns.length ? (
                turns.map((turn, index) => (
                  <div className={`turn ${turn.role}`} key={index}>
                    <span className="who">
                      {turn.role === "user" ? "Tu" : "VoiceMem"}
                    </span>
                    <div className="said">{turn.text}</div>
                  </div>
                ))
              ) : (
                <div className="empty">
                  <b>Questa conversazione è vuota.</b>Parla o scrivi a sinistra.
                </div>
              )}
            </div>
          </div>
        ) : (
          <BrainGraph memories={memories} activeMemoryIds={activeMemoryIds} />
        )}
      </div>
      {configOpen && (
        <div className="config-modal-backdrop" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setConfigOpen(false);
        }}>
          <div className="config-modal" role="dialog" aria-modal="true" aria-labelledby="session-config-title">
            <div className="config-modal-head">
              <div>
                <span className="auth-mark">SESSION</span>
                <h2 id="session-config-title">Configurazione sessione</h2>
              </div>
              <button className="modal-close" onClick={() => setConfigOpen(false)} title="Chiudi" aria-label="Chiudi">
                <X size={17} />
              </button>
            </div>
            <label className="config-field">
              System prompt
              <textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} placeholder="Istruzioni per l'assistente in questa sessione" rows={10} />
            </label>
            <div className="config-modal-actions">
              <button className="btn" onClick={() => setConfigOpen(false)}>Annulla</button>
              <button className="btn primary" onClick={saveConfig}>Salva</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
