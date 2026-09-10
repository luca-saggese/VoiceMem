import { useEffect, useState } from 'react';

const initialForm = { email: '', password: '', confirmation: '' };

export function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState(initialForm);
  const [token, setToken] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('verified')) setMessage('Email confermata. Ora puoi accedere.');
    const resetToken = params.get('reset');
    if (resetToken) { setToken(resetToken); setMode('reset'); }
  }, []);

  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.value }));
  const submit = async (event) => {
    event.preventDefault(); setBusy(true); setError(''); setMessage('');
    const endpoint = mode === 'login' ? '/api/auth/login' : mode === 'register' ? '/api/auth/register' : mode === 'forgot' ? '/api/auth/forgot' : '/api/auth/reset';
    const body = mode === 'forgot' ? { email: form.email } : mode === 'reset' ? { token, password: form.password, password_confirmation: form.confirmation } : { email: form.email, password: form.password, password_confirmation: form.confirmation };
    try {
      const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(body) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Operazione non riuscita');
      if (mode === 'login') onAuthenticated(data);
      else { setMessage(data.message || 'Operazione completata'); if (mode === 'reset') setMode('login'); }
    } catch (requestError) { setError(requestError.message); }
    finally { setBusy(false); }
  };

  const title = mode === 'login' ? 'Accedi a VoiceMem' : mode === 'register' ? 'Crea il tuo account' : mode === 'forgot' ? 'Recupera account' : 'Nuova password';
  return <main className="auth-screen"><section className="auth-card"><div className="auth-mark">VOICEMEM</div><h1>{title}</h1><p className="auth-sub">La tua memoria vocale, privata e personale.</p>{message && <div className="auth-message">{message}</div>}{error && <div className="auth-error">{error}</div>}
    {mode === 'login' && <a className="google-btn" href="/api/auth/google/start"><span className="google-g">G</span> Continua con Google</a>}
    {mode !== 'forgot' && mode !== 'reset' && <div className="auth-divider"><span>oppure</span></div>}
    <form onSubmit={submit} className="auth-form">
      {mode !== 'reset' && <label>Email<input name="email" type="email" value={form.email} onChange={update} required autoComplete="email" /></label>}
      {mode !== 'forgot' && <label>Password<input name="password" type="password" value={form.password} onChange={update} required minLength="8" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} /></label>}
      {(mode === 'register' || mode === 'reset') && <label>Conferma password<input name="confirmation" type="password" value={form.confirmation} onChange={update} required minLength="8" autoComplete="new-password" /></label>}
      <button className="auth-submit" disabled={busy}>{busy ? 'Attendi…' : mode === 'login' ? 'Accedi' : mode === 'register' ? 'Registrati' : mode === 'forgot' ? 'Invia email di recupero' : 'Salva nuova password'}</button>
    </form>
    <div className="auth-links">{mode === 'login' && <><button onClick={() => { setMode('register'); setError(''); setMessage(''); }}>Crea account</button><button onClick={() => { setMode('forgot'); setError(''); setMessage(''); }}>Password dimenticata?</button></>}{mode !== 'login' && <button onClick={() => { setMode('login'); setError(''); setMessage(''); }}>Torna al login</button>}</div>
  </section></main>;
}
