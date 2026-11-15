/* Rulebook Search Engine — frontend. No external assets. */
(function () {
  "use strict";
  const UI_VERSION = "0.3.3";   // shown in Settings so a stale cached copy of the UI is easy to spot
  const $ = (sel, el = document) => el.querySelector(sel);
  const form = $("#search-form"), input = $("#q"), go = $("#go");
  const statusEl = $("#status"), statusText = $("#status-text"), docinfo = $("#docinfo");
  const setup = $("#setup"), steps = $("#steps"), setupError = $("#setup-error");
  const meta = $("#meta"), results = $("#results");
  const pdfpane = $("#pdfpane"), pdfframe = $("#pdfframe"), pdfLabel = $("#pdf-label"), pdfOpen = $("#pdf-open");
  const tpl = $("#tpl-result");
  const scopeChip = $("#scope-chip");
  const settingsBtn = $("#settings-btn"), settings = $("#settings");
  const home = $("#home"), pinsList = $("#pins"), recentList = $("#recent"), pinsBlock = $("#pins-block"), recentBlock = $("#recent-block");

  let ready = false, seq = 0, inflight = null, pollTimer = null;
  const SEARCH_TIMEOUT_MS = 12000;

  // ---------- status / setup ----------
  function renderStatus(s) {
    statusEl.dataset.state = s.state === "ready" ? "ready" : s.state === "error" ? "error" : "loading";
    statusText.textContent = s.state === "ready" ? "Ready" : s.state === "error" ? "Error" : "Setting up…";
    const d = s.document || {};
    if (d.document_title) {
      docinfo.textContent = `${d.document_title} — ${d.document_version} · ${d.pdf_pages} pages · ${s.counts.rules} rules · UI ${UI_VERSION}`;
      docinfo.title = `PDF: ${d.pdf_file}\nSHA-256: ${d.pdf_sha256}\nIndex built: ${d.built_at}\nEmbedding: ${s.models.embedding}\nReranker: ${s.models.reranker} (${s.models.device})`;
    }
    steps.innerHTML = "";
    (s.steps || []).forEach(st => {
      const li = document.createElement("li");
      li.dataset.status = st.status;
      li.textContent = st.name + " ";
      if (st.detail) { const sm = document.createElement("small"); sm.textContent = "— " + st.detail; li.appendChild(sm); }
      steps.appendChild(li);
    });
    if (s.state === "error") {
      setupError.hidden = false;
      setupError.textContent = "Setup failed: " + (s.error || "unknown error") + ". Fix the problem (see README) and restart the app.";
    }
    const wasReady = ready;
    ready = s.state === "ready";
    input.disabled = go.disabled = !ready;
    if (ready) {
      setup.hidden = true;
      if (!wasReady) { input.focus(); maybeSearchFromHash(); syncEmpty(); renderHome(); }
    } else {
      setup.hidden = false;
    }
  }

  async function poll() {
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      const s = await r.json();
      renderStatus(s);
      if (s.state === "ready" || s.state === "error") { clearInterval(pollTimer); pollTimer = null; if (s.state === "error") return; }
    } catch (e) {
      statusEl.dataset.state = "error";
      statusText.textContent = "Server unreachable";
      setup.hidden = false;
      setupError.hidden = false;
      setupError.textContent = "Cannot reach the local server. If you closed the terminal window, start the app again.";
    }
  }
  pollTimer = setInterval(poll, 1000);
  poll();


  // ---------- motion: interruptible, velocity-aware springs ----------
  // Apple-style parameterisation: response (s) = period of the undamped spring, damping = ratio (1 = critically damped, no bounce).
  // Animations always continue from the live value and velocity, so they can be reversed mid-flight.
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  function spring(initial, opts) {
    const s = { value: initial, target: initial, velocity: 0, raf: 0 };
    const w0 = (2 * Math.PI) / (opts.response || 0.4), zeta = opts.damping == null ? 1 : opts.damping, eps = opts.epsilon || 0.001;
    let last = 0;
    function frame(t) {
      const dt = Math.min(0.064, last ? (t - last) / 1000 : 1 / 60); last = t;
      const steps = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / steps;   // sub-stepped semi-implicit Euler (stable at any frame rate)
      for (let i = 0; i < steps; i++) {
        const a = -w0 * w0 * (s.value - s.target) - 2 * zeta * w0 * s.velocity;
        s.velocity += a * h; s.value += s.velocity * h;
      }
      const settled = Math.abs(s.value - s.target) < eps && Math.abs(s.velocity) < eps * 10;
      if (settled) { s.value = s.target; s.velocity = 0; }
      opts.onUpdate(s.value);
      if (settled) { s.raf = 0; last = 0; if (opts.onRest) opts.onRest(); }
      else s.raf = requestAnimationFrame(frame);
    }
    s.to = (target, velocity) => {
      s.target = target; if (velocity != null) s.velocity = velocity;
      if (reduceMotion.matches) return s.set(target, true);
      if (!s.raf) { last = 0; s.raf = requestAnimationFrame(frame); }
      return s;
    };
    s.set = (v, rest) => { cancelAnimationFrame(s.raf); s.raf = 0; last = 0; s.target = s.value = v; s.velocity = 0; opts.onUpdate(v); if (rest && opts.onRest) opts.onRest(); return s; };
    return s;
  }

  // iOS-style segmented control: the thumb slides to the chosen segment on a critically damped spring.
  function segmented(root, onChange) {
    const thumb = $(".seg-thumb", root), btns = Array.from(root.querySelectorAll("button[role=radio]"));
    const pos = spring(0, { response: 0.35, damping: 1, epsilon: 0.05, onUpdate: x => { thumb.style.transform = `translateX(${x}px)`; } });
    let current = null;
    function layout(animate) {
      const b = btns.find(x => x.dataset.value === current);
      if (!b || !root.offsetWidth) return;            // not laid out yet (popover closed): measured again when it opens
      thumb.style.width = b.offsetWidth + "px";
      const x = b.offsetLeft - 2;                     // thumb sits at left:2px inside the 2px padding
      if (animate) pos.to(x); else pos.set(x);
    }
    function select(v, { animate = true, silent = false } = {}) {
      if (!btns.some(b => b.dataset.value === v)) return;
      const changed = v !== current;
      current = v;
      btns.forEach(b => { b.setAttribute("aria-checked", String(b.dataset.value === v)); b.tabIndex = b.dataset.value === v ? 0 : -1; });
      layout(animate);
      if (changed && !silent) onChange(v);
    }
    btns.forEach(b => b.addEventListener("click", () => select(b.dataset.value)));
    root.addEventListener("keydown", e => {
      const i = btns.findIndex(b => b.dataset.value === current);
      const d = (e.key === "ArrowRight" || e.key === "ArrowDown") ? 1 : (e.key === "ArrowLeft" || e.key === "ArrowUp") ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      const n = btns[(i + d + btns.length) % btns.length];
      select(n.dataset.value); n.focus();
    });
    return { select, layout, get value() { return current; } };
  }

  // ---------- preferences (persisted) ----------
  const PREFS_KEY = "rse.prefs";
  const SCOPES = ["ev", "cv", "all"];
  const prefs = { scope: "ev", live: true };
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") || {};
    if (SCOPES.includes(saved.scope)) prefs.scope = saved.scope;
    if (typeof saved.live === "boolean") prefs.live = saved.live;
  } catch (e) { /* ignore corrupt prefs */ }
  function savePrefs() { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (e) { /* private mode etc. */ } }


  function updateScopeChip() {
    scopeChip.textContent = prefs.scope === "all" ? "All rules" : prefs.scope.toUpperCase();
    scopeChip.title = (prefs.scope === "all" ? "All rules" : prefs.scope.toUpperCase() + " rules and general rules") + " — change under Settings › Preferences";
    scopeChip.dataset.scope = prefs.scope;
  }

  const scopeSeg = segmented($('.segmented[data-pref="scope"]'), v => setScope(v));
  const liveSeg = segmented($('.segmented[data-pref="live"]'), v => { prefs.live = v === "on"; savePrefs(); if (prefs.live) scheduleLive(); else cancelLive(); });
  scopeSeg.select(prefs.scope, { animate: false, silent: true });
  liveSeg.select(prefs.live ? "on" : "off", { animate: false, silent: true });
  updateScopeChip();

  // ---------- settings popover (anchored to the gear; materialises with blur+scale on a spring; same path in and out) ----------
  let popOpen = false;
  const popSpring = spring(0, {
    response: 0.3, damping: 1,
    onUpdate: p => {
      settings.style.opacity = String(Math.max(0, Math.min(1, p)));
      settings.style.transform = `translateY(${(1 - p) * -6}px) scale(${0.9 + 0.1 * p})`;
    },
    onRest: () => { if (!popOpen) settings.hidden = true; }
  });
  function openSettings() {
    if (popOpen) return;
    popOpen = true;
    settings.hidden = false;                          // must be laid out before the controls can measure themselves
    settingsBtn.setAttribute("aria-expanded", "true");
    scopeSeg.layout(false); liveSeg.layout(false);
    popSpring.to(1);
    const first = settings.querySelector('button[aria-checked="true"]');
    if (first) first.focus({ preventScroll: true });
  }
  function closeSettings() {
    if (!popOpen) return;
    popOpen = false;
    settingsBtn.setAttribute("aria-expanded", "false");
    if (settings.contains(document.activeElement)) settingsBtn.focus({ preventScroll: true });
    popSpring.to(0);
  }
  settingsBtn.addEventListener("click", () => { if (popOpen) closeSettings(); else openSettings(); });
  document.addEventListener("pointerdown", e => { if (popOpen && !settings.contains(e.target) && !settingsBtn.contains(e.target)) closeSettings(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && popOpen) { e.preventDefault(); closeSettings(); } });
  addEventListener("resize", () => { if (popOpen) { scopeSeg.layout(false); liveSeg.layout(false); } });

  // ---------- search ----------
  function scope() { return prefs.scope; }
  function setScope(v, { search = true } = {}) {
    if (!SCOPES.includes(v)) return;
    const changed = v !== prefs.scope;
    prefs.scope = v; savePrefs(); updateScopeChip();
    scopeSeg.select(v, { silent: true });
    if (changed && search && input.value.trim()) runSearch(input.value);
  }

  form.addEventListener("submit", ev => { ev.preventDefault(); cancelLive(); runSearch(input.value); });

  // ---------- search as you type ----------
  // The server runs searches one at a time (reranker), and aborting a request does not stop work already started.
  // So: fire only after a typing pause, never while a search is still in flight (the newest text is sent once it finishes),
  // and never resend the text that was last searched. Enter always searches immediately.
  const LIVE_DELAY_MS = 350, LIVE_MIN_CHARS = 3;
  let liveTimer = null, livePending = false, lastSubmitted = "";
  function cancelLive() { clearTimeout(liveTimer); liveTimer = null; livePending = false; }
  function scheduleLive() {
    cancelLive();
    if (!prefs.live || !ready) return;
    const q = input.value.trim();
    if (q.length < LIVE_MIN_CHARS || q === lastSubmitted) return;
    liveTimer = setTimeout(fireLive, LIVE_DELAY_MS);
  }
  function fireLive() {
    liveTimer = null;
    const q = input.value.trim();
    if (!prefs.live || q.length < LIVE_MIN_CHARS || q === lastSubmitted) return;
    if (inflight) { livePending = true; return; }   // picked up by runSearch when the current search finishes
    runSearch(q);
  }
  input.addEventListener("input", () => { syncEmpty(); scheduleLive(); });

  // ---------- recent searches & pinned rules (localStorage) ----------
  const RECENT_KEY = "rse.recent", PINS_KEY = "rse.pins", RECENT_MAX = 10, PINS_MAX = 30;
  function loadList(k) { try { const v = JSON.parse(localStorage.getItem(k) || "[]"); return Array.isArray(v) ? v : []; } catch (e) { return []; } }
  function saveList(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode etc. */ } }
  let recent = loadList(RECENT_KEY), pins = loadList(PINS_KEY);
  function noteRecent(q) {
    q = q.trim(); if (!q) return;
    const lq = q.toLowerCase();
    // the same query moves to the top; a query typed progressively ("brak" → "brake light") replaces its shorter prefix
    recent = recent.filter(r => { const lr = String(r.q).toLowerCase(); return lr !== lq && !lq.startsWith(lr); });
    recent.unshift({ q, t: Date.now() });
    recent = recent.slice(0, RECENT_MAX); saveList(RECENT_KEY, recent); renderHome();
  }
  function isPinned(code) { return pins.some(p => p.code === code); }
  function togglePin(r) {
    if (isPinned(r.code)) pins = pins.filter(p => p.code !== r.code);
    else pins.unshift({ code: r.code, title: r.section_title || r.title || "", page: (r.pages || [])[0] || null });
    pins = pins.slice(0, PINS_MAX); saveList(PINS_KEY, pins); renderHome();
    results.querySelectorAll(".rule").forEach(c => { if (c._syncPin) c._syncPin(); });
  }
  function homeItem(label, sub, onOpen, onRemove, removeTitle) {
    const li = el("li");
    const b = el("button", "homeitem"); b.type = "button";
    if (sub != null) { b.appendChild(el("span", "code", label)); b.appendChild(el("span", "sub", sub)); } else b.appendChild(el("span", "q", label));
    b.onclick = onOpen;
    const x = el("button", "x", "×"); x.type = "button"; x.title = removeTitle; x.setAttribute("aria-label", removeTitle); x.onclick = onRemove;
    li.append(b, x);
    return li;
  }
  function openQuery(q) { input.value = q; syncEmpty(); cancelLive(); runSearch(q); }
  function renderHome() {
    pinsList.innerHTML = ""; recentList.innerHTML = "";
    pins.forEach(p => pinsList.appendChild(homeItem(p.code, p.title || "", () => openQuery(p.code), () => togglePin({ code: p.code }), "Unpin " + p.code)));
    recent.forEach(r => recentList.appendChild(homeItem(r.q, null, () => openQuery(r.q), () => { recent = recent.filter(x => x !== r); saveList(RECENT_KEY, recent); renderHome(); }, "Remove from recent searches")));
    pinsBlock.hidden = !pins.length; recentBlock.hidden = !recent.length;
    home.hidden = !ready || (!pins.length && !recent.length);
  }
  $("#recent-clear").onclick = () => { recent = []; saveList(RECENT_KEY, recent); renderHome(); };
  function syncEmpty() { document.body.classList.toggle("empty", !input.value.trim()); }
  syncEmpty();

  // ---------- keyboard navigation ----------
  let selected = -1;
  function ruleCards() { return Array.from(results.querySelectorAll(":scope > .rule")); }
  function selectCard(i) {
    const cards = ruleCards(); if (!cards.length) return;
    i = Math.max(0, Math.min(cards.length - 1, i));
    cards.forEach((c, k) => c.classList.toggle("selected", k === i));
    selected = i;
    cards[i].focus({ preventScroll: true });
    cards[i].scrollIntoView({ block: "nearest", behavior: reduceMotion.matches ? "auto" : "smooth" });
  }
  function focusSearch() { input.focus(); input.select(); }
  document.addEventListener("keydown", e => {
    if (popOpen) return;
    const t = e.target, mod = e.metaKey || e.ctrlKey || e.altKey;
    if ((e.key === "/" && t !== input && !mod) || ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === "k")) { e.preventDefault(); focusSearch(); return; }
    const cards = ruleCards();
    if (t === input) { if (e.key === "ArrowDown" && cards.length) { e.preventDefault(); selectCard(0); } return; }
    const card = t.closest ? t.closest(".rule") : null;
    const onCard = t === document.body || (card && t === card);                       // the card itself (or nothing) has focus
    const inCard = !!card && !t.matches("input, textarea, select, [contenteditable]"); // a button / link inside a card has focus
    if (!onCard && !inCard) return;                                                    // typing in some other control: leave it alone
    const idx = card ? cards.indexOf(card) : selected;
    if (e.key === "ArrowDown" || (e.key === "j" && onCard)) { e.preventDefault(); selectCard(idx < 0 ? 0 : idx + 1); }
    else if (e.key === "ArrowUp" || (e.key === "k" && onCard)) { e.preventDefault(); if (idx <= 0) focusSearch(); else selectCard(idx - 1); }
    else if (e.key === "Escape") { e.preventDefault(); if (pdfShown) hidePdf(); else focusSearch(); }   // first Esc closes the PDF, next one returns to search
    else if (!onCard) return;                                                           // Enter / letters on an inner button keep their own meaning
    else if (e.key === "Enter" && card) { e.preventDefault(); if (card._open) card._open(); }
    else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && card) { e.preventDefault(); if (card._more) card._more(e.key === "ArrowRight"); }
    else if (e.key.toLowerCase() === "p" && card && !mod) { e.preventDefault(); if (card._pin) card._pin(); }
    else if (e.key.length === 1 && !mod) { focusSearch(); }   // start a new search: the character lands in the field
  });
  results.addEventListener("focusin", e => { const card = e.target.closest(".rule"); if (card) { const i = ruleCards().indexOf(card); if (i >= 0) { selected = i; ruleCards().forEach((c, k) => c.classList.toggle("selected", k === i)); } } });

  async function runSearch(q, opts = {}) {
    q = (q || "").trim();
    if (!q || !ready) return;
    const mySeq = ++seq;              // newer searches always win
    if (inflight) inflight.abort();   // an older in-flight request is cancelled, never rendered
    const ctrl = new AbortController();
    inflight = ctrl;
    lastSubmitted = q;
    syncEmpty();
    const timer = setTimeout(() => ctrl.abort(), SEARCH_TIMEOUT_MS);
    const t0 = performance.now();
    meta.hidden = false;
    meta.textContent = "Searching…";
    history.replaceState(null, "", "#" + encodeURIComponent(q) + "&scope=" + scope());   // no history entry per keystroke
    let data, status;
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&scope=${scope()}`, { signal: ctrl.signal, cache: "no-store" });
      status = r.status;
      data = await r.json();
    } catch (e) {
      if (mySeq !== seq) return;      // superseded; ignore silently
      data = { type: "error", message: e.name === "AbortError" ? `The search did not complete within ${SEARCH_TIMEOUT_MS / 1000} s and was abandoned. This counts as a failed search; try again or simplify the query.` : "Network error: " + e.message };
    } finally {
      clearTimeout(timer);
      if (inflight === ctrl) {
        inflight = null;
        if (livePending) { livePending = false; fireLive(); }   // send the text typed meanwhile, once
      }
    }
    if (mySeq !== seq) return;        // an older response must never overwrite a newer search
    const clientMs = performance.now() - t0;
    render(data, q, clientMs, status);
  }

  function maybeSearchFromHash() {
    if (!location.hash || location.hash.length < 2) return;
    const h = decodeURIComponent(location.hash.slice(1));
    const [q, sc] = h.split("&scope=");
    if (sc) setScope(sc, { search: false });
    input.value = q; syncEmpty();
    runSearch(q);
  }

  // ---------- rendering ----------
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  // ---------- inline rule drop-downs ----------
  // Every rule code that appears anywhere (inside rule text, in the referenced-rules list, in notes) opens in place,
  // directly under the line it is on. The result list, the query and the URL never change; nothing is a "new page".
  const CODE_RE = /\b(A|T|CV|EV|IN|S|D)\s?\d{1,2}(?:\.\d{1,3}){0,2}\b/g;
  const codeKey = c => c.replace(/\s+/g, "").toUpperCase();
  const refKeysOf = r => { const s = new Set((r.references || []).map(x => codeKey(x.code))); s.delete(codeKey(r.code)); return s; };
  const ruleCache = new Map();
  function fetchRule(code) {
    const key = codeKey(code);
    if (!ruleCache.has(key)) {
      ruleCache.set(key, (async () => {
        const r = await fetch(`/api/rule/${encodeURIComponent(code)}`, { cache: "no-store" });
        if (!r.ok) throw new Error(r.status === 404 ? `${code} is not a rule in this rulebook.` : "Could not load " + code + " (HTTP " + r.status + ")");
        const rule = await r.json();
        if (rule.kind !== "rule") {          // a section or chapter heading: show the rules under it
          const s = await fetch(`/api/section/${encodeURIComponent(code)}`, { cache: "no-store" });
          rule.section_rules_full = s.ok ? (await s.json()).rules : [];
        }
        return rule;
      })().catch(e => { ruleCache.delete(key); throw e; }));
    }
    return ruleCache.get(key);
  }
  function codeLink(code, label) {
    const b = el("button", "codelink", label == null ? code : label); b.type = "button";
    b.title = "Show " + code + " here"; b.setAttribute("aria-expanded", "false");
    b.onclick = () => toggleDrop(b, code);
    return b;
  }
  function toggleDrop(btn, code) {
    if (btn._drop) { btn._drop.remove(); btn._drop = null; btn.setAttribute("aria-expanded", "false"); return; }
    const drop = el("div", "drop", "Loading " + code + "…");
    const host = btn.closest("li, p, .droprow, .note, .notice") || btn.parentElement;
    if (host.tagName === "LI") host.appendChild(drop); else host.after(drop);   // sits right under the line with the link
    btn._drop = drop; btn.setAttribute("aria-expanded", "true");
    fetchRule(code).then(rule => { if (btn._drop !== drop) return; drop.innerHTML = ""; drop.appendChild(renderDrop(rule, () => toggleDrop(btn, code))); })
      .catch(e => { drop.textContent = e.message; drop.classList.add("err"); });
  }
  function dropHead(rule, extra, onClose) {
    const h = el("div", "drophead");
    const a = el("a", "code", rule.code); a.href = `/pdf#page=${rule.pages[0]}`; a.title = `Open the PDF at page ${rule.pages[0]}`;
    a.onclick = ev => { ev.preventDefault(); showPdf(rule.pages[0], rule.code); };
    h.appendChild(a);
    const title = rule.kind === "rule" ? (rule.section_title || "") : (rule.title || "");
    if (title) h.appendChild(el("span", "note", title));
    if (rule.kind === "rule" && rule.applicability) badgeFor(rule.applicability).slice(0, 1).forEach(b => h.appendChild(el("span", "badge " + b.cls, b.text)));
    h.appendChild(el("span", "note", (extra ? extra + " · " : "") + "page " + rule.pages.join(", ")));
    if (onClose) { const x = el("button", "x", "×"); x.type = "button"; x.title = "Close " + rule.code; x.setAttribute("aria-label", "Close " + rule.code); x.onclick = onClose; h.appendChild(x); }
    return h;
  }
  function renderDrop(rule, onClose) {
    const d = el("div");
    if (rule.kind !== "rule") {
      const rules = rule.section_rules_full || [];
      d.appendChild(dropHead(rule, `${rule.kind} · ${rules.length} rule${rules.length === 1 ? "" : "s"}`, onClose));
      rules.forEach(s => { const sub = el("div", "sub"); sub.appendChild(dropHead(s)); renderBlocks(sub, s.blocks, refKeysOf(s)); d.appendChild(sub); });
      return d;
    }
    d.appendChild(dropHead(rule, null, onClose));
    renderBlocks(d, rule.blocks, refKeysOf(rule));
    return d;
  }
  // text with highlighted query words, where every code the rule refers to becomes an inline drop-down link
  function appendText(parent, text, refKeys) {
    if (!refKeys || !refKeys.size) { appendHighlighted(parent, text); return; }
    let last = 0, m; CODE_RE.lastIndex = 0;
    while ((m = CODE_RE.exec(text))) {
      if (!refKeys.has(codeKey(m[0]))) continue;
      if (m.index > last) appendHighlighted(parent, text.slice(last, m.index));
      parent.appendChild(codeLink(m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) appendHighlighted(parent, text.slice(last));
  }

  // ---------- query-term highlighting ----------
  // Words of the (corrected) query, plus the synonyms / glossary expansions that were searched, are marked inside the rule text.
  // Matching is stem-tolerant on the client ("lights" marks "light", "brake" marks "braking") so it mirrors the stemmed BM25 index.
  let hlTerms = [];
  const STOP = new Set(("the a an and or of to in on for with is are be must not at by as it this that from what which how when where " +
    "does do can any all have has if than then there their its into per vs about between each while shall should may will").split(" "));
  function crude(w) { w = w.toLowerCase(); for (const suf of ["ing", "ies", "ied", "es", "ed", "s", "ly"]) { if (w.length - suf.length >= 3 && w.endsWith(suf)) return w.slice(0, -suf.length); } return w; }
  function termMatches(word, term) {
    const a = crude(word), b = crude(term);
    if (a === b) return true;
    const [s, l] = a.length < b.length ? [a, b] : [b, a];
    return s.length >= 4 && l.length - s.length <= 2 && l.startsWith(s);   // 'brak' ~ 'brake', but not 'fire' ~ 'firewall'
  }
  function setHighlight(data, q) {
    hlTerms = [];
    if (data.mode !== "hybrid") return;                 // rule-code lookups have nothing to highlight
    const words = new Set();
    const add = s => (s || "").split(/[^\p{L}\p{N}-]+/u).forEach(w => {
      const l = w.toLowerCase().replace(/^-+|-+$/g, "");
      if (l.length >= 2 && !STOP.has(l) && !/^\d+$/.test(l)) words.add(l.replace(/-/g, ""));   // "pre-charge" and "precharge" are the same term
      if (l.includes("-")) l.split("-").forEach(part => { if (part.length >= 3 && !STOP.has(part)) words.add(part); });
    });
    add(data.corrected_query || q);
    (data.synonyms || []).forEach(add);
    (data.expansions || []).forEach(e => add(e[1]));
    hlTerms = Array.from(words);
  }
  function appendHighlighted(parent, text) {
    if (!hlTerms.length) { parent.appendChild(document.createTextNode(text)); return; }
    // a hyphenated compound ("pre-charge") is one unit: it matches as a whole ("precharge") or through any of its parts ("charge")
    const re = /[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu; let last = 0, m;
    const hit = w => hlTerms.some(t => termMatches(w, t) || (w.includes("-") && (termMatches(w.replace(/-/g, ""), t) || w.split("-").some(part => part.length >= 3 && termMatches(part, t)))));
    while ((m = re.exec(text))) {
      if (!hit(m[0])) continue;
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      parent.appendChild(el("mark", "hl", m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  function render(data, q, clientMs, status) {
    results.innerHTML = ""; selected = -1;
    setHighlight(data, q);
    if (pdfShown) markViewing(null);
    if (data.type === "results" || data.type === "code" || data.type === "clarification") noteRecent(q);
    const serverMs = data.total_ms != null ? data.total_ms : null;
    const timeTxt = `${((serverMs != null ? serverMs : clientMs) / 1000).toFixed(2)} s`;
    meta.innerHTML = "";
    meta.appendChild(el("span", "", timeTxt));
    if (data.corrected_query) meta.appendChild(el("span", "", `Spelling adjusted: searched for “${data.corrected_query}”`));
    if (data.expansions && data.expansions.length) meta.appendChild(el("span", "", "Abbreviations: " + data.expansions.map(e => `${e[0]} = ${e[1]}`).join("; ")));
    if (data.synonyms && data.synonyms.length) meta.appendChild(el("span", "", "Also searched: " + data.synonyms.join(", ")));
    if (data.mode === "code") meta.appendChild(el("span", "", "Exact rule-code lookup (no ranking involved)"));

    if (data.type === "error") {
      const n = el("div", "notice err", data.message || "Search failed.");
      results.appendChild(n);
      return;
    }
    (data.degraded || []).forEach(d => results.appendChild(el("div", "notice err", "Reduced functionality: " + d)));

    if (data.type === "no_match") {
      const n = el("div", "notice");
      n.appendChild(el("strong", "", data.mode === "code" ? "No such rule code. " : "No reliable match found. "));
      n.appendChild(document.createTextNode(data.message && data.mode === "code" ? data.message : "No rule in this rulebook matches your query well enough to show it as an answer."));
      if (data.tips && data.tips.length) { const ul = el("ul"); data.tips.forEach(t => ul.appendChild(el("li", "", t))); n.appendChild(ul); }
      if (data.suggestions && data.suggestions.length) {
        n.appendChild(el("div", "", "Existing codes under the same heading (not an answer to your query):"));
        const sg = el("div", "suggest");
        data.suggestions.forEach(c => sg.appendChild(codeLink(c)));
        n.appendChild(sg);
      }
      if (data.near && data.near.length && data.mode !== "code") {
        n.appendChild(el("div", "note", "Closest (weak) candidates, shown only so you can check them yourself: " + data.near.join(", ")));
        const sg = el("div", "suggest");
        data.near.forEach(c => sg.appendChild(codeLink(c)));
        n.appendChild(sg);
      }
      results.appendChild(n);
      return;
    }

    if (data.type === "clarification" && data.clarification) {
      const c = data.clarification;
      const n = el("div", "notice info");
      n.appendChild(el("strong", "", "Clarification needed: "));
      n.appendChild(document.createTextNode(c.question + " "));
      const sg = el("div", "suggest");
      c.options.forEach(o => {
        const b = el("button", "optbtn", o.label); b.type = "button";
        b.onclick = () => { if (o.scope) setScope(o.scope, { search: false }); input.value = o.code; runSearch(o.code); };
        sg.appendChild(b);
      });
      n.appendChild(sg);
      n.appendChild(el("div", "note", "Both candidates are shown below so you can compare the original wording."));
      results.appendChild(n);
    }

    if (data.weak && data.type === "results") {
      results.appendChild(el("div", "notice", "Weak match: the rules below are the closest the search found, but they may not answer your question. Check the wording, or refine the query (e.g. use the rulebook's own term or a rule code)."));
    }
    if (data.heading) {
      const h = el("div", "heading-block");
      h.appendChild(el("h3", "", `${data.heading.code} ${data.heading.title}`));
      h.appendChild(el("div", "note", `${data.heading.kind} heading · page ${data.heading.pages[0]} · ${data.results.length} rule(s) below`));
      results.appendChild(h);
    }
    (data.results || []).forEach((r, i) => results.appendChild(renderRule(r, i === 0 && data.mode === "hybrid")));
    if (data.mentioned_codes && data.mentioned_codes.length) {
      const n = el("div", "note", "Your query mentions: ");
      data.mentioned_codes.forEach(c => { n.appendChild(codeLink(c)); n.appendChild(document.createTextNode(" ")); });
      results.appendChild(n);
    }
  }

  function badgeFor(app) {
    const out = [];
    const scopeCls = app.scope === "EV" ? "ev" : app.scope === "CV" ? "cv" : "gen";
    out.push({ cls: scopeCls, text: app.scope === "general" ? "All vehicles" : `${app.scope} only` });
    (app.tags || []).forEach(t => { if (t !== "EV" && t !== "CV") out.push({ cls: "other", text: `${t} only` }); });
    const mixed = (app.provision_tags || []).filter(t => t === "EV" || t === "CV");
    if (app.scope === "general" && mixed.length) out.push({ cls: "mixed", text: "contains " + mixed.map(t => t + "-only").join(" & ") + " provisions" });
    return out;
  }

  function textWithTags(li, text, refKeys) {
    // keep original text; add a coloured chip in front of [EV ONLY] / [CV ONLY] provisions
    const m = text.match(/^\s*\[(EV|CV|DC|HY|Autonomous Mode|Manual Mode) ONLY\]/);
    if (m) {
      const chip = el("span", "tagchip " + (m[1] === "EV" ? "ev" : m[1] === "CV" ? "cv" : ""), m[1] + " only");
      li.appendChild(chip);
      li.classList.add("tag-" + m[1].toLowerCase());
    }
    appendText(li, text, refKeys);
  }

  function renderBlocks(container, blocks, refKeys) {
    let listStack = []; // [{level, ul}]
    const closeLists = (toLevel) => { while (listStack.length && listStack[listStack.length - 1].level > toLevel) listStack.pop(); };
    blocks.forEach(b => {
      if (b.type === "li") {
        closeLists(b.level);
        if (!listStack.length || listStack[listStack.length - 1].level < b.level) {
          const ul = document.createElement("ul");
          ul.style.listStyleType = b.marker === "▪" ? "square" : "none";
          if (listStack.length) {
            const parentUl = listStack[listStack.length - 1].ul;
            (parentUl.lastElementChild || parentUl).appendChild(ul);
          } else container.appendChild(ul);
          listStack.push({ level: b.level, ul });
        }
        const li = document.createElement("li");
        if (b.marker !== "▪") li.appendChild(document.createTextNode(b.marker + " "));
        textWithTags(li, b.text, refKeys);
        listStack[listStack.length - 1].ul.appendChild(li);
        return;
      }
      closeLists(0);
      if (b.type === "p") { const p = el("p"); textWithTags(p, b.text, refKeys); container.appendChild(p); }
      else if (b.type === "note") container.appendChild(el("p", "note", b.text));
      else if (b.type === "table" || b.type === "figure" || b.type === "formula") container.appendChild(renderAsset(b.asset_data, b.type, b.text));
    });
  }

  function renderAsset(a, kind, formulaText) {
    const box = el("div", "asset");
    if (!a) { box.textContent = `[${kind} could not be loaded]`; return box; }
    const cap = el("div", "cap");
    const label = a.caption || (kind === "table" ? "Table (no caption in the PDF)" : kind === "formula" ? "Formula" : "Figure");
    cap.appendChild(el("span", "", `${label} · page ${a.pages && a.pages.length > 1 ? a.pages.join("–") : a.page}`));
    box.appendChild(cap);
    if (kind === "table" && a.rows && a.rows.length) {
      const table = document.createElement("table");
      a.rows.forEach((row, ri) => {
        const tr = document.createElement("tr");
        row.forEach(c => { const cell = document.createElement(ri === 0 ? "th" : "td"); cell.textContent = c; tr.appendChild(cell); });
        table.appendChild(tr);
      });
      box.appendChild(table);
      const tog = el("button", "linkbtn", "Show original PDF image"); tog.type = "button";
      const imgwrap = el("div"); imgwrap.hidden = true;
      imgwrap.appendChild(imgEl(a.image, label));
      (a.extra_images || []).forEach(x => imgwrap.appendChild(imgEl(x.image, label + " (continued on page " + x.page + ")")));
      tog.onclick = () => { imgwrap.hidden = !imgwrap.hidden; tog.textContent = imgwrap.hidden ? "Show original PDF image" : "Hide original PDF image"; };
      cap.appendChild(tog);
      box.appendChild(imgwrap);
      box.appendChild(el("div", "note", "Table text is extracted from the PDF cell by cell; use the original image to double-check layout."));
    } else {
      box.appendChild(imgEl(a.image, label));
      (a.extra_images || []).forEach(x => box.appendChild(imgEl(x.image, label + " (continued)")));
      if (kind === "formula") box.appendChild(el("div", "note", "Formula shown as the original PDF image; the flattened text is only used for search."));
    }
    return box;
  }
  function imgEl(src, alt) { const img = document.createElement("img"); img.src = "/" + src; img.alt = alt; img.loading = "lazy"; return img; }

  function renderRule(r, isTop) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    if (isTop) node.classList.add("top");
    if (r.why && r.why.referenced_by) node.classList.add("referenced");
    const codeLink = $(".code", node);
    codeLink.textContent = r.code;
    codeLink.href = `/pdf#page=${r.pages[0]}`;
    codeLink.title = `Open the PDF at page ${r.pages[0]}`;
    codeLink.onclick = ev => { ev.preventDefault(); showPdf(r.pages[0], r.code); };
    node._open = () => showPdf(r.pages[0], r.code);
    const pinBtn = $(".pin", node);
    node._syncPin = () => { const on = isPinned(r.code); pinBtn.setAttribute("aria-pressed", String(on)); pinBtn.title = (on ? "Unpin rule" : "Pin rule") + " (P)"; };
    node._pin = () => togglePin(r);
    pinBtn.onclick = node._pin;
    node._syncPin();
    const pathParts = [];
    if (r.chapter) pathParts.push(`${r.chapter} ${r.chapter_title}`);
    if (r.section) pathParts.push(`${r.section} ${r.section_title}`);
    appendHighlighted($(".path", node), pathParts.join(" › "));
    const badges = $(".badges", node);
    badgeFor(r.applicability).forEach(b => badges.appendChild(el("span", "badge " + b.cls, b.text)));
    renderBlocks($(".body", node), r.blocks, refKeysOf(r));
    if (r.changelog && r.changelog.length) {
      $(".body", node).appendChild(el("p", "note", "Changelog (version " + "1.0" + "): " + r.changelog.join(" · ")));
    }
    const pages = $(".pages", node);
    r.pages.forEach(p => {
      const a = el("a", "pill page", "Page " + p); a.href = `/pdf#page=${p}`; a.title = `Show page ${p} of the rulebook PDF`; a.dataset.page = String(p);
      a.onclick = (ev) => { ev.preventDefault(); showPdf(p, r.code); node.focus({ preventScroll: true }); };   // the card becomes the selected one
      pages.appendChild(a);
    });
    // one "Show more" reveals everything about the rule: its section, the rules it refers to, its figures / tables
    const extra = $(".extra", node), moreBtn = $(".more", node);
    const parts = [];
    if (r.references && r.references.length) parts.push(`${r.references.length} referenced rule${r.references.length > 1 ? "s" : ""}`);
    if (r.asset_refs && r.asset_refs.length) parts.push(r.asset_refs.map(a => `${a.kind === "table" ? "Table" : "Figure"} ${a.number}`).join(", "));
    moreBtn.title = "Section " + (r.section || "") + (parts.length ? " · " + parts.join(" · ") : "") + "  (→ / ←)";
    node._more = (open) => toggleMore(node, r, extra, moreBtn, open);
    moreBtn.onclick = () => node._more();
    if (r.why) {
      const bits = [];
      if (r.why.referenced_by) bits.push(`included because ${r.why.referenced_by} refers to it`);
      if (r.why.matched_terms && r.why.matched_terms.length) bits.push("matched: " + r.why.matched_terms.join(", "));
      $(".why", node).textContent = bits.join(" · ");
    }
    return node;
  }

  function toggleMore(node, r, extra, moreBtn, open) {
    const isOpen = !extra.hidden;
    if (open === undefined) open = !isOpen;
    if (open === isOpen) return;
    const label = $("span", moreBtn);
    if (!open) {
      extra.hidden = true; extra.innerHTML = "";
      moreBtn.setAttribute("aria-expanded", "false"); label.textContent = "Show more";
      node.focus({ preventScroll: true }); node.scrollIntoView({ block: "nearest" });   // back at the rule, ready for the next one
      return;
    }
    extra.hidden = false; extra.textContent = "Loading…";
    moreBtn.setAttribute("aria-expanded", "true"); label.textContent = "Show less";
    const build = async () => {
      const d = el("div");
      if (r.references && r.references.length) d.appendChild(renderRefs(r));
      if (r.asset_refs && r.asset_refs.length) { const f = el("div"); f.appendChild(el("h4", "", "Referenced figures and tables")); r.asset_refs.forEach(a => f.appendChild(renderAsset(a, a.kind))); d.appendChild(f); }
      d.appendChild(await renderContext(r));
      const less = el("button", "pill less"); less.type = "button"; less.innerHTML = moreBtn.innerHTML; $("span", less).textContent = "Show less"; less.querySelector("svg").style.transform = "rotate(180deg)";
      less.onclick = () => node._more(false);
      d.appendChild(less);
      return d;
    };
    build().then(d => { if (!extra.hidden) { extra.innerHTML = ""; extra.appendChild(d); } })
      .catch(e => { extra.textContent = "Could not load: " + e.message; });
  }

  async function renderContext(r) {
    const d = el("div");
    d.appendChild(el("h4", "", `Section ${r.section} ${r.section_title} — all rules`));
    const resp = await fetch(`/api/section/${encodeURIComponent(r.section)}`, { cache: "no-store" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const sec = await resp.json();
    sec.rules.forEach(s => {
      const sub = el("div", "sub" + (s.code === r.code ? " current" : ""));
      const head = el("div", "codes"); head.appendChild(el("span", "code", s.code)); head.appendChild(el("span", "note", ` page ${s.pages.join(", ")}`)); sub.appendChild(head);
      renderBlocks(sub, s.blocks, refKeysOf(s));
      d.appendChild(sub);
    });
    const nav = el("div", "note");
    const inSection = new Set(sec.rules.map(s => codeKey(s.code)));
    if (r.context && r.context.prev && !inSection.has(codeKey(r.context.prev))) { nav.appendChild(document.createTextNode("Previous rule: ")); nav.appendChild(codeLink(r.context.prev)); nav.appendChild(document.createTextNode("  ")); }
    if (r.context && r.context.next && !inSection.has(codeKey(r.context.next))) { nav.appendChild(document.createTextNode("Next rule: ")); nav.appendChild(codeLink(r.context.next)); }
    if (nav.childNodes.length) d.appendChild(nav);
    return d;
  }

  function renderRefs(r) {
    const d = el("div");
    d.appendChild(el("h4", "", "Rules referenced in " + r.code));
    const links = [];
    r.references.forEach(ref => {
      const row = el("div", "droprow");
      const link = codeLink(ref.code); link.classList.add("strong"); row.appendChild(link); links.push(link);
      row.appendChild(el("span", "note", `${ref.kind === "rule" ? ref.section_title : ref.title} · page ${ref.pages.join(", ")}`));
      d.appendChild(row);
    });
    if (links.length <= 3) links.forEach(l => l.click());   // a few references: open them straight away
    return d;
  }

  // ---------- PDF pane ----------
  // The pane slides in on a spring and the card being shown is marked, so the PDF reads as part of the result, not a separate page.
  const pdfToggle = $("#pdf-toggle"), pdfToggleText = $("#pdf-toggle span");
  let pdfShown = false;
  const paneSpring = spring(0, {
    response: 0.32, damping: 1,
    onUpdate: p => { pdfpane.style.opacity = String(Math.max(0, Math.min(1, p))); pdfpane.style.transform = `translateX(${(1 - p) * 28}px)`; },
    onRest: () => { if (!pdfShown) { pdfpane.hidden = true; document.body.classList.remove("pdf-open"); } }
  });
  function markViewing(code, page) {
    results.querySelectorAll(".rule.viewing").forEach(c => c.classList.remove("viewing"));
    results.querySelectorAll(".pill.page.on").forEach(a => a.classList.remove("on"));
    if (!code) return;
    const card = ruleCards().find(c => $(".code", c).textContent === code);
    if (card) { card.classList.add("viewing"); const a = card.querySelector(`.pill.page[data-page="${page}"]`); if (a) a.classList.add("on"); }
  }
  function showPdf(page, code) {
    pdfLabel.textContent = code ? `${code} · page ${page}` : `Page ${page}`;
    pdfOpen.href = `/pdf#page=${page}`;
    const src = `/pdf#page=${page}&zoom=page-width&navpanes=0`;   // navpanes=0: no thumbnail sidebar, the page gets the width
    if (pdfframe.getAttribute("src") !== src) pdfframe.src = src;
    markViewing(code, page);
    pdfToggleText.textContent = "Hide PDF"; pdfToggle.classList.add("on");
    if (!pdfShown) { pdfShown = true; pdfpane.hidden = false; document.body.classList.add("pdf-open"); paneSpring.to(1); }
  }
  function hidePdf() {
    if (!pdfShown) return;
    pdfShown = false; markViewing(null);
    pdfToggleText.textContent = "Show PDF"; pdfToggle.classList.remove("on");
    paneSpring.to(0);
  }
  pdfToggle.onclick = () => { if (pdfShown) hidePdf(); else showPdf(1, ""); };
  $("#pdf-close").onclick = hidePdf;
})();
