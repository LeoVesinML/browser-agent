/**
 * Page perception layer.
 *
 * Injected into every frame. Produces a compact, semantic description of what a
 * human would perceive on the page, plus a registry that maps opaque refs
 * ("e17") back to live DOM elements so the agent can act on them without ever
 * writing a CSS selector.
 *
 * Nothing here is site-specific: roles and names are derived from HTML/ARIA
 * semantics only.
 */
(() => {
  if (window.__agent && window.__agent.version === 3) return;

  const SKIP_TAGS = new Set([
    'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'HEAD', 'META', 'LINK', 'TITLE',
    'BASE', 'PATH', 'DEFS', 'CLIPPATH', 'G', 'CIRCLE', 'RECT', 'POLYGON',
  ]);

  const ROLE_BY_TAG = {
    A: 'link', BUTTON: 'button', SELECT: 'combobox', TEXTAREA: 'textbox',
    IMG: 'image', H1: 'heading', H2: 'heading', H3: 'heading', H4: 'heading',
    H5: 'heading', H6: 'heading', LI: 'listitem', UL: 'list', OL: 'list',
    TABLE: 'table', TR: 'row', TD: 'cell', TH: 'columnheader', FORM: 'form',
    NAV: 'navigation', MAIN: 'main', HEADER: 'banner', FOOTER: 'contentinfo',
    ASIDE: 'complementary', DIALOG: 'dialog', OPTION: 'option', LABEL: 'label',
    SUMMARY: 'disclosure', IFRAME: 'iframe', VIDEO: 'video', AUDIO: 'audio',
    CANVAS: 'canvas', P: 'paragraph', ARTICLE: 'article', SECTION: 'section',
  };

  const INPUT_ROLE = {
    button: 'button', submit: 'button', reset: 'button', image: 'button',
    checkbox: 'checkbox', radio: 'radio', range: 'slider', file: 'filebutton',
    hidden: null,
  };

  // Roles the agent can act on. Everything else is context.
  const INTERACTIVE = new Set([
    'link', 'button', 'checkbox', 'radio', 'textbox', 'combobox', 'searchbox',
    'slider', 'filebutton', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
    'tab', 'option', 'switch', 'spinbutton', 'disclosure', 'treeitem',
  ]);

  // Roles that carry page structure worth keeping for orientation.
  const LANDMARK = new Set([
    'navigation', 'main', 'banner', 'contentinfo', 'complementary', 'form',
    'dialog', 'alertdialog', 'list', 'listitem', 'table', 'row', 'article',
    'search', 'tablist', 'menu', 'alert', 'status', 'heading', 'iframe',
  ]);

  // Containers whose name must come from ARIA only: using their innerText would
  // duplicate every descendant's text and blow up the snapshot.
  const CONTAINER = new Set([
    'navigation', 'main', 'banner', 'contentinfo', 'complementary', 'form',
    'list', 'table', 'row', 'article', 'section', 'dialog', 'alertdialog',
    'search', 'tablist', 'menu', 'generic-container', 'iframe',
  ]);

  // Structure worth one token even when unnamed: it tells the model where it is.
  const ALWAYS_SHOW = new Set([
    'navigation', 'main', 'banner', 'contentinfo', 'form', 'dialog',
    'alertdialog', 'table', 'tablist', 'iframe', 'search',
  ]);

  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

  const clip = (s, n) => {
    s = norm(s);
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  };

  function roleOf(el) {
    const explicit = norm(el.getAttribute && el.getAttribute('role'));
    if (explicit) return explicit.split(' ')[0];
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t in INPUT_ROLE) return INPUT_ROLE[t];
      if (t === 'search') return 'searchbox';
      return 'textbox';
    }
    if (tag === 'A') return el.hasAttribute('href') ? 'link' : 'generic';
    if (el.isContentEditable) return 'textbox';
    return ROLE_BY_TAG[tag] || 'generic';
  }

  function ownText(el) {
    // Text that belongs to this element directly, not to its element children.
    let out = '';
    for (const n of el.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) out += n.nodeValue;
    }
    return norm(out);
  }

  function labelText(el) {
    const id = el.getAttribute && el.getAttribute('id');
    if (id) {
      try {
        const lbl = el.getRootNode().querySelector(`label[for="${CSS.escape(id)}"]`);
        if (lbl) return norm(lbl.innerText || lbl.textContent);
      } catch (e) { /* malformed id */ }
    }
    const wrapper = el.closest && el.closest('label');
    if (wrapper) return norm(wrapper.innerText || wrapper.textContent);
    return '';
  }

  function accessibleName(el, role) {
    const at = (n) => norm(el.getAttribute && el.getAttribute(n));
    let name = at('aria-label');
    if (!name) {
      const labelledBy = at('aria-labelledby');
      if (labelledBy) {
        name = labelledBy.split(/\s+/)
          .map((id) => {
            try { return el.getRootNode().getElementById(id); } catch (e) { return null; }
          })
          .filter(Boolean)
          .map((n) => norm(n.innerText || n.textContent))
          .join(' ');
      }
    }
    if (!name && el.tagName === 'IMG') name = at('alt');
    if (!name && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) {
      name = labelText(el) || at('placeholder') || at('name');
      if (!name && el.type && ['button', 'submit', 'reset'].includes(el.type)) name = at('value');
    }
    if (!name) name = at('title');
    if (!name && CONTAINER.has(role)) return '';
    if (!name) {
      // innerText is expensive but only needed for compact, leaf-ish nodes.
      const raw = el.innerText !== undefined ? el.innerText : el.textContent;
      // Keep a visible separator so "sender / subject / time" does not read as one phrase.
      const txt = (raw || '').replace(/\n+/g, ' · ');
      const n = norm(txt);
      if (n && n.length <= 200) name = n;
      else name = ownText(el);
    }
    if (!name && role === 'link') {
      const href = at('href');
      if (href) name = clip(href.replace(/^https?:\/\//, ''), 60);
    }
    return clip(name, 160);
  }

  function isHidden(el, style) {
    if (el.getAttribute && el.getAttribute('aria-hidden') === 'true') return true;
    if (el.hasAttribute && el.hasAttribute('inert')) return true;
    if (!style) return false;
    if (style.display === 'none' || style.visibility === 'hidden') return true;
    if (parseFloat(style.opacity || '1') === 0) return true;
    return false;
  }

  function stateOf(el, role) {
    const st = {};
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') st.disabled = true;
    if (role === 'checkbox' || role === 'radio' || role === 'switch') {
      st.checked = el.checked !== undefined
        ? !!el.checked
        : el.getAttribute('aria-checked') === 'true';
    }
    const expanded = el.getAttribute('aria-expanded');
    if (expanded !== null) st.expanded = expanded === 'true';
    const selected = el.getAttribute('aria-selected');
    if (selected !== null) st.selected = selected === 'true';
    if (el.getAttribute('aria-current')) st.current = true;
    if (el.required) st.required = true;
    if (role === 'textbox' || role === 'searchbox' || role === 'spinbutton') {
      const v = el.isContentEditable ? norm(el.innerText) : el.value;
      if (v) st.value = clip(v, 80);
    }
    if (role === 'combobox' && el.selectedOptions) {
      const v = Array.from(el.selectedOptions).map((o) => norm(o.textContent)).join(', ');
      if (v) st.value = clip(v, 80);
      st.options = Array.from(el.options || []).slice(0, 25).map((o) => clip(o.textContent || o.value, 40));
    }
    if (el.tagName === 'INPUT') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t !== 'text') st.inputType = t;
      // Never surface the contents of a password field to the model.
      if (t === 'password') delete st.value;
    }
    if (el.tagName === 'H1') st.level = 1;
    else if (/^H[1-6]$/.test(el.tagName)) st.level = +el.tagName[1];
    return st;
  }

  function isScrollable(el, style) {
    if (!style) return false;
    const oy = style.overflowY;
    return (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 40;
  }

  /** Topmost open modal/dialog, if any: popups must dominate the snapshot. */
  function findModal(root) {
    const candidates = Array.from(
      root.querySelectorAll('dialog[open], [role="dialog"], [role="alertdialog"], [aria-modal="true"]')
    ).filter((el) => {
      const r = el.getBoundingClientRect();
      const s = getComputedStyle(el);
      return r.width > 40 && r.height > 40 && !isHidden(el, s);
    });
    return candidates.length ? candidates[candidates.length - 1] : null;
  }

  function collect(opts) {
    const cfg = Object.assign(
      { maxNodes: 600, scope: 'viewport', viewportMargin: 200, interactiveOnly: false },
      opts || {}
    );
    // Refs accumulate for the lifetime of the document and are never reused.
    // A ref from an older snapshot therefore either still points at its original
    // element or resolves to a detached node we can report as stale - it never
    // silently aliases a *different* element, which would be far worse.
    const refs = window.__agent.refs;
    if (refs.size > 3000) {
      for (const [k, v] of refs) if (!v.isConnected) refs.delete(k);
    }
    let refSeq = Math.max(window.__agent.seq || 0, cfg.refStart || 0);
    const nodes = [];
    let omitted = 0;

    const vh = window.innerHeight || 0;
    const vw = window.innerWidth || 0;

    const modal = findModal(document);
    const root = modal || document.body || document.documentElement;

    function visit(el, depth, insidePointer) {
      if (nodes.length >= cfg.maxNodes) { omitted += 1; return; }
      if (!el || el.nodeType !== Node.ELEMENT_NODE) return;
      if (SKIP_TAGS.has(el.tagName)) return;

      let style = null;
      try { style = getComputedStyle(el); } catch (e) { /* detached */ }
      if (isHidden(el, style)) return;

      const rect = el.getBoundingClientRect();
      const hasBox = rect.width > 0 || rect.height > 0;
      const inView = hasBox
        && rect.bottom > -cfg.viewportMargin && rect.top < vh + cfg.viewportMargin
        && rect.right > -cfg.viewportMargin && rect.left < vw + cfg.viewportMargin;

      const role = roleOf(el);
      const nativelyInteractive = INTERACTIVE.has(role)
        || (el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1' && el.tagName !== 'BODY')
        || !!el.getAttribute('onclick');
      // Sites routinely make plain <div>s clickable with a delegated listener; the
      // only thing a human sees is the pointer cursor, so we trust that too - but
      // only for the outermost such element, or every wrapper would get a ref.
      const pointer = !!style && style.cursor === 'pointer';
      const clickable = nativelyInteractive || (pointer && !insidePointer);
      const interactive = clickable;
      const scrollable = isScrollable(el, style);
      const text = ownText(el);
      const landmark = LANDMARK.has(role);
      const keep = hasBox && (interactive || scrollable || landmark || (text && text.length > 1)
        || (role === 'image' && el.getAttribute('alt')));

      let emitted = false;
      let nameFromSubtree = false;
      if (keep && (cfg.scope !== 'viewport' || inView) && !(cfg.interactiveOnly && !interactive)) {
        let name;
        if (interactive || landmark || role === 'image') {
          name = accessibleName(el, role);
          nameFromSubtree = !!name && !text;
        } else {
          name = clip(text, 160);
        }
        if (name || interactive || scrollable || ALWAYS_SHOW.has(role)) {
          const node = { role, name, depth, tag: el.tagName.toLowerCase() };
          if (pointer && !nativelyInteractive) node.clickable = true;
          if (!inView) node.offscreen = true;
          const st = stateOf(el, role);
          if (Object.keys(st).length) node.state = st;
          if (interactive || scrollable) {
            // Stable identity: the same element keeps the same ref across
            // snapshots, which is what makes snapshot-to-snapshot diffs small.
            let ref = window.__agent.byEl.get(el);
            if (!ref) {
              ref = 'e' + (++refSeq);
              refs.set(ref, el);
              window.__agent.byEl.set(el, ref);
            }
            node.ref = ref;
            if (scrollable) node.scrollable = true;
          }
          nodes.push(node);
          emitted = true;
        }
      } else if (keep) {
        omitted += 1;
      }

      if (el.tagName === 'SELECT') return;

      // Collapse: if this node's name already reproduces the subtree's text and the
      // subtree holds no other control, descending would only duplicate tokens.
      if (emitted && (nameFromSubtree || interactive) && !el.querySelector(
        'a,button,input,select,textarea,[role],[onclick],[tabindex],[contenteditable]')) {
        return;
      }

      const childDepth = emitted ? depth + 1 : depth;
      const kids = el.shadowRoot ? [...el.shadowRoot.children, ...el.children] : el.children;
      for (const child of kids) visit(child, childDepth + 0, insidePointer || pointer);
    }

    visit(root, 0, false);

    window.__agent.epoch = (window.__agent.epoch || 0) + 1;
    window.__agent.seq = refSeq;

    return {
      url: location.href,
      title: document.title,
      epoch: window.__agent.epoch,
      modal: !!modal,
      scroll: { y: Math.round(window.scrollY), max: Math.max(0, document.body.scrollHeight - vh) },
      viewport: { w: vw, h: vh },
      nodes,
      stats: { emitted: nodes.length, omitted, refEnd: refSeq },
    };
  }

  /** Text search over rendered content; returns refs, not selectors. */
  function find(query, limit) {
    const q = (query || '').toLowerCase().trim();
    if (!q) return { matches: [] };
    const refs = window.__agent.refs || new Map();
    const out = [];
    const seen = new Set();

    // 1) already-registered interactive elements
    for (const [ref, el] of refs) {
      if (!el.isConnected) continue;
      const hay = ((el.innerText || el.textContent || '') + ' ' +
        (el.getAttribute('aria-label') || '') + ' ' +
        (el.getAttribute('placeholder') || '') + ' ' +
        (el.getAttribute('title') || '') + ' ' +
        (el.getAttribute('value') || '')).toLowerCase();
      if (hay.includes(q)) {
        const role = roleOf(el);
        out.push({ ref, role, name: accessibleName(el, role), context: clip(el.parentElement ? el.parentElement.innerText : '', 160) });
        seen.add(el);
        if (out.length >= (limit || 20)) return { matches: out, truncated: true };
      }
    }
    // 2) any other visible element whose own text matches
    const all = document.body ? document.body.querySelectorAll('*') : [];
    for (const el of all) {
      if (seen.has(el) || SKIP_TAGS.has(el.tagName)) continue;
      const t = ownText(el).toLowerCase();
      if (!t || !t.includes(q)) continue;
      let style = null;
      try { style = getComputedStyle(el); } catch (e) { continue; }
      if (isHidden(el, style)) continue;
      const rect = el.getBoundingClientRect();
      if (!rect.width && !rect.height) continue;
      // Register a ref on the nearest actionable ancestor so the match is usable.
      const actionable = el.closest('a,button,input,select,textarea,[role],[onclick],[tabindex]') || el;
      let ref = window.__agent.byEl.get(actionable);
      if (!ref) {
        ref = 'e' + (++window.__agent.findSeq) + 'f';
        window.__agent.refs.set(ref, actionable);
        window.__agent.byEl.set(actionable, ref);
      }
      const role = roleOf(actionable);
      out.push({ ref, role, name: accessibleName(actionable, role), context: clip(el.innerText || el.textContent, 160) });
      if (out.length >= (limit || 20)) return { matches: out, truncated: true };
    }
    return { matches: out };
  }

  /** Readable text of the page (or of one ref), for reading-heavy steps. */
  function readText(ref, start, maxChars) {
    let el = null;
    if (ref) el = window.__agent.refs.get(ref) || null;
    if (!el) {
      el = document.querySelector('article') || document.querySelector('main') || document.body;
    }
    const clone = el.cloneNode(true);
    clone.querySelectorAll('script,style,noscript,svg,nav,footer,header').forEach((n) => n.remove());
    const text = (clone.innerText || clone.textContent || '')
      .replace(/[ \t ]+/g, ' ')
      .replace(/\n\s*\n\s*\n+/g, '\n\n')
      .trim();
    const chunk = text.slice(start || 0, (start || 0) + (maxChars || 4000));
    return { text: chunk, start: start || 0, returned: chunk.length, total: text.length };
  }

  function resolve(ref) {
    const el = window.__agent.refs.get(ref);
    if (!el) return { ok: false, reason: 'unknown_ref' };
    if (!el.isConnected) return { ok: false, reason: 'detached' };
    return { ok: true };
  }

  window.__agent = {
    version: 3,
    epoch: 0,
    seq: 0,
    findSeq: 900000,
    refs: new Map(),
    byEl: new WeakMap(),
    collect,
    find,
    readText,
    resolve,
    get: (ref) => window.__agent.refs.get(ref),
  };
})();
