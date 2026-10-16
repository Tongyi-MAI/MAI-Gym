/* MAI-Gym project page. Plain JS, no dependencies; data comes from assets/data.js. */
(function () {
  'use strict';
  const D = window.SITE_DATA;
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
  };

  /* ------------------------------------------------------------ header */
  const menu = $('#menu-toggle'), nav = $('#site-nav');
  menu.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menu.setAttribute('aria-expanded', open);
  });
  nav.addEventListener('click', e => { if (e.target.closest('a')) { nav.classList.remove('open'); menu.setAttribute('aria-expanded', false); } });

  /* ------------------------------------------------------------ zoom dialog */
  const zoomDialog = $('#zoom-dialog');
  function zoom(src, title) {
    $('#zoom-img').src = src;
    $('#zoom-img').alt = title;
    $('#zoom-title').textContent = title;
    zoomDialog.showModal();
  }
  $('#zoom-close').addEventListener('click', () => zoomDialog.close());
  zoomDialog.addEventListener('click', e => { if (e.target === zoomDialog) zoomDialog.close(); });
  document.addEventListener('click', e => {
    const z = e.target.closest('[data-zoom]');
    if (z) zoom(z.dataset.zoom, z.dataset.zoomTitle || '');
  });

  /* ------------------------------------------------------------ page chrome */
  const header = $('#site-header');
  const darkSections = [...document.querySelectorAll('.hero, .board-section, .reel-section, .site-footer')];
  let chromeQueued = false;
  function syncHeader() {
    chromeQueued = false;
    const y = 32, day = document.documentElement.dataset.theme === 'day';
    header.classList.toggle('on-dark', !day && darkSections.some(s => { const r = s.getBoundingClientRect(); return r.top <= y && r.bottom >= y; }));
  }

  // Day / night theme. The choice was applied before first paint by the inline script in <head>.
  const themeBtn = $('#theme-toggle');
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const day = theme === 'day';
    themeBtn.setAttribute('aria-pressed', String(day));
    themeBtn.setAttribute('aria-label', day ? 'Switch to night theme' : 'Switch to day theme');
    themeBtn.title = day ? 'Night theme' : 'Day theme';
    document.querySelector('meta[name=theme-color]').content = day ? '#fbfbfe' : '#0b0920';
    syncHeader();
  }
  themeBtn.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'day' ? 'night' : 'day';
    try { localStorage.setItem('maigym.theme', next); } catch (e) { /* private mode: the choice lasts this visit */ }
    applyTheme(next);
  });
  applyTheme(document.documentElement.dataset.theme === 'day' ? 'day' : 'night');
  window.addEventListener('scroll', () => { if (!chromeQueued) { chromeQueued = true; requestAnimationFrame(syncHeader); } }, { passive: true });
  syncHeader();

  function countUp(el) {
    const target = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), t0 = performance.now(), dur = 1400;
    (function step(now) {
      const k = Math.min(1, (now - t0) / dur), v = target * (1 - Math.pow(1 - k, 3));
      el.textContent = v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const revealer = new IntersectionObserver(entries => entries.forEach(en => {
    if (!en.isIntersecting) return;
    en.target.classList.add('in');
    if (!reduceMotion) en.target.querySelectorAll('[data-count]').forEach(countUp);
    revealer.unobserve(en.target);
  }), { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal').forEach(el => revealer.observe(el));
  if (!reduceMotion) document.querySelectorAll('.hero [data-count]').forEach(countUp);

  /* ------------------------------------------------------------ mock vs real */
  const mockApps = [...D.comparisons, { ...D.mockOnly, mockOnly: true, pairs: D.mockOnly.shots.map(s => ({ screen: s.screen, mock: s.mock, task: s.task })) }];
  const compare = { app: 0, screen: 0, pos: 50 };
  const view = $('#compare-view'), mockList = $('#mock-list'), mockSide = $('#mock-side');
  mockList.innerHTML = mockApps.map((c, i) => `<button type="button" role="tab" data-i="${i}"><b>${esc(c.en)}</b><span>${esc(c.domain)}${c.mockOnly ? ' · mock only' : ''}</span><em>${String(i + 1).padStart(2, '0')}</em></button>`).join('');
  mockList.addEventListener('click', e => {
    const b = e.target.closest('[data-i]');
    if (b) { compare.app = +b.dataset.i; compare.screen = 0; renderCompare(); }
  });
  function renderCompare() {
    const c = mockApps[compare.app], p = c.pairs[compare.screen];
    [...mockList.children].forEach((b, i) => b.setAttribute('aria-selected', i === compare.app));
    view.classList.toggle('mock-only', !p.real);
    $('#compare-mock').src = p.mock;
    $('#compare-mock').alt = `${c.en} mock in MAI-Gym: ${p.screen.split(' · ').pop()}`;
    $('#compare-real').src = p.real || p.mock;
    $('#compare-real').alt = p.real ? `${c.en} real app: ${p.screen.split(' · ').pop()}` : '';
    const desc = c.mockOnly
      ? '<p class="mock-desc">These flows cannot be reproduced on the real app without a real bank card, so a policy could never practise them live. In the mock, every transfer and unbinding runs against a backend that resets between attempts.</p>'
      : '';
    mockSide.innerHTML = `<p class="mock-kicker">${esc(c.en)} · ${esc(c.domain)}</p>
      ${p.task ? `<p class="mock-task"><b>Task</b><span>${esc(p.task)}</span></p>` : ''}${desc}
      <div class="screen-thumbs">${c.pairs.map((q, i) => `<button type="button" data-screen="${i}" aria-pressed="${i === compare.screen}">
        <span class="thumb-pair${q.real ? '' : ' one'}">${q.real ? `<img src="${q.real}" alt="" loading="lazy">` : ''}<img src="${q.mock}" alt="" loading="lazy"></span>
        <b>${esc(q.screen.split(' · ')[1] || q.screen)}</b><small>${q.real ? 'real · mock' : 'mock'}</small></button>`).join('')}</div>
      `;
  }
  mockSide.addEventListener('click', e => {
    const b = e.target.closest('[data-screen]');
    if (b) { compare.screen = +b.dataset.screen; renderCompare(); }
  });
  function setPos(p) {
    compare.pos = Math.max(0, Math.min(100, p));
    view.style.setProperty('--pos', compare.pos + '%');
    view.setAttribute('aria-valuenow', Math.round(compare.pos));
    view.setAttribute('aria-valuetext', `${Math.round(compare.pos)}% real app`);
  }
  function posFromEvent(e) {
    const r = view.getBoundingClientRect();
    setPos(((e.clientX - r.left) / r.width) * 100);
  }
  view.addEventListener('pointerdown', e => { if (view.classList.contains('mock-only')) return; view.setPointerCapture(e.pointerId); posFromEvent(e); });
  view.addEventListener('pointermove', e => { if (view.hasPointerCapture(e.pointerId)) posFromEvent(e); });
  view.addEventListener('keydown', e => {
    const step = { ArrowLeft: -4, ArrowRight: 4, PageDown: -20, PageUp: 20 }[e.key];
    if (step) { setPos(compare.pos + step); e.preventDefault(); }
    if (e.key === 'Home') { setPos(0); e.preventDefault(); }
    if (e.key === 'End') { setPos(100); e.preventDefault(); }
  });
  renderCompare();
  setPos(50);
  // A gentle sweep the first time the phone scrolls into view, so the slider reads as interactive.
  if (!reduceMotion) {
    const sweep = new IntersectionObserver(es => {
      if (!es[0].isIntersecting) return;
      sweep.disconnect();
      const t0 = performance.now();
      (function step(now) {
        const k = Math.min(1, (now - t0) / 1800);
        setPos(50 + Math.sin(k * Math.PI * 2) * 28 * (1 - k));
        if (k < 1) requestAnimationFrame(step);
      })(t0);
    }, { threshold: 0.6 });
    sweep.observe(view);
  }

  /* ------------------------------------------------------------ demo video */
  // The file is only fetched once someone asks for it: the <video> is preload="none"
  // and the source is loaded on the first open.
  const vdlg = $('#video-dialog'), vid = $('#demo-video');
  if (vdlg && vid) {
    const openVideo = () => {
      vdlg.showModal();
      document.body.classList.add('is-modal-open');
      if (!vid.dataset.ready) { vid.load(); vid.dataset.ready = '1'; }
      vid.play().catch(() => {});
    };
    const closeVideo = () => { vid.pause(); vdlg.close(); };
    $('#watch-demo')?.addEventListener('click', openVideo);
    $('#video-close')?.addEventListener('click', closeVideo);
    vdlg.addEventListener('click', e => { if (e.target === vdlg) closeVideo(); });
    vdlg.addEventListener('close', () => { vid.pause(); document.body.classList.remove('is-modal-open'); });
  }

  /* ------------------------------------------------------------ cover wall */
  // A slow wall of the environments the gym actually runs — phones and desktops —
  // set back in perspective behind the title. Decorative: aria-hidden, and the
  // animation stops under reduce-motion.
  const wall = $('#cover-wall');
  if (wall && D.apps.length) {
    const DESKS = ['d0', 'd3', 'd4', 'd6', 'd7', 'd9', 'd10', 'd11'];
    const phones = D.apps.map(a => `assets/thumbs/${a.id}.webp`);
    const desks = DESKS.map(d => `assets/desk/${d}.webp`);
    const COLS = 4;
    for (let c = 0; c < COLS; c++) {
      const col = document.createElement('div');
      col.className = 'wall-col';
      col.style.setProperty('--d', `${96 + c * 19}s`);
      col.style.setProperty('--o', `${c % 2 ? '-' : ''}${6 + c * 4}%`);
      // every column mixes both platforms, so the wall reads as mobile + computer
      const tiles = [];
      for (let i = 0; i < 6; i++) {
        tiles.push({ src: phones[(c * 5 + i * 3) % phones.length], kind: 'phone' });
        if (i % 2 === 1) tiles.push({ src: desks[(c * 2 + i) % desks.length], kind: 'desk' });
      }
      for (let k = 0; k < 2; k++) {          // twice through, so the loop has no seam
        tiles.forEach(t => {
          const im = document.createElement('img');
          im.src = t.src;
          im.className = t.kind;
          im.alt = '';
          im.loading = c > 1 ? 'lazy' : 'eager';
          im.decoding = 'async';
          col.appendChild(im);
        });
      }
      wall.appendChild(col);
    }
  }

  /* ------------------------------------------------------------ reel */
  function reelCard(app) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'reel-card';
    b.dataset.app = app.id;
    b.setAttribute('aria-label', `Play ${app.name}, ${app.domain}`);
    b.innerHTML = `<span class="reel-phone"><img src="assets/thumbs/${app.id}.webp" alt="" loading="lazy" width="160" height="346"><span class="reel-play">▶ Play</span></span>
      <span class="reel-caption"><span class="reel-name"><b>${String(app.index).padStart(2, '0')}</b>${esc(app.name)}</span><span class="reel-domain">${esc(app.domain)}</span></span>`;
    return b;
  }
  function fillReel(track, apps) {
    for (let k = 0; k < 2; k++) {
      const set = document.createElement('div');
      set.className = 'reel-set';
      apps.forEach(a => set.appendChild(reelCard(a)));
      // The copy keeps the strip seamless, and the reverse row shows it most of the time,
      // so it stays clickable: hidden from assistive tech and out of the tab order instead.
      if (k) {
        set.setAttribute('aria-hidden', 'true');
        [...set.children].forEach(b => { b.tabIndex = -1; });
      }
      track.appendChild(set);
    }
    track.style.setProperty('--reel-duration', `${apps.length * 6}s`);
  }
  const half = Math.ceil(D.apps.length / 2);
  fillReel($('#reel-row-one'), D.apps.slice(0, half));
  fillReel($('#reel-row-two'), D.apps.slice(half));
  $('#apps').addEventListener('click', e => {
    const card = e.target.closest('.reel-card');
    if (card) Player.open(card.dataset.app);
  });

  /* ------------------------------------------------------------ app player */
  const Player = (() => {
    const dialog = $('#player-dialog'), frame = $('#player-frame'), tapLayer = $('#tap-layer');
    const cache = {};
    const solved = new Set(store.get('maigym.solved', []));
    let app = null, data = null, task = null, levelFilter = 'all', playing = null, poll = null, scale = 1;

    const env = () => { try { return frame.contentWindow.__ENV__; } catch { return null; } };

    async function load(id) {
      if (!cache[id]) {
        const [tasks, oracle] = await Promise.all([
          fetch(`apps/${id}/tasks.json`).then(r => r.json()),
          fetch(`apps/${id}/oracle.json`).then(r => r.json()),
        ]);
        cache[id] = { tasks: tasks.tasks, oracle: oracle.solutions || {} };
      }
      return cache[id];
    }

    function fit() {
      const col = $('#phone-col');
      const stacked = window.matchMedia('(max-width: 860px)').matches;
      const availH = stacked ? window.innerHeight * 0.72 : col.clientHeight - 64;
      const availW = (stacked ? col.clientWidth : 460) - 40;
      scale = Math.min(1, availH / 862, availW / 408);
      col.style.setProperty('--s', scale.toFixed(4));
    }

    function waitReady() {
      return new Promise(resolve => {
        const t0 = performance.now();
        (function check() {
          let ready = false;
          try { ready = frame.contentWindow.__ENV_READY__ === true; } catch { /* cross-origin */ }
          if (ready || performance.now() - t0 > 8000) resolve(ready);
          else setTimeout(check, 60);
        })();
      });
    }

    async function open(id, taskId) {
      app = D.apps.find(a => a.id === id);
      if (!app) return;
      task = null;
      $('#player-title').innerHTML = `<b>${String(app.index).padStart(2, '0')}</b>${esc(app.name)}`;
      $('#player-sub').textContent = app.domain;
      $('#player-open').href = `apps/${id}/index.html`;
      $('#app-desc').textContent = app.functions;
      if (!dialog.open) dialog.showModal();
      document.body.classList.add('is-modal-open');
      fit();
      frame.src = `apps/${id}/index.html?bare=1`;
      renderActive();
      data = await load(id);
      renderTasks();
      await waitReady();
      startPoll();
      if (taskId) {
        const t = data.tasks.find(x => x.id === taskId);
        if (t) selectTask(t);
      }
      setHash();
    }

    function close() {
      stopPlayback();
      clearInterval(poll);
      frame.src = 'about:blank';
      document.body.classList.remove('is-modal-open');
      if (dialog.open) dialog.close();
      history.replaceState(null, '', location.pathname + location.search + '#apps');
    }

    function setHash() {
      if (!app) return;
      history.replaceState(null, '', `${location.pathname}${location.search}#play/${app.id}${task ? '/' + task.id : ''}`);
    }

    function startPoll() {
      clearInterval(poll);
      poll = setInterval(() => {
        const e = env();
        if (!e) return;
        try {
          const st = e.getState();
          $('#meta-steps').textContent = `${st.meta.steps} step${st.meta.steps === 1 ? '' : 's'}${task ? ` / ${task.max_steps} max` : ''}`;
        } catch { /* app still booting */ }
      }, 400);
    }

    const needsAnswer = t => t.goal.some(g => g.source === 'answer');
    const LEVEL = { L1: 'Easy', L2: 'Medium', L3: 'Hard' };   // the paper's three difficulty levels

    function renderActive(verdict) {
      const box = $('#active-task');
      if (!task) {
        box.className = 'active-task free p-top';
        box.innerHTML = `<div class="task-top"><span class="level L2">Free play</span></div>
          <p class="instruction">Explore the app freely — it runs entirely in your browser. Pick a task below to get an instruction, then press <b>Check</b> to see whether you solved it, or <b>Watch solution</b> to see it done.</p>
          <div class="task-actions"><button class="button secondary" type="button" data-act="reset">↺ Reset app</button></div>`;
        return;
      }
      box.className = 'active-task p-top';
      const hasSolution = (data.oracle[task.id] || []).length > 0;
      box.innerHTML = `<div class="task-top"><span class="level ${task.level}">${LEVEL[task.level] || task.level}</span><span>up to ${task.max_steps} steps</span></div>
        <p class="instruction">${esc(task.instruction)}</p>
        ${needsAnswer(task) ? `<div class="answer-row"><input id="answer-input" type="text" placeholder="Type your answer" aria-label="Your answer"><button class="button secondary" type="button" data-act="answer">Submit</button></div>` : ''}
        <div class="task-actions">
          <button class="button primary" type="button" data-act="check">✓ Check</button>
          ${hasSolution ? `<button class="button secondary" type="button" data-act="${playing ? 'stop' : 'watch'}">${playing ? '■ Stop' : '▶ Watch solution'}</button>` : ''}
          <button class="button secondary" type="button" data-act="restart">↺ Restart task</button>
        </div>
        <div id="verdict">${verdict || ''}</div>`;
    }

    function renderTasks() {
      $('#level-filter').innerHTML = ['all', 'L1', 'L2', 'L3'].map(l =>
        `<button type="button" data-level="${l}" aria-pressed="${levelFilter === l}">${l === 'all' ? 'All' : LEVEL[l]}</button>`).join('');
      const list = data.tasks.filter(t => levelFilter === 'all' || t.level === levelFilter);
      $('#task-list').innerHTML = list.map(t => `<li><button type="button" data-task="${esc(t.id)}" aria-current="${task && task.id === t.id}">
        <span class="level ${t.level}">${LEVEL[t.level] || t.level}</span>
        <span class="t-text"><b class="${solved.has(app.id + '/' + t.id) ? 'done' : ''}">${esc(t.name || t.id)}</b>${esc(t.instruction)}</span></button></li>`).join('');
    }

    function resetTo(t) {
      const e = env();
      if (!e) return;
      e.reset(t ? { seed: t.seed || 0, patch: (t.init && t.init.patch) || [] } : { seed: 0 });
    }

    function selectTask(t) {
      stopPlayback();
      task = t;
      resetTo(t);
      renderActive();
      renderTasks();
      setHash();
    }

    function trajectory(e, st) {
      const actions = e.trajectory();
      return { steps: st.meta.steps, n_actions: actions.length, actions, ids: actions.map(a => a.id ?? null), errors: [], truncated: false };
    }

    function check() {
      const e = env();
      if (!e || !task) return;
      const st = e.getState();
      const r = window.MaiVerifier.verify(task, st, trajectory(e, st));
      if (r.success) {
        solved.add(app.id + '/' + task.id);
        store.set('maigym.solved', [...solved]);
        renderTasks();
      }
      const items = r.criteria.map(c => {
        const cls = c.passed ? 'pass' : c.critical ? 'fail' : 'warn';
        return `<li class="${cls}"><span class="mark">${c.passed ? '✓' : c.critical ? '✗' : '!'}</span><span>${esc(c.name)}</span></li>`;
      }).join('');
      renderActive(`<div class="verdict"><div class="verdict-head ${r.success ? 'ok' : 'fail'}"><span>${r.success ? 'Task solved' : 'Not solved yet'}</span><span>reward ${r.score.toFixed(2)}</span></div><ul>${items}</ul></div>`);
    }

    function submitAnswer() {
      const e = env(), input = $('#answer-input');
      if (!e || !input) return;
      e.step({ action: 'finish', answer: input.value });
      check();
    }

    function tapAt(id) {
      const e = env();
      let b;
      try { b = id && e.bounds()[id]; } catch { b = null; }
      if (!b) return;
      const dot = document.createElement('span');
      dot.className = 'tap-dot';
      dot.style.left = (b.cx * scale) + 'px';
      dot.style.top = (b.cy * scale) + 'px';
      tapLayer.appendChild(dot);
      setTimeout(() => dot.remove(), 750);
    }

    async function watch() {
      const actions = data.oracle[task.id] || [];
      const token = playing = {};
      tapLayer.style.pointerEvents = 'auto';
      resetTo(task);
      renderActive();
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      await sleep(500);
      for (const a of actions) {
        if (playing !== token) return;
        if (a.id) { tapAt(a.id); await sleep(380); }
        if (a.action === 'finish' && a.answer !== undefined) {
          const input = $('#answer-input');
          if (input) input.value = a.answer;
        }
        try { env().step(a); } catch { /* keep replaying */ }
        await sleep(650);
      }
      if (playing !== token) return;
      playing = null;
      tapLayer.style.pointerEvents = '';
      check();
    }

    function stopPlayback() {
      playing = null;
      tapLayer.style.pointerEvents = '';
    }

    $('#active-task').addEventListener('click', e => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'reset') resetTo(null);
      if (act === 'restart') { stopPlayback(); resetTo(task); renderActive(); }
      if (act === 'check') check();
      if (act === 'answer') submitAnswer();
      if (act === 'watch') watch();
      if (act === 'stop') { stopPlayback(); renderActive(); }
    });
    $('#active-task').addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.id === 'answer-input') submitAnswer(); });
    $('#task-list').addEventListener('click', e => {
      const b = e.target.closest('[data-task]');
      if (b) {
        selectTask(data.tasks.find(t => t.id === b.dataset.task));
        $('#panel-col').scrollTo({ top: 0, behavior: 'smooth' });
        $('.player-body').scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
    $('#level-filter').addEventListener('click', e => {
      const b = e.target.closest('[data-level]');
      if (b) { levelFilter = b.dataset.level; renderTasks(); }
    });
    $('#player-close').addEventListener('click', close);
    dialog.addEventListener('cancel', e => { e.preventDefault(); close(); });
    dialog.addEventListener('click', e => { if (e.target === dialog) close(); });
    window.addEventListener('resize', () => { if (dialog.open) fit(); });
    return { open };
  })();

  /* ------------------------------------------------------------ overview */
  $('#overview-img').src = D.figures.overview.src;
  $('#overview-img').dataset.zoom = D.figures.overview.src;
  $('#overview-img').dataset.zoomTitle = 'Overview of MAI-Gym';
  // The dashboard answers three questions (paper, Dashboard paragraph); parts follow its figure caption.
  const DASH = [
    { key: 'dash_trajectory', label: 'Trajectory', q: 'Where did an attempt go wrong?',
      d: 'Each screenshot lines up with the model\'s output and the action it took, next to the verifier\'s result.',
      parts: ['Task goal and verifier result', 'Screenshots with the actions drawn on them', 'Model output and executed coordinates'] },
    { key: 'dash_run', label: 'Runs', q: 'Which tasks explain a score?',
      d: 'Follow a run from its summary down to every task and attempt, reruns included.',
      parts: ['Run navigation', 'Evaluation rollouts', 'Selected training batches'] },
    { key: 'dash_fleet', label: 'Fleet', q: 'Is the fleet healthy?',
      d: 'Availability, active sessions, and throughput over time expose shortages and stalled or unhealthy instances.',
      parts: ['Fleet capacity in use', 'Session and throughput history', 'Session inventory'] },
    { key: 'dash_benchmarks', label: 'Benchmarks', q: 'Which tasks explain a score?',
      d: 'Every benchmark score links to its per-task results and the runs behind them.',
      parts: ['Benchmark selection', 'Scores and task counts', 'Underlying runs'] },
  ];
  $('#dash-thumbs').innerHTML = DASH.map((v, i) => `<button type="button" role="tab" data-k="${v.key}" aria-selected="${i === 0}">${v.label}</button>`).join('');
  function showDash(k) {
    const v = DASH.find(d => d.key === k);
    $('#dash-img').src = D.figures[v.key].src;
    $('#dash-img').alt = `MAI-Gym dashboard, ${v.label} view`;
    $('#dash-zoom').dataset.zoom = D.figures[v.key].src;
    $('#dash-zoom').dataset.zoomTitle = `Dashboard · ${v.label}`;
    $('#dash-bar-label').textContent = `Dashboard · ${v.label}`;
    $('#dash-q').textContent = v.q;
    $('#dash-d').textContent = v.d;
    $('#dash-parts').innerHTML = v.parts.map(t => `<li>${esc(t)}</li>`).join('');
    [...$('#dash-thumbs').children].forEach(b => b.setAttribute('aria-selected', b.dataset.k === v.key));
  }
  $('#dash-thumbs').addEventListener('click', e => { const b = e.target.closest('[data-k]'); if (b) showDash(b.dataset.k); });
  showDash(DASH[0].key);

  /* ------------------------------------------------------------ citation */
  const bib = $('#citation-code').textContent;
  $('#copy-citation').addEventListener('click', async e => {
    try { await navigator.clipboard.writeText(bib); e.target.textContent = 'Copied'; }
    catch { e.target.textContent = 'Select and copy'; }
    setTimeout(() => { e.target.textContent = 'Copy'; }, 1600);
  });
  $('#download-citation').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bib], { type: 'text/plain' }));
    a.download = 'maigym.bib';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  /* ------------------------------------------------------------ deep links */
  function openFromHash() {
    const m = location.hash.match(/^#play\/([\w-]+)(?:\/([\w-]+))?/);
    if (m) Player.open(m[1], m[2]);
  }
  window.addEventListener('hashchange', openFromHash);
  openFromHash();
})();
