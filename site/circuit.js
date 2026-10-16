/* MAI-Gym training board: an interactive, circuit-style diagram of how one
   rollout path serves RL training, evaluation, and data collection.
   Layout mirrors the paper's architecture figure. Text stays in plain words:
   no API calls, formulas, or internal states. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1280, H = 720;
  const TYPES = {
    // colors are CSS variables so the day and night themes can each pick legible ones (styles.css)
    ctrl: { color: 'var(--sig-ctrl)', label: 'Requests & actions' },
    obs: { color: 'var(--sig-obs)', label: 'Observations & screenshots' },
    reward: { color: 'var(--sig-reward)', label: 'Rewards & trajectories' },
    weights: { color: 'var(--sig-weights)', label: 'Policy weights' },
    loop: { color: 'var(--sig-loop)', label: 'Diagnose → synthesize loop' },
  };

  /* ------------------------------------------------------------ parts */
  const ZONES = [
    { x: 20, y: 40, w: 440, h: 562, label: 'ROLLOUT · INFERENCE · TRAINING' },
    { x: 476, y: 40, w: 430, h: 562, label: 'ENVIRONMENT MANAGER' },
    { x: 914, y: 40, w: 346, h: 562, label: 'GUI ENVIRONMENT FLEET · 20,000 TESTED' },
    { x: 20, y: 614, w: 828, h: 92, label: 'OBSERVE · DIAGNOSE · SYNTHESIZE' },
  ];
  const CHIPS = [
    { id: 'pool', code: 'U1', x: 44, y: 70, w: 176, h: 72, t: 'Task Pool', s: 'synthesized, verifiable tasks' },
    { id: 'sched', code: 'U2', x: 260, y: 70, w: 176, h: 72, t: 'Rollout Scheduler', s: 'picks tasks each round' },
    { id: 'work', code: 'U3', x: 140, y: 176, w: 210, h: 114, t: 'Rollout Workers ×N', s: 'run attempts in parallel', kind: 'stack' },
    { id: 'gw', code: 'U4', x: 44, y: 346, w: 160, h: 66, t: 'Model Gateway', s: 'routes model calls' },
    { id: 'eng', code: 'U5', x: 44, y: 452, w: 160, h: 126, t: 'SGLang Engines', s: 'serve the policy', kind: 'gpu' },
    { id: 'buf', code: 'U6', x: 250, y: 346, w: 186, h: 66, t: 'Trajectory Buffer', s: 'trajectory · reward · outcome' },
    { id: 'train', code: 'U7', x: 250, y: 452, w: 186, h: 126, t: 'Trainer · Megatron', s: 'slime', kind: 'train' },
    { id: 'cli', code: 'U8', x: 492, y: 170, w: 104, h: 64, t: 'EnvClient', s: 'one interface' },
    { id: 'oss', code: 'U10', x: 482, y: 470, w: 134, h: 64, t: 'Trajectory Store', s: 'written asynchronously' },
    { id: 'mgr', code: 'U9', x: 624, y: 62, w: 204, h: 458, t: 'Environment Manager', s: '', kind: 'board' },
    { id: 'prov', code: '', x: 638, y: 100, w: 176, h: 62, t: 'Provisioning', s: 'start · keep ready', sub: true },
    { id: 'sess', code: '', x: 638, y: 170, w: 176, h: 62, t: 'Sessions', s: 'hand out · reset · clean up', sub: true },
    { id: 'task', code: '', x: 638, y: 240, w: 176, h: 62, t: 'Tasks', s: 'register task sets', sub: true },
    { id: 'rew', code: '', x: 638, y: 310, w: 176, h: 62, t: 'Reward', s: 'code checks · model judge', sub: true },
        { id: 'tel', code: '', x: 638, y: 380, w: 176, h: 62, t: 'Telemetry', s: 'status → dashboard', sub: true },
    { id: 'bus', code: '', x: 872, y: 70, w: 26, h: 460, t: 'ENVIRONMENT ADAPTORS', s: '', kind: 'bus' },
    { id: 'mob', code: 'U11', x: 924, y: 70, w: 320, h: 118, t: 'Mobile', s: 'MobileWorld · AndroidWorld · MobileGym', kind: 'fleet' },
    { id: 'cmp', code: 'U12', x: 924, y: 206, w: 320, h: 118, t: 'Computer', s: 'OSWorld · MacAgentBench · MyPCBench', kind: 'fleet' },
    { id: 'web', code: 'U13', x: 924, y: 342, w: 320, h: 118, t: 'Web', s: 'WebArena · WebVoyager · locally served apps', kind: 'fleet' },
    { id: 'dep', code: 'U14', x: 924, y: 490, w: 320, h: 88, t: 'Deployment Backends', s: 'Kubernetes · Docker · physical devices' },
    { id: 'synth', code: 'U15', x: 44, y: 636, w: 280, h: 56, t: 'Task Synthesis', s: 'world → task → verifier' },
    { id: 'diag', code: 'U16', x: 344, y: 636, w: 260, h: 56, t: 'Diagnosis Agent', s: 'coding agents read trajectories' },
    { id: 'dash', code: 'U17', x: 624, y: 636, w: 204, h: 56, t: 'Dashboard', s: 'fleet · runs · scores' },
  ];
  const chipById = Object.fromEntries(CHIPS.map(c => [c.id, c]));

  // Orthogonal waypoints; corners are cut at 45° when drawn, PCB-style.
  const TRACES = [
    { id: 'pool_sched', a: 'pool', b: 'sched', type: 'ctrl', p: [[220, 106], [260, 106]] },
    { id: 'sched_work', a: 'sched', b: 'work', type: 'ctrl', p: [[330, 142], [330, 176]] },
    { id: 'work_gw', a: 'work', b: 'gw', type: 'ctrl', p: [[176, 290], [176, 320], [124, 320], [124, 346]], label: ['model calls', 132, 336] },
    { id: 'gw_eng', a: 'gw', b: 'eng', type: 'ctrl', p: [[124, 412], [124, 452]] },
    { id: 'eng_work', a: 'eng', b: 'work', type: 'ctrl', p: [[204, 470], [226, 470], [226, 290]] },
    { id: 'work_cli', a: 'work', b: 'cli', type: 'ctrl', p: [[350, 192], [492, 192]], label: ['requests', 404, 184] },
    { id: 'cli_work', a: 'cli', b: 'work', type: 'obs', p: [[492, 212], [350, 212]] },
    { id: 'oss_work', a: 'oss', b: 'work', type: 'obs', p: [[482, 502], [466, 502], [466, 262], [350, 262]], label: ['image', 400, 254] },
    { id: 'cli_sess', a: 'cli', b: 'sess', type: 'ctrl', p: [[596, 192], [638, 192]] },
    { id: 'sess_cli', a: 'sess', b: 'cli', type: 'obs', p: [[638, 212], [596, 212]] },
    { id: 'prov_bus', a: 'prov', b: 'bus', type: 'ctrl', p: [[814, 131], [872, 131]] },
    { id: 'sess_bus', a: 'sess', b: 'bus', type: 'ctrl', p: [[814, 192], [872, 192]] },
    { id: 'bus_sess', a: 'bus', b: 'sess', type: 'obs', p: [[872, 212], [814, 212]] },
    { id: 'bus_rew', a: 'bus', b: 'rew', type: 'obs', p: [[872, 341], [814, 341]] },
    { id: 'bus_mob', a: 'bus', b: 'mob', type: 'ctrl', p: [[898, 112], [924, 112]] },
    { id: 'mob_bus', a: 'mob', b: 'bus', type: 'obs', p: [[924, 134], [898, 134]] },
    { id: 'bus_cmp', a: 'bus', b: 'cmp', type: 'ctrl', p: [[898, 248], [924, 248]] },
    { id: 'cmp_bus', a: 'cmp', b: 'bus', type: 'obs', p: [[924, 270], [898, 270]] },
    { id: 'bus_web', a: 'bus', b: 'web', type: 'ctrl', p: [[898, 384], [924, 384]] },
    { id: 'web_bus', a: 'web', b: 'bus', type: 'obs', p: [[924, 406], [898, 406]] },
    { id: 'bus_dep', a: 'bus', b: 'dep', type: 'ctrl', p: [[898, 520], [924, 520]] },
    { id: 'dep_mob', a: 'dep', b: 'mob', type: 'ctrl', p: [[1244, 534], [1258, 534], [1258, 129], [1244, 129]] },
    { id: 'dep_cmp', a: 'dep', b: 'cmp', type: 'ctrl', p: [[1258, 265], [1244, 265]] },
    { id: 'dep_web', a: 'dep', b: 'web', type: 'ctrl', p: [[1258, 401], [1244, 401]] },
    { id: 'bus_oss', a: 'bus', b: 'oss', type: 'obs', p: [[885, 530], [885, 552], [544, 552], [544, 534]], label: ['screenshots', 600, 568] },
    { id: 'rew_buf', a: 'rew', b: 'buf', type: 'reward', p: [[638, 362], [436, 362]], label: ['reward', 548, 354] },
    { id: 'work_buf', a: 'work', b: 'buf', type: 'reward', p: [[330, 290], [330, 346]] },
    { id: 'buf_train', a: 'buf', b: 'train', type: 'reward', p: [[343, 412], [343, 452]] },
    { id: 'train_eng', a: 'train', b: 'eng', type: 'weights', p: [[250, 540], [204, 540]], label: ['weights', 210, 532] },
    { id: 'tel_dash', a: 'tel', b: 'dash', type: 'loop', p: [[726, 442], [726, 636]] },
    { id: 'dash_diag', a: 'dash', b: 'diag', type: 'loop', p: [[624, 664], [604, 664]] },
    { id: 'diag_synth', a: 'diag', b: 'synth', type: 'loop', p: [[344, 664], [324, 664]] },
    { id: 'synth_pool', a: 'synth', b: 'pool', type: 'loop', p: [[44, 664], [10, 664], [10, 106], [44, 106]], label: ['new tasks → task pool', 7, 470, -90] },
  ];
  const traceById = Object.fromEntries(TRACES.map(t => [t.id, t]));

  /* ------------------------------------------------------------ stories */
  const S = {
    sample: { t: 'Sample tasks', chips: ['pool', 'sched', 'work'], traces: ['pool_sched', 'sched_work'],
      d: 'Before each round the scheduler estimates every task\'s success rate from its past attempts. Unseen and uncertain tasks get the full group of attempts; never-solved and mastered tasks get smaller budgets, so their status keeps being rechecked as the policy moves.',
      w: '' },
    acquire: { t: 'Acquire & reset', chips: ['work', 'cli', 'sess', 'prov', 'bus', 'mob', 'cmp', 'web', 'dep'], traces: ['work_cli', 'cli_sess', 'sess_bus', 'prov_bus', 'bus_mob', 'bus_cmp', 'bus_web', 'bus_dep'],
      d: 'A worker takes a ready environment through the EnvClient, and the manager resets it to the task\'s starting state: a clean snapshot on phones, a checked reset on computers, a fresh session in the browser.',
      w: '' },
    observe: { t: 'Observe', chips: ['mob', 'cmp', 'web', 'bus', 'sess', 'cli', 'oss', 'work'], traces: ['mob_bus', 'cmp_bus', 'web_bus', 'bus_sess', 'sess_cli', 'cli_work', 'bus_oss', 'oss_work'],
      d: 'The step result returns through the manager, while screenshots travel through the screenshot store instead — which keeps the manager\'s traffic small at 20,000 environments.',
      w: '' },
    think: { t: 'Think', chips: ['work', 'gw', 'eng'], traces: ['work_gw', 'gw_eng'],
      d: 'The agent builds its prompt from the latest screenshot and its recent history. The model gateway sends every turn of an attempt to the same engine, so the engine can reuse the work it has already done.',
      w: '' },
    act: { t: 'Act', chips: ['eng', 'work', 'cli', 'sess', 'bus', 'mob', 'cmp', 'web'], traces: ['eng_work', 'work_cli', 'cli_sess', 'sess_bus', 'bus_mob', 'bus_cmp', 'bus_web'],
      d: 'The model\'s reply becomes an action — a tap, a click, or typed text — that runs in the environment. Each action is applied exactly once, even if a request is retried. Observe → think → act repeats until the agent finishes or reaches the step limit.',
      w: '' },
    verify: { t: 'Verify', chips: ['work', 'cli', 'sess', 'bus', 'rew'], traces: ['work_cli', 'cli_sess', 'bus_rew'],
      d: 'The task\'s own verifier checks the final state and returns the reward. If the environment itself breaks, the attempt is marked as an environment failure instead of scoring zero.',
      w: '' },
    collect: { t: 'Collect', chips: ['work', 'rew', 'buf', 'bus', 'tel'], traces: ['work_buf', 'rew_buf'],
      d: 'The attempt, its reward, and its outcome go into the trajectory buffer, and the manager saves the attempt. Up to this point, training, evaluation, and data collection run exactly the same code.',
      w: '' },
    learn: { t: 'Learn', chips: ['buf', 'train'], traces: ['buf_train'],
      d: 'When every attempt at a task has finished, the trainer compares them with each other: attempts that did better than the group are reinforced, the others discouraged (GRPO).',
      w: '' },
    sync: { t: 'Sync weights', chips: ['train', 'eng'], traces: ['train_eng'],
      d: 'The updated policy is loaded into every inference engine and the next round begins. The same trajectories also work with DAPO, GSPO, REINFORCE++, SAO, and PPO.',
      w: '' },
    loop: { t: 'Diagnose & synthesize', chips: ['tel', 'dash', 'diag', 'synth', 'pool'], traces: ['tel_dash', 'dash_diag', 'diag_synth', 'synth_pool'],
      d: 'Coding agents read the attempts in the dashboard and report what the model cannot do yet. The synthesis pipeline builds a realistic world, writes new tasks aimed at those weaknesses, and keeps a task only after its verifier passes tests on live runs.',
      w: '' },
    queue: { t: 'Queue benchmarks', chips: ['pool', 'sched', 'work'], traces: ['pool_sched', 'sched_work'],
      d: 'One command queues every task of the chosen benchmarks, from 10 supported across mobile, computer, and web — with repeated trials and each benchmark\'s own step limit — for a local checkpoint or an API-served model.',
      w: 'mai-gym eval --model <ckpt|endpoint> --benchmarks mobileworld osworld-verified … --trials 3' },
    score: { t: 'Aggregate', chips: ['buf', 'train', 'tel', 'dash'], traces: ['buf_train', 'tel_dash'],
      d: 'Each benchmark is scored with its own metric. Attempts that failed because of the environment are rerun and stay visible in the dashboard. No training happens: the model stays the same.',
      w: '' },
    store: { t: 'Store', chips: ['buf', 'train'], traces: ['buf_train'],
      d: 'Attempts are saved as they are, for fine-tuning or analysis — recorded with the same agent code the model will later be trained and evaluated with.',
      w: '' },
  };
  const MODES = {
    train: { label: 'RL training', stages: ['sample', 'acquire', 'observe', 'think', 'act', 'verify', 'collect', 'learn', 'sync', 'loop'] },
    eval: { label: 'Evaluation', stages: ['queue', 'acquire', 'observe', 'think', 'act', 'verify', 'collect', 'score', 'loop'],
      relabel: { pool: ['Benchmark suite', '10 GUI benchmarks'], sched: ['Rollout Scheduler', 'repeated trials · step limits'], train: ['Scorer', 'success rate per benchmark'] }, off: ['train_eng'] },
    data: { label: 'Data collection', stages: ['sample', 'acquire', 'observe', 'think', 'act', 'verify', 'collect', 'store'],
      relabel: { train: ['Trajectory Writer', 'saves attempts for later training'] }, off: ['train_eng'] },
  };

  const INFO = {
    pool: ['Task Pool', 'Synthesized, verifiable tasks across MobileWorld and OSWorld environments, plus tasks in the apps MAI-Gym builds itself.', ['2,000 tasks released', 'never overlaps a benchmark task']],
    sched: ['Rollout Scheduler', 'Queues a set of attempts each round and closes the round only when every attempt has finished.', ['more attempts where the model is uncertain', 'fewer on tasks it always solves or never solves']],
    work: ['Rollout workers', 'Each worker takes an environment, runs the agent on a task, and records what happened. The same agent code serves training, evaluation, and data collection.', ['one agent, three workloads', 'a fix reaches all three at once']],
    gw: ['Model Gateway', 'Routes every model call to an inference engine. All turns of one attempt go to the same engine so it can reuse its earlier work; if that engine fails, the attempt moves to another.', []],
    eng: ['SGLang Engines', 'Run the current policy to choose every action. Consecutive prompts of an attempt start the same way, so the engine reuses much of its earlier work.', []],
    buf: ['Trajectory Buffer', 'Every attempt with its reward and outcome. What happens next is the only difference between training, evaluation, and data collection.', ['environment failures are marked, not scored as zero']],
    train: ['Trainer', 'Trains the policy on the collected attempts with Megatron-LM, inside the slime RL framework.', ['GRPO · DAPO · GSPO · REINFORCE++ · SAO · PPO']],
    cli: ['EnvClient', 'The single client every agent uses. The agent side and the environment side talk only through this common interface.', ['acquire · reset · step · evaluate · release']],
    oss: ['Trajectory Store', 'Attempts are written to shared storage as they run, and step results carry only references, so the manager\'s traffic stays small as the fleet grows.', ['images travel by URL, not inline']],
    mgr: ['Environment Manager', 'Sits behind the common interface: starts environments, hands them out and resets them, registers tasks, computes rewards, saves attempts, and reports status.', ['every operation has a time limit']],
    prov: ['Provisioning', 'Starts thousands of environments at once and keeps some ready in advance, so an attempt rarely waits for a fresh start.', []],
    sess: ['Sessions', 'Tracks each environment from start to cleanup, so only one worker uses it at a time and a broken one is repaired before it is handed out again.', []],
    task: ['Tasks', 'New task sets — instruction, files, setup, and verifier — are registered with the manager and run on the existing environments.', []],
    rew: ['Reward', 'Each task\'s own code checks the final state; a model-based judge is optional. Checks run isolated and with a time limit.', ['a stuck check cannot stall the run']],
    tel: ['Telemetry', 'Reports environment, session, and run status to the dashboard.', []],
    bus: ['Environment adaptors', 'The manager\'s adapter layer fans every session out to the environment that serves it, and carries observations back.', ['add a benchmark = add an adapter']],
    mob: ['Mobile', 'Android phones run as lightweight containers, and each reset restores a clean snapshot of the device and its services.', ['MobileWorld · AndroidWorld · MobileGym']],
    cmp: ['Computer', 'Desktop instances with strict reset and readiness checks, isolated per-instance state, and time for apps to finish saving before verification.', ['OSWorld · MacAgentBench · MyPCBench']],
    web: ['Web', 'Headless browsers host real websites and locally served apps, including the ones MAI-Gym builds itself.', ['WebArena · WebVoyager · locally served apps']],
    dep: ['Deployment Backends', 'Kubernetes or Docker, or existing phones, computers, and browser sessions. Tested with 20,000 parallel instances; not a limit of the design.', ['20,000 instances tested']],
    synth: ['Task Synthesis', 'Builds a realistic world of files and app data, writes tasks inside it, and generates a verifier for each. A task is kept only after its verifier passes tests on live runs.', []],
    diag: ['Diagnosis Agent', 'Coding agents such as Claude Code or Codex compare successful and failed attempts and report what the model cannot do yet.', ['weaknesses steer the next tasks']],
    dash: ['Dashboard', 'Fleet health, runs, benchmark scores, and step-by-step trajectories — every number links back to the screenshots behind it.', []],
  };

  /* ------------------------------------------------------------ svg helpers */
  function el(tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function chamfer(pts, r = 10) {
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [px, py] = pts[i - 1], [x, y] = pts[i], [nx, ny] = pts[i + 1];
      const l1 = Math.hypot(x - px, y - py), l2 = Math.hypot(nx - x, ny - y);
      const c = Math.min(r, l1 / 2, l2 / 2);
      d += `L${x - ((x - px) / l1) * c},${y - ((y - py) / l1) * c}L${x + ((nx - x) / l2) * c},${y + ((ny - y) / l2) * c}`;
    }
    const last = pts[pts.length - 1];
    return d + `L${last[0]},${last[1]}`;
  }

  /* ------------------------------------------------------------ build */
  function build(root) {
    const ctl = (k, label, icon) => `<button type="button" class="ctl${k === 'play' ? ' play' : ''}" data-ctl="${k}" aria-label="${label}">${icon}</button>`;
    root.innerHTML = `
      <div class="board-main">
        <div class="board-view">
          <div class="board-scroll"><div class="board-canvas"></div></div>
          <p class="swipe-hint">← swipe to explore the board · it follows the active stage →</p>
        </div>
        <div class="board-side">
          <div class="mode-switch" role="tablist" aria-label="Workload">${Object.entries(MODES).map(([k, m]) => `<button type="button" role="tab" data-mode="${k}">${m.label}</button>`).join('')}</div>
          <ol class="stage-rail" aria-label="Stages"></ol>
          <div class="readout">
            <div class="stage-card" aria-live="polite">
              <div class="card-head"><p class="stage-kicker"></p><div class="board-ctrl">${ctl('prev', 'Previous stage', '<svg viewBox="0 0 16 16"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>')}${ctl('play', 'Pause', '<svg viewBox="0 0 16 16" class="i-pause"><path d="M5 3v10M11 3v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><svg viewBox="0 0 16 16" class="i-play"><path d="M5 3l8 5-8 5z" fill="currentColor"/></svg>')}${ctl('next', 'Next stage', '<svg viewBox="0 0 16 16"><path d="m6 3 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>')}</div></div>
              <h3></h3><p class="stage-text"></p><code class="wire"></code>
            </div>
            <div class="inspector" aria-live="polite"><p class="stage-kicker">Inspector</p><h3></h3><p class="stage-text"></p><ul></ul></div>
          </div>
        </div>
      </div>`;

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'board-svg', role: 'img', 'aria-label': 'Circuit diagram of the MAI-Gym training system' });
    root.querySelector('.board-canvas').appendChild(svg);
    const defs = el('defs', {}, svg);
    const glow = el('filter', { id: 'bglow', x: '-50%', y: '-50%', width: '200%', height: '200%' }, defs);
    el('feGaussianBlur', { stdDeviation: '2.6', result: 'b' }, glow);
    const merge = el('feMerge', {}, glow);
    el('feMergeNode', { in: 'b' }, merge);
    el('feMergeNode', { in: 'SourceGraphic' }, merge);
    const pat = el('pattern', { id: 'bdots', width: 16, height: 16, patternUnits: 'userSpaceOnUse' }, defs);
    el('circle', { cx: 1, cy: 1, r: 0.9, class: 'bdot' }, pat);
    const grad = el('linearGradient', { id: 'chipg', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    el('stop', { offset: '0', 'stop-color': '#1d1846' }, grad);
    el('stop', { offset: '1', 'stop-color': '#141032' }, grad);

    el('rect', { x: 0, y: 0, width: W, height: H, fill: 'url(#bdots)' }, svg);
    const gz = el('g', { class: 'zones' }, svg);
    ZONES.forEach(z => {
      el('rect', { x: z.x, y: z.y, width: z.w, height: z.h, rx: 14, class: 'zone' }, gz);
      el('text', { x: z.x + 14, y: z.y - 8, class: 'silk' }, gz, z.label);
    });

    const gt = el('g', { class: 'traces' }, svg);
    TRACES.forEach(t => {
      const d = chamfer(t.p);
      el('path', { d, class: 'under' }, gt);
      t.halo = el('path', { d, class: 'halo' }, gt);
      t.halo.style.stroke = TYPES[t.type].color;
      t.el = el('path', { d, class: 'trace', 'data-type': t.type }, gt);
      t.el.style.setProperty('--c', TYPES[t.type].color);
      t.len = t.el.getTotalLength();
      if (t.label) {
        const [txt, x, y, rot] = t.label;
        el('text', { x, y, class: 'silk small', transform: rot ? `rotate(${rot} ${x} ${y})` : '' }, gt, txt);
      }
    });
    // junction vias on the host trace
    [[1258, 265], [1258, 401]].forEach(([x, y]) => el('circle', { cx: x, cy: y, r: 3.4, class: 'via' }, gt));

    // signal key, silkscreened into the free corner of the board
    const gk = el('g', { class: 'key' }, svg);
    el('rect', { x: 864, y: 614, width: 396, height: 92, rx: 14, class: 'zone' }, gk);
    el('text', { x: 878, y: 606, class: 'silk' }, gk, 'SIGNALS');
    [['ctrl', 0, 0], ['obs', 1, 0], ['reward', 0, 1], ['loop', 1, 1], ['weights', 0, 2]].forEach(([k, col, row]) => {
      const t = TYPES[k], x = 882 + col * 166, y = 640 + row * 24;
      el('line', { x1: x, y1: y, x2: x + 18, y2: y, class: 'key-line' }, gk).style.stroke = t.color;
      el('circle', { cx: x + 9, cy: y, r: 2.6, filter: 'url(#bglow)' }, gk).style.fill = t.color;
      el('text', { x: x + 27, y: y + 4, class: 'key-t' }, gk, t.label);
    });

    const gc = el('g', { class: 'chips' }, svg);
    CHIPS.forEach(c => drawChip(gc, c));

    const gp = el('g', { class: 'pads' }, svg);
    TRACES.forEach(t => {
      [t.p[0], t.p[t.p.length - 1]].forEach(([x, y]) => el('circle', { cx: x, cy: y, r: 2.6, class: 'pad' }, gp));
    });
    const pulses = el('g', { class: 'pulses', filter: 'url(#bglow)' }, svg);
    return { svg, pulses };
  }

  function drawChip(g, c) {
    const grp = el('g', { class: `chip${c.sub ? ' sub' : ''}${c.kind ? ' k-' + c.kind : ''}`, 'data-id': c.id, tabindex: c.kind === 'board' ? -1 : 0, role: 'button', 'aria-label': c.t }, g);
    c.g = grp;
    if (c.kind === 'stack') {
      el('rect', { x: c.x + 12, y: c.y - 12, width: c.w, height: c.h, rx: 10, class: 'ghost' }, grp);
      el('rect', { x: c.x + 6, y: c.y - 6, width: c.w, height: c.h, rx: 10, class: 'ghost' }, grp);
    }
    if (c.kind === 'bus') {
      el('rect', { x: c.x, y: c.y, width: c.w, height: c.h, rx: 8, class: 'body' }, grp);
      el('text', { x: c.x + c.w / 2 + 4, y: c.y + c.h / 2, class: 'bus-label', transform: `rotate(-90 ${c.x + c.w / 2 + 4} ${c.y + c.h / 2})`, 'text-anchor': 'middle' }, grp, c.t);
      return;
    }
    if (!c.sub && c.kind !== 'board') {
      const legs = Math.floor((c.w - 28) / 14);
      for (let i = 0; i < legs; i++) {
        const x = c.x + 18 + i * 14;
        el('rect', { x, y: c.y - 3, width: 5, height: 3, rx: 1, class: 'leg' }, grp);
        el('rect', { x, y: c.y + c.h, width: 5, height: 3, rx: 1, class: 'leg' }, grp);
      }
    }
    el('rect', { x: c.x, y: c.y, width: c.w, height: c.h, rx: c.sub ? 8 : 10, class: 'body' }, grp);
    if (c.kind === 'board') {
      el('text', { x: c.x + 14, y: c.y + 24, class: 'chip-t' }, grp, c.t);
      return;
    }
    if (c.code) el('circle', { cx: c.x + 10, cy: c.y + 10, r: 2.2, class: 'notch' }, grp);
    const ty = c.sub ? c.y + 26 : c.y + 31;
    c.tEl = el('text', { x: c.x + 14, y: ty, class: c.sub ? 'chip-t sm' : 'chip-t' }, grp, c.t);
    c.sEl = el('text', { x: c.x + 14, y: ty + (c.sub ? 17 : 19), class: 'chip-s' }, grp, c.s);
    if (c.kind === 'fleet') {
      c.leds = [];
      const cols = 26, rows = 3;
      for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
        c.leds.push(el('rect', { x: c.x + 14 + k * 11.3, y: c.y + 70 + r * 11, width: 7, height: 7, rx: 1.5, class: 'led' }, grp));
      }
    }
    if (c.kind === 'gpu') {
      c.gpus = [];
      for (let r = 0; r < 2; r++) for (let k = 0; k < 4; k++) {
        c.gpus.push(el('rect', { x: c.x + 14 + k * 34, y: c.y + 64 + r * 28, width: 26, height: 20, rx: 3, class: 'gpu' }, grp));
      }
    }
    if (c.kind === 'train') {
      c.steps = ['① compare attempts', '② compute update', '③ update policy'].map((s, i) =>
        el('text', { x: c.x + 14, y: c.y + 76 + i * 18, class: 'chip-step' }, grp, s));
    }
    if (c.kind === 'stack') {
      el('rect', { x: c.x + 14, y: c.y + 64, width: c.w - 28, height: 34, rx: 7, class: 'harness' }, grp);
      el('text', { x: c.x + c.w / 2, y: c.y + 85, class: 'chip-step', 'text-anchor': 'middle' }, grp, 'shared agent code');
    }
  }

  /* ------------------------------------------------------------ runtime */
  function init(root) {
    const { svg, pulses } = build(root);
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const rail = root.querySelector('.stage-rail');
    const card = root.querySelector('.stage-card');
    const insp = root.querySelector('.inspector');
    const playBtn = root.querySelector('[data-ctl=play]');
    const st = { mode: 'train', i: 0, playing: !reduce, pinned: null, hover: null, t0: performance.now(), visible: false };
    const STAGE_MS = 5600;
    const pulseEls = [];

    function stageKeys() { return MODES[st.mode].stages; }
    function renderRail() {
      rail.innerHTML = stageKeys().map((k, i) => `<li><button type="button" data-i="${i}" title="${S[k].t}"><b>${String(i + 1).padStart(2, '0')}</b><span>${S[k].t}</span><i></i></button></li>`).join('');
    }
    function applyMode() {
      (root.closest('.board-section') || root).querySelectorAll('[data-mode]')
        .forEach(b => b.setAttribute('aria-selected', b.dataset.mode === st.mode));
      const m = MODES[st.mode];
      CHIPS.forEach(c => {
        if (!c.tEl) return;
        const r = (m.relabel || {})[c.id];
        c.tEl.textContent = r ? r[0] : c.t;
        c.sEl.textContent = r ? r[1] : c.s;
      });
      CHIPS.find(c => c.id === 'train').steps.forEach(s => { s.style.display = st.mode === 'train' ? '' : 'none'; });
      TRACES.forEach(t => t.el.classList.toggle('off', (m.off || []).includes(t.id)));
      renderRail();
    }
    const scroller = root.querySelector('.board-scroll');
    function follow() {
      // On narrow screens the board scrolls sideways; keep the active stage in view.
      if (scroller.scrollWidth <= scroller.clientWidth + 4 || !st.visible) return;
      const ids = S[stageKeys()[st.i]].chips.map(id => chipById[id]);
      const cx = ids.reduce((sum, c) => sum + c.x + c.w / 2, 0) / ids.length;
      const px = (cx / W) * scroller.scrollWidth;
      scroller.scrollTo({ left: px - scroller.clientWidth / 2, behavior: reduce ? 'auto' : 'smooth' });
    }
    function setStage(i, user) {
      const keys = stageKeys();
      st.i = (i + keys.length) % keys.length;
      st.t0 = performance.now();
      if (user) { st.playing = false; syncPlay(); }
      paint();
      follow();
    }
    function activeSets() {
      if (st.pinned || st.hover) {
        const id = st.pinned || st.hover;
        const ids = id === 'mgr' ? ['mgr', 'prov', 'sess', 'task', 'rew', 'tel'] : [id];
        const traces = TRACES.filter(t => ids.includes(t.a) || ids.includes(t.b)).map(t => t.id);
        const chips = new Set(ids);
        traces.forEach(tid => { chips.add(traceById[tid].a); chips.add(traceById[tid].b); });
        return { traces: new Set(traces), chips, focus: id };
      }
      const s = S[stageKeys()[st.i]];
      return { traces: new Set(s.traces), chips: new Set(s.chips), focus: null };
    }
    function paint() {
      const a = activeSets();
      TRACES.forEach(t => { const on = a.traces.has(t.id); t.el.classList.toggle('on', on); t.halo.classList.toggle('on', on); });
      CHIPS.forEach(c => c.g.classList.toggle('on', a.chips.has(c.id) || (c.id === 'mgr' && ['prov', 'sess', 'task', 'rew', 'tel'].some(x => a.chips.has(x)))));
      svg.classList.toggle('focusing', !!a.focus);
      root.classList.toggle('inspecting', !!a.focus);
      CHIPS.forEach(c => c.g.classList.toggle('focus', c.id === a.focus));
      const k = stageKeys()[st.i], s = S[k];
      card.querySelector('.stage-kicker').textContent = `${MODES[st.mode].label} · stage ${st.i + 1} of ${stageKeys().length}`;
      card.querySelector('h3').textContent = s.t;
      card.querySelector('.stage-text').textContent = s.d;
      card.querySelector('.wire').textContent = s.w;
      card.querySelector('.wire').hidden = !s.w;
      rail.querySelectorAll('button').forEach((b, j) => {
        b.classList.toggle('on', j === st.i);
        b.classList.toggle('done', j < st.i);
      });
      rail.style.setProperty('--dur', STAGE_MS + 'ms');
      rail.classList.toggle('running', st.playing);
      // restart the progress animation
      const cur = rail.querySelector('button.on i');
      if (cur) { cur.style.animation = 'none'; void cur.offsetWidth; cur.style.animation = ''; }
      buildPulses(a.traces);
    }
    function showInfo(id) {
      const info = INFO[id];
      if (!info) return;
      insp.querySelector('.stage-kicker').textContent = st.pinned ? 'Inspector · pinned' : 'Inspector';
      insp.querySelector('h3').textContent = info[0];
      insp.querySelector('.stage-text').textContent = info[1];
      insp.querySelector('ul').innerHTML = info[2].map(f => `<li>${f}</li>`).join('');
      insp.classList.add('live');
    }
    function buildPulses(active) {
      pulses.innerHTML = '';
      pulseEls.length = 0;
      TRACES.forEach(t => {
        if (t.el.classList.contains('off')) return;
        const on = active.has(t.id);
        const n = on ? Math.max(1, Math.round(t.len / 90)) : 1;
        for (let k = 0; k < n; k++) {
          const c = el('circle', { r: on ? 3.2 : 1.8, opacity: on ? 1 : 0.35 }, pulses);
          c.style.fill = on ? TYPES[t.type].color : 'var(--sig-idle)';
          pulseEls.push({ c, t, off: (k / n) * t.len + (on ? 0 : Math.random() * t.len), speed: on ? 150 : 34 });
        }
      });
    }
    function syncPlay() {
      playBtn.classList.toggle('paused', !st.playing);
      playBtn.setAttribute('aria-label', st.playing ? 'Pause' : 'Play');
      rail.classList.toggle('running', st.playing);
    }

    // LEDs: a fleet that never sleeps
    const leds = CHIPS.filter(c => c.leds).flatMap(c => c.leds);
    const gpus = CHIPS.find(c => c.id === 'eng').gpus;
    const ledState = leds.map(() => Math.random());
    function tickLeds() {
      const k = stageKeys()[st.i];
      const busy = ['acquire', 'observe', 'act', 'verify'].includes(k) ? 0.55 : 0.32;
      for (let n = 0; n < leds.length; n++) {
        if (Math.random() < 0.07) ledState[n] = Math.random();
        const v = ledState[n];
        leds[n].setAttribute('class', 'led ' + (v < busy ? 'busy' : v < busy + 0.3 ? 'ready' : v < 0.97 ? '' : 'done'));
      }
      const hot = k === 'think' ? 0.85 : k === 'sync' ? 1 : 0.3;
      gpus.forEach(g => g.classList.toggle('hot', Math.random() < hot));
      g_trainSteps();
    }
    function g_trainSteps() {
      const steps = CHIPS.find(c => c.id === 'train').steps;
      const k = stageKeys()[st.i];
      const frac = Math.min(0.999, (performance.now() - st.t0) / STAGE_MS);
      steps.forEach((s, j) => s.classList.toggle('hot', k === 'learn' && j <= Math.floor(frac * 3)));
    }

    let last = performance.now();
    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (st.visible) {
        for (const p of pulseEls) {
          p.off = (p.off + p.speed * dt) % p.t.len;
          const pt = p.t.el.getPointAtLength(p.off);
          p.c.setAttribute('cx', pt.x.toFixed(1));
          p.c.setAttribute('cy', pt.y.toFixed(1));
        }
        if (st.playing && !st.pinned && now - st.t0 > STAGE_MS) setStage(st.i + 1);
      }
      requestAnimationFrame(frame);
    }

    // events
    root.querySelector('.mode-switch').addEventListener('click', e => {
      const b = e.target.closest('[data-mode]');
      if (!b || b.dataset.mode === st.mode) return;
      st.mode = b.dataset.mode;
      applyMode();
      setStage(0);
    });
    rail.addEventListener('click', e => { const b = e.target.closest('[data-i]'); if (b) setStage(+b.dataset.i, true); });
    root.querySelector('[data-ctl=prev]').addEventListener('click', () => setStage(st.i - 1, true));
    root.querySelector('[data-ctl=next]').addEventListener('click', () => setStage(st.i + 1, true));
    playBtn.addEventListener('click', () => { st.playing = !st.playing; st.t0 = performance.now(); st.pinned = null; syncPlay(); paint(); });
    svg.addEventListener('pointerover', e => {
      const g = e.target.closest('.chip');
      const id = g && g.dataset.id;
      if (id && id !== st.hover) { st.hover = id; showInfo(id); paint(); }
    });
    svg.addEventListener('pointerleave', () => { st.hover = null; paint(); });
    svg.addEventListener('pointerout', e => { if (!e.relatedTarget || !e.relatedTarget.closest || !e.relatedTarget.closest('.chip')) { if (st.hover) { st.hover = null; paint(); } } });
    svg.addEventListener('click', e => {
      const g = e.target.closest('.chip');
      st.pinned = g && g.dataset.id !== st.pinned ? g.dataset.id : null;
      if (st.pinned) { st.playing = false; syncPlay(); showInfo(st.pinned); }
      paint();
    });
    svg.addEventListener('keydown', e => {
      const g = e.target.closest('.chip');
      if (g && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); g.dispatchEvent(new MouseEvent('click', { bubbles: true })); }
    });
    svg.addEventListener('focusin', e => { const g = e.target.closest('.chip'); if (g) { st.hover = g.dataset.id; showInfo(st.hover); paint(); } });
    svg.addEventListener('focusout', e => { if (st.hover && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('.chip'))) { st.hover = null; paint(); } });
    new IntersectionObserver(es => es.forEach(x => { st.visible = x.isIntersecting; if (x.isIntersecting) st.t0 = performance.now() - 200; }), { threshold: 0.15 }).observe(root);

    // Scripted control for the demo video (demo_video/compose.html); the page itself never calls it.
    root.boardApi = {
      setMode(m) { if (m !== st.mode && MODES[m]) { st.mode = m; applyMode(); setStage(0); } },
      setStage(k) {
        const i = typeof k === 'number' ? k : stageKeys().indexOf(k);
        if (i < 0) return;
        st.playing = false; st.pinned = null; st.hover = null; syncPlay();
        if (i !== st.i) setStage(i); else paint();
      },
      stage: k => ({ ...S[k], chips: S[k].chips.map(id => { const c = chipById[id]; return { id, x: c.x, y: c.y, w: c.w, h: c.h }; }) }),
      keys: () => stageKeys().slice(),
      size: { W, H },
    };

    // The SVG is as wide as the viewport height allows (see .board-main in styles.css).
    const main = root.querySelector('.board-main');
    const sec = root.closest('.board-section');

    function fit() {
      if (sec) {
        // the board owns a whole screen: give the circuit every pixel the heading leaves it
        const cs = getComputedStyle(sec);
        const head = sec.querySelector('.board-head');
        const chrome = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)   // padding already clears the fixed header
          + (head ? head.getBoundingClientRect().height + parseFloat(getComputedStyle(head).marginBottom) : 0)
          + 36;                                                                   // the board card's own padding
        root.style.setProperty('--fold', Math.round(chrome) + 'px');
        return;
      }
      root.style.setProperty('--fold', Math.round(main.getBoundingClientRect().top + window.scrollY + 34) + 'px');
    }
    fit();
    window.addEventListener('resize', fit);
    if (document.fonts) document.fonts.ready.then(fit);

    applyMode();
    syncPlay();
    paint();
    setInterval(() => { if (st.visible) tickLeds(); }, 140);
    requestAnimationFrame(frame);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const root = document.getElementById('board');
    if (root) init(root);
  });
})();
