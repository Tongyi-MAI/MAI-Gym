/* In-browser port of mocked_envs/pipeline/verifier.py and jsonpath.py.
   Scores a finished attempt against a task's goal criteria, exactly as the
   training pipeline does, so visitors see the reward an agent would get. */
(function () {
  'use strict';
  const MISSING = Symbol('missing');
  const TOKEN = /\[\s*(-?\d+|last|\*)\s*\]|\[\s*\?([^=\]]+)=([^\]]*)\]|\.?([^.[\]]+)/y;
  const NUM = /-?\d+(?:\.\d+)?/;

  function parsePath(path) {
    const steps = [];
    path = path.trim();
    TOKEN.lastIndex = 0;
    while (TOKEN.lastIndex < path.length) {
      const start = TOKEN.lastIndex;
      const m = TOKEN.exec(path);
      if (!m || m.index !== start) throw new Error(`bad path ${path}`);
      if (m[1] !== undefined) steps.push(m[1] === '*' ? ['star'] : ['index', m[1]]);
      else if (m[2] !== undefined) steps.push(['filter', [m[2].trim(), m[3].trim()]]);
      else steps.push(['name', m[4]]);
    }
    return steps;
  }

  function coerceEq(actual, literal) {
    if (actual === MISSING) return false;
    if (typeof actual === 'boolean') return String(actual) === literal.toLowerCase();
    if (typeof actual === 'number') return actual === parseFloat(literal);
    return String(actual) === literal;
  }

  function one(cur, kind, arg) {
    if (kind === 'name') {
      return cur !== null && typeof cur === 'object' && !Array.isArray(cur) && arg in cur ? cur[arg] : MISSING;
    }
    if (kind === 'index') {
      if (!Array.isArray(cur)) return MISSING;
      if (arg === 'last' || arg === '-1') return cur.length ? cur[cur.length - 1] : MISSING;
      let i = parseInt(arg, 10);
      if (i < -cur.length || i >= cur.length) return MISSING;
      if (i < 0) i += cur.length;
      return cur[i];
    }
    if (kind === 'filter') {
      if (!Array.isArray(cur)) return MISSING;
      return cur.filter(e => coerceEq(resolve(e, arg[0]), arg[1]));
    }
    throw new Error(kind);
  }

  function resolve(obj, path) {
    if (path === '' || path === '$') return obj;
    let cur = obj, fanned = false;
    for (const [kind, arg] of parsePath(path)) {
      if (kind === 'star') {
        if (!Array.isArray(cur)) return MISSING;
        fanned = true;
        continue;
      }
      cur = fanned ? cur.map(e => one(e, kind, arg)).filter(v => v !== MISSING) : one(cur, kind, arg);
      if (cur === MISSING) return MISSING;
    }
    return cur;
  }

  const asList = v => (v === MISSING || v === null || v === undefined) ? [] : Array.isArray(v) ? v : [v];
  function num(v) {
    if (typeof v === 'boolean') return null;
    if (typeof v === 'number') return v;
    if (typeof v === 'string') { const m = v.match(NUM); return m ? parseFloat(m[0]) : null; }
    return null;
  }
  function freeze(v) {
    if (Array.isArray(v)) return '[' + v.map(freeze).join(',') + ']';
    if (v && typeof v === 'object') return '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + freeze(v[k])).join(',') + '}';
    return JSON.stringify(v);
  }
  function project(items, paths) {
    return asList(items).map(it => {
      if (!paths) return it;
      return paths.map(p => { const v = resolve(it, p); return v === MISSING ? null : v; });
    });
  }
  function deepEq(a, b) { return freeze(a) === freeze(b); }

  function apply(op, actual, crit) {
    const exp = crit.value;
    if (op === 'exists') return actual !== MISSING;
    if (op === 'absent') return actual === MISSING;
    if (actual === MISSING && !['ne', 'not_in', 'not_contains', 'none_where'].includes(op)) return false;
    switch (op) {
      case 'eq':
        if (typeof actual === 'string' && typeof exp === 'string') return actual.trim() === exp.trim();
        return deepEq(actual, exp);
      case 'ne': return !deepEq(actual, exp);
      case 'lt': case 'lte': case 'gt': case 'gte': {
        const a = num(actual), e = num(exp);
        if (a === null || e === null) return false;
        return { lt: a < e, lte: a <= e, gt: a > e, gte: a >= e }[op];
      }
      case 'approx': {
        const a = num(actual), e = num(exp);
        return a !== null && e !== null && Math.abs(a - e) <= (crit.tol ?? 1e-6);
      }
      case 'in': return asList(exp).some(x => deepEq(x, actual));
      case 'not_in': return !asList(exp).some(x => deepEq(x, actual));
      case 'contains':
        if (typeof actual === 'string') return actual.includes(String(exp));
        return asList(actual).some(x => deepEq(x, exp));
      case 'not_contains':
        if (typeof actual === 'string') return !actual.includes(String(exp));
        return !asList(actual).some(x => deepEq(x, exp));
      case 'regex': return new RegExp(pyRegex(String(exp))).test(String(actual));
      case 'len_eq': case 'len_gte': case 'len_lte': {
        const e = parseInt(exp, 10);
        const n = typeof actual === 'string' || Array.isArray(actual) ? actual.length
          : actual && typeof actual === 'object' ? Object.keys(actual).length : 0;
        return { len_eq: n === e, len_gte: n >= e, len_lte: n <= e }[op];
      }
      case 'set_eq': {
        const a = new Set(project(actual, crit.project).map(freeze));
        const e = new Set(project(exp, crit.project).map(freeze));
        return a.size === e.size && [...a].every(x => e.has(x));
      }
      case 'multiset_eq': {
        const a = project(actual, crit.project).map(freeze).sort();
        const e = project(exp, crit.project).map(freeze).sort();
        return a.length === e.length && a.every((x, i) => x === e[i]);
      }
    }
    if (op.endsWith('_where')) {
      const sub = { name: crit.name + '.where', source: 'state', ...crit.where };
      const items = asList(actual);
      const hits = items.filter(x => evalOne(sub, x, {}).passed);
      if (op === 'count_where') return hits.length === parseInt(exp, 10);
      if (op === 'any_where') return hits.length > 0;
      if (op === 'all_where') return hits.length === items.length;
      if (op === 'none_where') return hits.length === 0;
    }
    throw new Error(`unimplemented op: ${op}`);
  }

  // Python's (?i) inline flag is the only construct the task files use that JS lacks.
  function pyRegex(src) {
    return src.startsWith('(?i)') ? new RegExp(src.slice(4), 'i') : src;
  }

  function evalOne(crit, root, ctx) {
    const source = crit.source || 'state';
    let actual;
    if (source === 'answer') {
      actual = ctx.answer;
      if (actual === null || actual === undefined) actual = MISSING;
    } else {
      actual = crit.path ? resolve(root, crit.path) : root;
    }
    if (crit.extract === 'number' && actual !== MISSING) {
      const n = num(actual);
      actual = n === null ? MISSING : n;
    }
    let ok;
    try { ok = apply(crit.op, actual, crit); }
    catch (err) { ok = false; actual = `<verifier error: ${err.message}>`; }
    return {
      name: crit.name, op: crit.op, path: crit.path || '', source,
      expected: crit.value, actual: actual === MISSING ? null : actual,
      passed: Boolean(ok), critical: crit.critical !== false,
      weight: crit.weight ?? 1, hint: crit.hint || '',
    };
  }

  function verify(task, state, trajectory) {
    const answer = (state.meta || {}).answer;
    const ctx = { state, trajectory, answer };
    const criteria = task.goal.map(c => evalOne(c, ctx[c.source || 'state'] || {}, ctx));
    const total = criteria.reduce((s, c) => s + c.weight, 0) || 1;
    const score = criteria.filter(c => c.passed).reduce((s, c) => s + c.weight, 0) / total;
    const success = criteria.filter(c => c.critical).every(c => c.passed);
    return { success, score: Math.round(score * 1e4) / 1e4, criteria };
  }

  window.MaiVerifier = { verify, resolve, MISSING };
})();
