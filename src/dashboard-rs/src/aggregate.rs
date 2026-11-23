use regex::Regex;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;
pub const UNGROUPED: &str = "__ungrouped__";
pub fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("")
}
pub fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.)
}
fn present(v: &Value) -> bool {
    !v.is_null() && v != "" && v != false
}
pub fn success(s: f64) -> bool {
    static THRESHOLD: LazyLock<f64> = LazyLock::new(|| {
        std::env::var("MAI_GYM_SUCCESS_SCORE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.8)
    });
    s >= *THRESHOLD - 1e-9
}

pub fn rollout_seg(s: &str) -> bool {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(eval_)?r(\d+|_misc)$").unwrap());
    RE.is_match(s)
}
pub fn run_key(e: &Value) -> String {
    if text(e, "train_id").is_empty() {
        UNGROUPED.into()
    } else if text(e, "model_slug").is_empty() {
        text(e, "train_id").into()
    } else {
        format!("{}/{}", text(e, "model_slug"), text(e, "train_id"))
    }
}
pub fn verifier(reason: &str) -> Option<Value> {
    let mut v: Value = serde_json::from_str(reason).ok()?;
    let valid = v["evaluation_contract"] == "railway_delivery_v130"
        && (v["result_schema_version"] == 2
            || (v["result_schema_version"] == 3
                && ["railway_relative_v1.5.0", "railway12306_70_v1.5.0"]
                    .contains(&text(&v, "verifier"))
                && v["evaluation_score_policy"] == "verifier_result"))
        && ["pass", "fail", "insufficient_evidence", "error"].contains(&text(&v, "status"));
    if !valid {
        return None;
    }
    for k in ["quality", "score_valid", "training_eligible"] {
        v.as_object_mut()?.remove(k);
    }
    Some(v)
}
pub fn normalize(mut e: Value) -> Value {
    for key in [
        "evaluation_terminal",
        "verifier_status",
        "verifier_score",
        "verifier_result",
    ] {
        if let Some(object) = e.as_object_mut() {
            object.remove(key);
        }
    }
    let parts: Vec<_> = text(&e, "id").split('/').collect();
    let slug = if parts.len() >= 3 && rollout_seg(parts[2]) {
        json!(parts[0])
    } else {
        Value::Null
    };
    e["model_slug"] = slug;
    if let Some(v) = verifier(text(&e, "reason")) {
        let score = if ["pass", "fail"].contains(&text(&v, "status"))
            && v["score"]
                .as_f64()
                .is_some_and(|x| (0.0..=1.0).contains(&x))
        {
            v["score"].clone()
        } else {
            Value::Null
        };
        e["score"] = score.clone();
        e["verifier_score"] = score;
        e["verifier_status"] = v["status"].clone();
        e["verifier_result"] = v;
        e["evaluation_terminal"] = json!(true);
    }
    let marker = text(&e, "reason")
        .trim()
        .lines()
        .next()
        .unwrap_or("")
        .to_lowercase();
    let method = if marker == "built_in+llm_judge" {
        json!("both")
    } else if marker == "llm_judge" {
        json!("llm_judge")
    } else if e["score"].is_number() {
        json!("built_in")
    } else {
        Value::Null
    };
    let judge = if method == "llm_judge" && e["judge_reward"].is_null() {
        e["score"].clone()
    } else if method == "both" || method == "llm_judge" {
        e["judge_reward"].clone()
    } else {
        Value::Null
    };
    e["built_in_score"] = if method == "llm_judge" {
        Value::Null
    } else {
        e["score"].clone()
    };
    e["llm_judge_score"] = judge;
    e["evaluation_method"] = method;
    let timeout = text(&e, "error")
        .to_lowercase()
        .contains("timeout[wall-clock]:");
    e["timeout_kind"] = if timeout {
        json!("explicit")
    } else {
        Value::Null
    };
    e["timeout"] = json!(timeout);
    e["timeout_inferred"] = json!(false);
    e
}
pub fn parse_result(s: &str) -> (Option<f64>, Option<String>) {
    let mut lines = s.lines();
    let first = lines.next().unwrap_or("").trim();
    let first = if first.to_lowercase().starts_with("score:") {
        first[6..].trim()
    } else {
        first
    };
    let rest = lines.collect::<Vec<_>>().join("\n");
    let rest = rest.trim();
    let rest = if rest.to_lowercase().starts_with("reason:") {
        rest[7..].trim()
    } else {
        rest
    };
    (
        first.parse::<f64>().ok().filter(|v| v.is_finite()),
        (!rest.is_empty()).then(|| rest.into()),
    )
}
#[derive(Default)]
struct Name {
    rid: Option<i64>,
    eval: bool,
    group: Option<i64>,
    sample: Option<i64>,
    retry: Option<i64>,
}
fn name(s: &str) -> Name {
    static R: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|_)(eval_)?r(\d+)").unwrap());
    static G: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:^|_)g(\d+)_s(\d+)(?:_a(\d+))?").unwrap());
    static S: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|_)s(\d+)(?:_a(\d+))?").unwrap());
    let mut n = Name {
        eval: s.split('_').any(|x| x == "eval"),
        ..Default::default()
    };
    if let Some(c) = R.captures(s) {
        n.rid = c[2].parse().ok();
        n.eval |= c.get(1).is_some();
    }
    if let Some(c) = G.captures(s) {
        n.group = c[1].parse().ok();
        n.sample = c[2].parse().ok();
        n.retry = Some(c.get(3).and_then(|v| v.as_str().parse().ok()).unwrap_or(0));
    } else if let Some(c) = S.captures(s) {
        n.sample = c[1].parse().ok();
        n.retry = Some(c.get(2).and_then(|v| v.as_str().parse().ok()).unwrap_or(0));
    }
    n
}
pub fn attempt_ts(e: &Value) -> f64 {
    let t = num(e, "created_at");
    if t != 0. { t } else { num(e, "mtime") }
}
#[derive(Default, Clone)]
struct Stats {
    count: u64,
    scored: u64,
    passed: u64,
    sum: f64,
    failed: u64,
    timeouts: u64,
    inferred: u64,
    builtin: u64,
    builtin_pass: u64,
    builtin_sum: f64,
    judge: u64,
    judge_pass: u64,
    judge_sum: f64,
    last: f64,
    verifier: BTreeMap<String, u64>,
}
impl Stats {
    fn add(&mut self, e: &Value) {
        self.count += 1;
        if let Some(s) = e["score"].as_f64() {
            self.scored += 1;
            self.sum += s;
            self.passed += u64::from(success(s));
        }
        if present(&e["error"]) {
            self.failed += 1;
            self.timeouts += u64::from(e["timeout"] == true);
            self.inferred += u64::from(e["timeout_inferred"] == true);
        }
        if let Some(s) = e["built_in_score"].as_f64() {
            self.builtin += 1;
            self.builtin_sum += s;
            self.builtin_pass += u64::from(success(s));
        }
        if let Some(s) = e["llm_judge_score"].as_f64() {
            self.judge += 1;
            self.judge_sum += s;
            self.judge_pass += u64::from(success(s));
        }
        self.last = self.last.max(num(e, "mtime"));
        if e["evaluation_terminal"] == true {
            for k in [
                "verifier_completed",
                "verifier_unscored",
                "verifier_scored",
                "verifier_passed",
                "verifier_failed",
                "verifier_insufficient",
                "verifier_errors",
            ] {
                self.verifier.entry(k.into()).or_default();
            }
            *self.verifier.get_mut("verifier_completed").unwrap() += 1;
            *self
                .verifier
                .get_mut(if e["verifier_score"].is_null() {
                    "verifier_unscored"
                } else {
                    "verifier_scored"
                })
                .unwrap() += 1;
            let k = match text(e, "verifier_status") {
                "pass" => "verifier_passed",
                "fail" => "verifier_failed",
                "insufficient_evidence" => "verifier_insufficient",
                _ => "verifier_errors",
            };
            *self.verifier.get_mut(k).unwrap() += 1;
        }
    }
    fn json(&self) -> Value {
        let mean = |s: f64, n: u64| {
            if n == 0 {
                Value::Null
            } else {
                json!(s / n as f64)
            }
        };
        let mut v = json!({"count":self.count,"scored":self.scored,"passed":self.passed,"failed":self.failed,"timeouts":self.timeouts,"inferred_timeouts":self.inferred,"mean_score":mean(self.sum,self.scored),"built_in_scored":self.builtin,"built_in_passed":self.builtin_pass,"built_in_mean_score":mean(self.builtin_sum,self.builtin),"judge_scored":self.judge,"judge_passed":self.judge_pass,"judge_mean_score":mean(self.judge_sum,self.judge),"last_mtime":self.last});
        for (k, n) in &self.verifier {
            v[k] = json!(n);
        }
        v
    }
}
pub fn tasks(items: &[Value]) -> Vec<Value> {
    task_refs(items.iter())
}
fn task_refs<'a>(items: impl IntoIterator<Item = &'a Value>) -> Vec<Value> {
    let mut groups: BTreeMap<String, Stats> = BTreeMap::new();
    for e in items {
        let t = if text(e, "task_id").is_empty() {
            "(no task)"
        } else {
            text(e, "task_id")
        };
        groups.entry(t.into()).or_default().add(e);
    }
    let mut out: Vec<_> = groups
        .into_iter()
        .map(|(t, s)| {
            let mut v = s.json();
            v["task_id"] = json!(t);
            v
        })
        .collect();
    out.sort_by(|a, b| num(b, "last_mtime").total_cmp(&num(a, "last_mtime")));
    out
}
fn assign(items: &[Value]) -> (Vec<String>, BTreeMap<String, Value>) {
    let package = items.iter().any(|e| present(&e["eval_pkg"]));
    let mut ordered: Vec<_> = (0..items.len()).collect();
    ordered.sort_by(|&a, &b| attempt_ts(&items[a]).total_cmp(&attempt_ts(&items[b])));
    let mut keys = vec![String::new(); items.len()];
    let mut buckets: BTreeMap<String, Value> = BTreeMap::new();
    let mut cur = String::new();
    let mut cur_eval = false;
    let mut sigs = HashSet::new();
    let mut prev = None;
    for i in ordered {
        let e = &items[i];
        let n = name(text(e, "attempt"));
        let rid = e["rollout_id"].as_i64().or(n.rid);
        let eval = e["evaluation"].as_bool().unwrap_or(n.eval);
        let ts = attempt_ts(e);
        let inferred = !package && rid.is_none();
        let key = if package {
            format!(
                "pkg_{}",
                if present(&e["eval_pkg"]) {
                    text(e, "eval_pkg")
                } else {
                    "(unlisted)"
                }
            )
        } else if let Some(r) = rid {
            format!("{}{:05}", if eval { "eval_r" } else { "r" }, r)
        } else {
            let sig = (n.group, n.sample, n.retry);
            let has = n.group.is_some() || n.sample.is_some();
            if cur.is_empty()
                || cur_eval != eval
                || (has && sigs.contains(&sig))
                || (!has && prev.is_some_and(|p: f64| ts - p > 1800.))
            {
                cur = format!("t{}", ts as i64);
                cur_eval = eval;
                sigs.clear();
            }
            if has {
                sigs.insert(sig);
            }
            cur.clone()
        };
        let b=buckets.entry(key.clone()).or_insert_with(||json!({"rollout_key":key,"t_start":ts,"t_end":ts,"is_eval":eval,"inferred":inferred,"rollout_id":if package{None}else{rid}}));
        if ts != 0. {
            let start = num(b, "t_start");
            b["t_start"] = json!(if start == 0. { ts } else { start.min(ts) });
            b["t_end"] = json!(num(b, "t_end").max(ts));
        }
        if package {
            b["is_eval"] = json!(b["is_eval"] == true || eval);
            b["label"] = json!(key.strip_prefix("pkg_").unwrap());
        }
        keys[i] = key;
        prev = Some(ts);
    }
    let mut order: Vec<_> = buckets.keys().cloned().collect();
    order.sort_by(|a, b| num(&buckets[a], "t_start").total_cmp(&num(&buckets[b], "t_start")));
    let (mut en, mut tn) = (0, 0);
    for k in order {
        let b = buckets.get_mut(&k).unwrap();
        if !package {
            b["label"] = if let Some(r) = b["rollout_id"].as_i64() {
                json!(if b["is_eval"] == true {
                    format!("Eval · iter {r}")
                } else {
                    format!("Rollout {r}")
                })
            } else if b["is_eval"] == true {
                en += 1;
                json!(format!("Eval ~{en}"))
            } else {
                tn += 1;
                json!(format!("Rollout ~{tn}"))
            };
        }
    }
    (keys, buckets)
}
fn env_float(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn infer_timeouts(items: &mut [Value], keys: &[String]) {
    let limit = env_float("MAI_GYM_TRAJ_TIMEOUT_LIMIT_S", 10800.);
    let tolerance = env_float("MAI_GYM_TRAJ_TIMEOUT_TOLERANCE_S", 900.);
    let cluster = env_float("MAI_GYM_TRAJ_TIMEOUT_CLUSTER_S", 180.);
    let minimum = env_float("MAI_GYM_TRAJ_TIMEOUT_CLUSTER_MIN", 3.).max(2.) as usize;
    let mut starts: HashMap<&str, f64> = HashMap::new();
    let mut generic: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut inferred = HashSet::new();
    for (i, e) in items.iter().enumerate() {
        let ts = attempt_ts(e);
        if ts > 0. {
            starts
                .entry(&keys[i])
                .and_modify(|t| *t = t.min(ts))
                .or_insert(ts);
        }
        if text(e, "error")
            .trim()
            .eq_ignore_ascii_case("trajectory released without evaluation (abandoned)")
            && num(e, "error_mtime") > 0.
        {
            generic.entry(&keys[i]).or_default().push(i);
            if (num(e, "error_mtime") - ts - limit).abs() <= tolerance {
                inferred.insert(i);
            }
        }
    }
    for (k, mut ids) in generic {
        let Some(start) = starts.get(k) else { continue };
        ids.sort_by(|&a, &b| {
            num(&items[a], "error_mtime").total_cmp(&num(&items[b], "error_mtime"))
        });
        let mut i = 0;
        while i < ids.len() {
            let mut j = i + 1;
            while j < ids.len()
                && num(&items[ids[j]], "error_mtime") - num(&items[ids[i]], "error_mtime")
                    <= cluster
            {
                j += 1;
            }
            let mid =
                (num(&items[ids[i]], "error_mtime") + num(&items[ids[j - 1]], "error_mtime")) / 2.;
            if j - i >= minimum && (mid - start - limit).abs() <= tolerance {
                inferred.extend(&ids[i..j]);
            }
            i = j;
        }
    }
    for i in inferred {
        items[i]["timeout_kind"] = json!("inferred");
        items[i]["timeout"] = json!(true);
        items[i]["timeout_inferred"] = json!(true);
    }
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RunView {
    pub run: Value,
    pub items: Vec<Value>,
    pub rollouts: Vec<Value>,
    pub tasks: Vec<Value>,
    pub tasks_by_rollout: BTreeMap<String, Vec<Value>>,
    pub membership: BTreeMap<String, Vec<String>>,
}
pub fn build_run(mut items: Vec<Value>, root: &str, trained: &BTreeMap<String, Value>) -> RunView {
    for e in &mut items {
        if e.get("evaluation_method").is_none() {
            *e = normalize(e.take());
        }
    }
    let (keys, mut buckets) = assign(&items);
    infer_timeouts(&mut items, &keys);
    let mut stats = Stats::default();
    for e in &items {
        stats.add(e);
    }
    let mut run = stats.json();
    let first = items.first().cloned().unwrap_or(json!({}));
    let rk = run_key(&first);
    run["run_key"] = json!(rk);
    for k in [
        "train_id",
        "model_slug",
        "suite",
        "benchmark_model",
        "benchmark_run_id",
    ] {
        run[k] = items
            .iter()
            .find(|e| !e[k].is_null())
            .map(|e| e[k].clone())
            .unwrap_or(Value::Null);
    }
    run["oss_path"] = if root.is_empty() {
        Value::Null
    } else if rk == UNGROUPED {
        json!(root.trim_end_matches('/'))
    } else {
        json!(format!("{}/{}", root.trim_end_matches('/'), rk))
    };
    let package = items.iter().any(|e| present(&e["eval_pkg"]));
    let all_eval = text(&run, "train_id").to_lowercase().contains("eval")
        && !buckets.values().any(|b| b["is_eval"] == true);
    let mut passes: BTreeMap<String, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for (i, e) in items.iter().enumerate() {
        let k = &keys[i];
        if package || all_eval || buckets[k]["is_eval"] == true {
            let pass = if package {
                format!(
                    "pass{:05}",
                    e["rollout_id"]
                        .as_i64()
                        .or(name(text(e, "attempt")).rid)
                        .unwrap_or(0)
                )
            } else {
                k.clone()
            };
            passes.entry(pass).or_default().insert(
                if text(e, "task_id").is_empty() {
                    "(no task)".into()
                } else {
                    text(e, "task_id").into()
                },
                e["score"].as_f64(),
            );
        }
    }
    for k in [
        "pass_at_k",
        "avg_at_k",
        "n_passes",
        "n_tasks_eval",
        "n_scored_eval",
        "n_passed_eval",
    ] {
        run[k] = Value::Null;
    }
    if !passes.is_empty() {
        let mut sr = 0.;
        let mut all: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for p in passes.values() {
            let scored: Vec<_> = p.values().flatten().copied().collect();
            sr += if scored.is_empty() {
                0.
            } else {
                scored.iter().filter(|&&s| success(s)).count() as f64 / scored.len() as f64
            };
            for (t, s) in p {
                let v = all.entry(t.clone()).or_default();
                if let Some(s) = s {
                    v.push(*s);
                }
            }
        }
        let scored = all.values().filter(|v| !v.is_empty()).count();
        let passed = all
            .values()
            .filter(|v| v.iter().any(|&s| success(s)))
            .count();
        run["pass_at_k"] = json!(if scored == 0 {
            0.
        } else {
            passed as f64 / scored as f64
        });
        run["avg_at_k"] = json!(sr / passes.len() as f64);
        run["n_passes"] = json!(passes.len());
        run["n_tasks_eval"] = json!(all.len());
        run["n_scored_eval"] = json!(scored);
        run["n_passed_eval"] = json!(passed);
    }
    for (i, e) in items.iter_mut().enumerate() {
        e["rollout_key"] = json!(keys[i]);
    }
    let mut grouped: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for (i, e) in items.iter().enumerate() {
        grouped.entry(keys[i].clone()).or_default().push(e);
    }
    let mut out = Vec::new();
    for (k, b) in &mut buckets {
        let group = &grouped[k];
        let mut s = Stats::default();
        for e in group {
            s.add(e);
        }
        let mut v = s.json();
        v.as_object_mut().unwrap().remove("last_mtime");
        v.as_object_mut()
            .unwrap()
            .extend(b.as_object().unwrap().clone());
        v["category"] = json!(if v["is_eval"] == true {
            "eval"
        } else {
            "rollout"
        });
        out.push(v.clone());
        if v["is_eval"] != true
            && let Some(m) = trained.get(k)
        {
            let kept: HashSet<_> = m["tasks"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|t| t["kept"] == true)
                .map(|t| text(t, "task_seg"))
                .collect();
            let filtered: Vec<_> = group
                .iter()
                .filter(|e| kept.contains(text(e, "task_id")) && !present(&e["error"]))
                .copied()
                .collect();
            let mut s = Stats::default();
            for e in &filtered {
                s.add(e);
            }
            s.passed = filtered.iter().filter(|e| num(e, "score") > 0.).count() as u64;
            let mut sb = s.json();
            for k in [
                "last_mtime",
                "verifier_completed",
                "verifier_unscored",
                "verifier_scored",
                "verifier_passed",
                "verifier_failed",
                "verifier_insufficient",
                "verifier_errors",
            ] {
                sb.as_object_mut().unwrap().remove(k);
            }
            let sk = format!("{k}__trained");
            sb["rollout_key"] = json!(sk);
            sb["label"] = json!(format!("{} - Trained", text(&v, "label")));
            sb["category"] = json!("trained");
            sb["t_start"] = v["t_start"].clone();
            sb["t_end"] = json!(num(&v, "t_end") - 0.001);
            sb["is_eval"] = json!(false);
            sb["inferred"] = json!(false);
            let mut summary = json!({});
            for k in [
                "rollout_id",
                "degrade_level",
                "advantage_estimator",
                "num_groups",
                "num_trainable_tasks",
                "num_kept_tasks",
                "num_trained_trajectories",
                "created_at",
            ] {
                summary[k] = m[k].clone();
            }
            summary["tasks"] = json!(
                m["tasks"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|t| {
                        let mut v = json!({});
                        for k in [
                            "task_id",
                            "task_seg",
                            "pass_rate",
                            "n_valid",
                            "n_pass",
                            "kept",
                            "degraded",
                        ] {
                            v[k] = t[k].clone();
                        }
                        v
                    })
                    .collect::<Vec<_>>()
            );
            sb["trained"] = summary;
            out.push(sb);
            grouped.insert(sk, filtered);
        }
    }
    out.sort_by(|a, b| num(b, "t_end").total_cmp(&num(a, "t_end")));
    let tasks = tasks(&items);
    let tasks_by_rollout = grouped
        .iter()
        .map(|(k, v)| (k.clone(), task_refs(v.iter().copied())))
        .collect();
    let membership = grouped
        .into_iter()
        .map(|(k, v)| (k, v.iter().map(|e| text(e, "id").to_owned()).collect()))
        .collect();
    items.sort_by(|a, b| num(b, "mtime").total_cmp(&num(a, "mtime")));
    RunView {
        run,
        items,
        rollouts: out,
        tasks,
        tasks_by_rollout,
        membership,
    }
}
pub fn benchmarks(runs: Vec<(String, String, Value)>) -> Value {
    let mut slots: BTreeMap<(String, String), BTreeMap<String, BTreeMap<String, Value>>> =
        BTreeMap::new();
    for (m, label, r) in runs {
        if r["pass_at_k"].is_null() || text(&r, "suite").is_empty() {
            continue;
        }
        let model = text(&r, "benchmark_model").to_string();
        if model.is_empty() {
            continue;
        }
        let rid = if text(&r, "benchmark_run_id").is_empty() {
            "(default)"
        } else {
            text(&r, "benchmark_run_id")
        };
        let per = slots
            .entry((text(&r, "suite").into(), model))
            .or_default()
            .entry(m.clone())
            .or_default();
        if per
            .get(rid)
            .is_some_and(|v| num(v, "last_mtime") >= num(&r, "last_mtime"))
        {
            continue;
        }
        per.insert(rid.into(),json!({"manager_key":m,"manager_label":label,"run_key":r["run_key"],"run_id":rid,"pass_at_k":r["pass_at_k"],"avg_at_k":r["avg_at_k"],"n_passes":r["n_passes"],"n_tasks":r["n_tasks_eval"],"last_mtime":r["last_mtime"]}));
    }
    let mut out = json!({});
    for ((suite, model), mgrs) in slots {
        let mut cells = json!({});
        let mut ids = BTreeSet::new();
        let mut latest = Value::Null;
        let mut time = 0.;
        for (m, runs) in mgrs {
            ids.extend(runs.keys().cloned());
            let v = runs
                .values()
                .max_by(|a, b| num(a, "last_mtime").total_cmp(&num(b, "last_mtime")))
                .unwrap();
            if num(v, "last_mtime") > time {
                time = num(v, "last_mtime");
                latest = v["run_id"].clone();
            }
            cells[&m] = v.clone();
        }
        if out[&suite].is_null() {
            out[&suite] = json!({"suite":suite,"models":[]});
        }
        out[&suite]["models"].as_array_mut().unwrap().push(json!({"model":model,"cells":cells,"runs":ids,"latest_run_id":latest,"last_mtime":time}));
    }
    for v in out.as_object_mut().unwrap().values_mut() {
        v["models"]
            .as_array_mut()
            .unwrap()
            .sort_by(|a, b| num(b, "last_mtime").total_cmp(&num(a, "last_mtime")));
    }
    out
}
