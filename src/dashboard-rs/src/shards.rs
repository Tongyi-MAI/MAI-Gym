use crate::{
    aggregate::{self, num, text},
    db,
};
use anyhow::Result;
use regex::Regex;
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::LazyLock,
};

pub const PREFIX: &str = "__shards__:";

pub fn shard_name(name: &str) -> Option<(String, u64)> {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:^|_)(?:s|shard_?)(\d+)").unwrap());
    let mut matches = RE.captures_iter(name).filter(|cap| {
        let end = cap.get(0).unwrap().end();
        end == name.len() || name.as_bytes()[end] == b'_'
    });
    let cap = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let matched = cap.get(0)?;
    let shard = cap[1].parse().ok()?;
    let left = &name[..matched.start()];
    let right = &name[matched.end()..];
    let right = right.strip_prefix('_').unwrap_or(right);
    let base = match (left.is_empty(), right.is_empty()) {
        (false, false) => format!("{left}_{right}"),
        (false, true) => left.into(),
        (true, false) => right.into(),
        (true, true) => return None,
    };
    Some((base, shard))
}

fn identity(run: &Value) -> Option<(String, u64)> {
    let train = text(run, "train_id");
    if text(run, "suite").is_empty() && !train.to_lowercase().contains("eval") {
        return None;
    }
    let (base, index) = shard_name(train)?;
    let slug = text(run, "model_slug");
    let key = if slug.is_empty() {
        base
    } else {
        format!("{slug}/{base}")
    };
    let tag = text(run, "benchmark_run_id");
    let tag = if let Some((base, tag_index)) = shard_name(tag) {
        if index != tag_index {
            return None;
        }
        base
    } else {
        tag.into()
    };
    Some((
        format!(
            "{PREFIX}{}",
            json!([key, run["suite"], run["benchmark_model"], tag])
        ),
        index,
    ))
}

fn published(c: &Connection, m: &str) -> Result<Vec<Value>> {
    db::values(c, "SELECT data FROM views WHERE m=? AND kind='run'", &[&m])
}

pub fn merge_stats(rows: &[Value]) -> Value {
    let mut merged = rows.first().cloned().unwrap_or(json!({}));
    for field in [
        "count",
        "scored",
        "passed",
        "failed",
        "timeouts",
        "inferred_timeouts",
        "built_in_scored",
        "built_in_passed",
        "judge_scored",
        "judge_passed",
        "verifier_completed",
        "verifier_unscored",
        "verifier_scored",
        "verifier_passed",
        "verifier_failed",
        "verifier_insufficient",
        "verifier_errors",
    ] {
        if rows.iter().any(|r| r[field].is_number()) {
            merged[field] = json!(
                rows.iter()
                    .map(|r| r[field].as_u64().unwrap_or(0))
                    .sum::<u64>()
            );
        }
    }
    for (mean, count) in [
        ("mean_score", "scored"),
        ("built_in_mean_score", "built_in_scored"),
        ("judge_mean_score", "judge_scored"),
    ] {
        let denominator = num(&merged, count);
        merged[mean] = if denominator == 0. {
            Value::Null
        } else {
            json!(
                rows.iter()
                    .map(|r| num(r, mean) * num(r, count))
                    .sum::<f64>()
                    / denominator
            )
        };
    }
    for field in ["last_mtime", "t_end"] {
        if rows.iter().any(|r| r[field].is_number()) {
            merged[field] = json!(rows.iter().map(|r| num(r, field)).fold(0., f64::max));
        }
    }
    if let Some(start) = rows
        .iter()
        .map(|r| num(r, "t_start"))
        .filter(|t| *t > 0.)
        .min_by(f64::total_cmp)
    {
        merged["t_start"] = json!(start);
    }
    merged
}

pub fn member_keys(rows: &[Value]) -> String {
    json!(rows.iter().map(|r| text(r, "run_key")).collect::<Vec<_>>()).to_string()
}

fn merged_run(key: &str, mut members: Vec<Value>) -> Result<Value> {
    members.sort_by_key(|r| identity(r).unwrap().1);
    let mut run = merge_stats(&members);
    let descriptor: Value = serde_json::from_str(key.strip_prefix(PREFIX).unwrap())?;
    run["run_key"] = json!(key);
    run["train_id"] = json!(descriptor[0].as_str().unwrap().rsplit('/').next().unwrap());
    run["benchmark_run_id"] = descriptor[3].clone();
    run["oss_path"] = Value::Null;
    run["shard_count"] = json!(members.len());
    run["shard_indices"] = json!(
        members
            .iter()
            .map(|r| identity(r).unwrap().1)
            .collect::<Vec<_>>()
    );
    for field in [
        "pass_at_k",
        "avg_at_k",
        "n_passes",
        "n_tasks_eval",
        "n_scored_eval",
        "n_passed_eval",
    ] {
        run[field] = Value::Null;
    }
    run["shards"] = json!(members);
    Ok(run)
}

pub fn runs(c: &Connection, m: &str) -> Result<Vec<Value>> {
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut out = Vec::new();
    for run in published(c, m)? {
        if let Some((key, _)) = identity(&run) {
            groups.entry(key).or_default().push(run);
        } else {
            out.push(run);
        }
    }
    for (key, members) in groups {
        let indices: BTreeSet<_> = members.iter().map(|r| identity(r).unwrap().1).collect();
        if members.len() < 2 || indices.len() != members.len() {
            out.extend(members);
        } else {
            out.push(merged_run(&key, members)?);
        }
    }
    Ok(out)
}

pub fn resolve(c: &Connection, m: &str, key: &str) -> Result<Option<Vec<Value>>> {
    if !key.starts_with(PREFIX) {
        return Ok(None);
    }
    let mut members: Vec<_> = published(c, m)?
        .into_iter()
        .filter(|r| identity(r).is_some_and(|(k, _)| k == key))
        .collect();
    anyhow::ensure!(!members.is_empty(), "merged run is unavailable");
    let indices: BTreeSet<_> = members.iter().map(|r| identity(r).unwrap().1).collect();
    anyhow::ensure!(indices.len() == members.len(), "ambiguous shard identity");
    members.sort_by_key(|r| identity(r).unwrap().1);
    Ok(Some(members))
}

pub fn hydrate_pass_metrics(c: &Connection, m: &str, run: &mut Value) -> Result<()> {
    let Some(members) = run["shards"].as_array() else {
        return Ok(());
    };
    if !members.iter().any(|r| !r["pass_at_k"].is_null()) {
        return Ok(());
    }
    let keys = member_keys(members);
    let rows = db::values(
        c,
        "SELECT json_object('run',a.run,'task',a.task,'score',json_extract(a.api,'$.score'),'rid',json_extract(a.api,'$.rollout_id'),'pkg',json_extract(a.api,'$.eval_pkg'),'attempt',json_extract(a.api,'$.attempt'),'rollout',x.rollout,'is_eval',json_extract(v.data,'$.is_eval')) FROM attempts a JOIN membership x ON x.m=a.m AND x.run=a.run AND x.id=a.id JOIN views v ON v.m=x.m AND v.kind='rollout' AND v.run=x.run AND v.rollout='' AND v.id=x.rollout WHERE a.m=? AND a.api IS NOT NULL AND a.run IN(SELECT value FROM json_each(?)) AND json_extract(v.data,'$.category') IS NOT 'trained'",
        &[&m, &keys.as_str()],
    )?;
    let package = rows.iter().any(|r| !text(r, "pkg").is_empty());
    let eval_runs: BTreeSet<_> = rows
        .iter()
        .filter(|r| r["is_eval"] == 1 || r["is_eval"] == true)
        .map(|r| text(r, "run"))
        .collect();
    let all_eval: BTreeSet<_> = members
        .iter()
        .filter(|r| {
            text(r, "train_id").to_lowercase().contains("eval")
                && !eval_runs.contains(text(r, "run_key"))
        })
        .map(|r| text(r, "run_key"))
        .collect();
    static RID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|_)(?:eval_)?r(\d+)").unwrap());
    let mut passes: BTreeMap<String, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for row in &rows {
        if !package
            && !all_eval.contains(text(row, "run"))
            && row["is_eval"] != 1
            && row["is_eval"] != true
        {
            continue;
        }
        let pass = if package {
            let rid = row["rid"]
                .as_i64()
                .or_else(|| {
                    RID.captures(text(row, "attempt"))
                        .and_then(|cap| cap[1].parse().ok())
                })
                .unwrap_or(0);
            format!("pass{rid:05}")
        } else {
            text(row, "rollout").into()
        };
        let score = row["score"].as_f64();
        let existing = passes
            .entry(pass)
            .or_default()
            .entry(text(row, "task").into())
            .or_default();
        if let Some(score) = score {
            *existing = Some(existing.map_or(score, |s| s.max(score)));
        }
    }
    if passes.is_empty() {
        return Ok(());
    }
    let mut tasks: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut accuracies = 0.;
    for pass in passes.values() {
        let scores: Vec<_> = pass.values().flatten().copied().collect();
        if !scores.is_empty() {
            accuracies += scores.iter().filter(|&&s| aggregate::success(s)).count() as f64
                / scores.len() as f64;
        }
        for (task, score) in pass {
            if let Some(s) = score {
                tasks.entry(task).or_default().push(*s);
            } else {
                tasks.entry(task).or_default();
            }
        }
    }
    let scored = tasks.values().filter(|v| !v.is_empty()).count();
    let passed = tasks
        .values()
        .filter(|v| v.iter().any(|&s| aggregate::success(s)))
        .count();
    run["n_passes"] = json!(passes.len());
    run["n_tasks_eval"] = json!(tasks.len());
    run["n_scored_eval"] = json!(scored);
    run["n_passed_eval"] = json!(passed);
    run["pass_at_k"] = json!(if scored == 0 {
        0.
    } else {
        passed as f64 / scored as f64
    });
    run["avg_at_k"] = json!(accuracies / passes.len() as f64);
    Ok(())
}

pub fn views(
    c: &Connection,
    m: &str,
    kind: &str,
    rollout: &str,
    members: &[Value],
) -> Result<Vec<Value>> {
    let keys = member_keys(members);
    let mut statement = c.prepare("SELECT data FROM views WHERE m=? AND kind=? AND rollout=? AND run IN(SELECT value FROM json_each(?))")?;
    let rows: Vec<Value> = statement
        .query_map(params![m, kind, rollout, keys], |r| r.get::<_, String>(0))?
        .map(|r| Ok(serde_json::from_str(&r?)?))
        .collect::<Result<_>>()?;
    let field = if kind == "rollout" {
        "rollout_key"
    } else {
        "task_id"
    };
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for row in rows {
        groups
            .entry(text(&row, field).into())
            .or_default()
            .push(row);
    }
    Ok(groups
        .into_values()
        .map(|rows| merge_stats(&rows))
        .collect())
}
