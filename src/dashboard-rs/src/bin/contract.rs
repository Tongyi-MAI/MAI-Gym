use serde_json::{Value, json};
use std::io::{self, BufRead};
fn main() -> anyhow::Result<()> {
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line?)?;
        let trained = serde_json::from_value(v.get("trained").cloned().unwrap_or(json!({})))?;
        let started = std::time::Instant::now();
        let result = mai_dashboard::aggregate::build_run(
            serde_json::from_value(v["items"].clone())?,
            v["root"].as_str().unwrap_or(""),
            &trained,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        if v["timing"] == true {
            println!(
                "{}",
                json!({"aggregation_ms":elapsed,"attempts":result.items.len()})
            );
        } else {
            println!("{}", serde_json::to_string(&result)?);
        }
    }
    Ok(())
}
