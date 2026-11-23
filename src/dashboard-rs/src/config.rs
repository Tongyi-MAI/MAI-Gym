use serde::{Deserialize, Serialize};
use std::path::Path;
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manager {
    pub label: String,
    pub url: String,
    pub traj_root: Option<String>,
    pub capacity_mode: Option<String>,
    pub k8s_inventory: Option<serde_json::Value>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub managers: Vec<Manager>,
    #[serde(default)]
    pub url_base: String,
}
impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let c: Self = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        anyhow::ensure!(!c.managers.is_empty(), "no managers configured");
        let mut roots = std::collections::HashSet::new();
        for m in &c.managers {
            anyhow::ensure!(roots.insert(&m.url), "duplicate manager URL");
        }
        Ok(c)
    }
    pub fn manager(&self, key: &str) -> anyhow::Result<&Manager> {
        self.managers
            .get(key.parse::<usize>()?)
            .ok_or_else(|| anyhow::anyhow!("unknown manager"))
    }
}
