CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempts(seq INTEGER PRIMARY KEY,m TEXT NOT NULL,id TEXT NOT NULL,run TEXT NOT NULL,task TEXT NOT NULL,mtime REAL NOT NULL,payload TEXT NOT NULL,api TEXT,UNIQUE(m,id));
CREATE INDEX IF NOT EXISTS attempts_run ON attempts(m,run,seq);
CREATE INDEX IF NOT EXISTS attempts_time ON attempts(m,mtime DESC,id);
CREATE INDEX IF NOT EXISTS attempts_task ON attempts(m,run,task,mtime DESC,id);
CREATE TABLE IF NOT EXISTS views(m TEXT NOT NULL,kind TEXT NOT NULL,run TEXT NOT NULL,rollout TEXT NOT NULL,id TEXT NOT NULL,sort REAL NOT NULL,data TEXT NOT NULL,PRIMARY KEY(m,kind,run,rollout,id));
CREATE INDEX IF NOT EXISTS views_page ON views(m,kind,run,rollout,sort DESC,id);
CREATE INDEX IF NOT EXISTS views_runs ON views(m,kind,sort DESC,id);
CREATE INDEX IF NOT EXISTS views_task ON views(m,kind,id,rollout);
CREATE TABLE IF NOT EXISTS membership(m TEXT NOT NULL,run TEXT NOT NULL,rollout TEXT NOT NULL,id TEXT NOT NULL,PRIMARY KEY(m,run,rollout,id));
CREATE TABLE IF NOT EXISTS dirty_runs(m TEXT NOT NULL,run TEXT NOT NULL,PRIMARY KEY(m,run));
CREATE TABLE IF NOT EXISTS trained(m TEXT NOT NULL,train TEXT NOT NULL,seg TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(m,train,seg));
CREATE TABLE IF NOT EXISTS runtime(m TEXT PRIMARY KEY,updated REAL NOT NULL,health TEXT NOT NULL,sessions TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples(m TEXT NOT NULL,t REAL NOT NULL,data TEXT NOT NULL,PRIMARY KEY(m,t));
CREATE TABLE IF NOT EXISTS discovery(m TEXT NOT NULL,prefix TEXT NOT NULL,cursor TEXT NOT NULL DEFAULT '',generation INTEGER NOT NULL DEFAULT 1,due REAL NOT NULL DEFAULT 0,failures INTEGER NOT NULL DEFAULT 0,priority INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(m,prefix));
CREATE INDEX IF NOT EXISTS discovery_due ON discovery(due,priority);
CREATE TABLE IF NOT EXISTS objects(m TEXT NOT NULL,path TEXT NOT NULL,etag TEXT,mtime REAL NOT NULL,size INTEGER NOT NULL,scope TEXT NOT NULL,generation INTEGER NOT NULL,PRIMARY KEY(m,path));
CREATE INDEX IF NOT EXISTS objects_scope ON objects(m,scope,generation);
CREATE TABLE IF NOT EXISTS active(m TEXT NOT NULL,id TEXT NOT NULL,due REAL NOT NULL,failures INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(m,id));
CREATE INDEX IF NOT EXISTS active_due ON active(due);
CREATE TABLE IF NOT EXISTS scan_jobs(m TEXT NOT NULL,prefix TEXT NOT NULL,kind TEXT NOT NULL,cursor TEXT NOT NULL DEFAULT '',generation INTEGER NOT NULL DEFAULT 1,due REAL NOT NULL DEFAULT 0,priority INTEGER NOT NULL,failures INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(m,prefix,kind));
CREATE INDEX IF NOT EXISTS scan_jobs_due ON scan_jobs(m,due,priority);
CREATE TABLE IF NOT EXISTS live_run_watches(m TEXT NOT NULL,root TEXT NOT NULL,expires REAL NOT NULL,PRIMARY KEY(m,root));
CREATE TABLE IF NOT EXISTS discovery_activity(m TEXT NOT NULL,prefix TEXT NOT NULL,fingerprint TEXT NOT NULL DEFAULT '',pending TEXT NOT NULL DEFAULT '',unchanged INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(m,prefix));
CREATE TABLE IF NOT EXISTS scan_stage(m TEXT NOT NULL,scope TEXT NOT NULL,generation INTEGER NOT NULL,path TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(m,scope,generation,path));
CREATE TABLE IF NOT EXISTS scan_seen(m TEXT NOT NULL,scope TEXT NOT NULL,id TEXT NOT NULL,generation INTEGER NOT NULL,PRIMARY KEY(m,scope,id));
CREATE INDEX IF NOT EXISTS attempts_run_time ON attempts(m,run,mtime);
CREATE TABLE IF NOT EXISTS legacy_updates(seq INTEGER PRIMARY KEY AUTOINCREMENT,m TEXT NOT NULL,id TEXT NOT NULL,payload TEXT);
CREATE TABLE IF NOT EXISTS run_versions(m TEXT NOT NULL,run TEXT NOT NULL,version INTEGER NOT NULL,PRIMARY KEY(m,run));
CREATE TRIGGER IF NOT EXISTS version_attempt_insert AFTER INSERT ON attempts BEGIN
 INSERT INTO run_versions VALUES(NEW.m,NEW.run,1) ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TRIGGER IF NOT EXISTS version_attempt_update AFTER UPDATE OF payload,run,task,mtime ON attempts BEGIN
 INSERT INTO run_versions VALUES(OLD.m,OLD.run,1) ON CONFLICT(m,run) DO UPDATE SET version=version+1;
 INSERT INTO run_versions VALUES(NEW.m,NEW.run,1) ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TRIGGER IF NOT EXISTS version_attempt_delete AFTER DELETE ON attempts BEGIN
 INSERT INTO run_versions VALUES(OLD.m,OLD.run,1) ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TABLE IF NOT EXISTS run_identity(m TEXT NOT NULL,run TEXT NOT NULL,train TEXT,PRIMARY KEY(m,run));
CREATE INDEX IF NOT EXISTS run_identity_train ON run_identity(m,train,run);
CREATE TRIGGER IF NOT EXISTS identity_attempt_insert AFTER INSERT ON attempts BEGIN
 INSERT INTO run_identity VALUES(NEW.m,NEW.run,json_extract(NEW.payload,'$.train_id')) ON CONFLICT(m,run) DO UPDATE SET train=excluded.train WHERE train IS NOT excluded.train;
END;
CREATE TRIGGER IF NOT EXISTS identity_attempt_update AFTER UPDATE OF payload,run ON attempts BEGIN
 INSERT INTO run_identity VALUES(NEW.m,NEW.run,json_extract(NEW.payload,'$.train_id')) ON CONFLICT(m,run) DO UPDATE SET train=excluded.train WHERE train IS NOT excluded.train;
END;
CREATE TRIGGER IF NOT EXISTS version_trained_insert AFTER INSERT ON trained BEGIN
 INSERT INTO run_versions SELECT m,run,1 FROM run_identity WHERE m=NEW.m AND train=NEW.train ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TRIGGER IF NOT EXISTS version_trained_update AFTER UPDATE ON trained BEGIN
 INSERT INTO run_versions SELECT m,run,1 FROM run_identity WHERE m=NEW.m AND train=NEW.train ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TRIGGER IF NOT EXISTS version_trained_delete AFTER DELETE ON trained BEGIN
 INSERT INTO run_versions SELECT m,run,1 FROM run_identity WHERE m=OLD.m AND train=OLD.train ON CONFLICT(m,run) DO UPDATE SET version=version+1;
END;
CREATE TABLE IF NOT EXISTS scan_completed(m TEXT NOT NULL,prefix TEXT NOT NULL,kind TEXT NOT NULL,at REAL NOT NULL,PRIMARY KEY(m,prefix,kind));
CREATE INDEX IF NOT EXISTS attempts_run_page ON attempts(m,run,mtime DESC,id);
CREATE INDEX IF NOT EXISTS attempts_task_global ON attempts(m,task,mtime DESC,id);
CREATE TABLE IF NOT EXISTS freshness_events(m TEXT NOT NULL,id TEXT NOT NULL,kind TEXT NOT NULL,source_at REAL NOT NULL,observed_at REAL NOT NULL,published_at REAL,PRIMARY KEY(m,id,kind));
CREATE INDEX IF NOT EXISTS freshness_pending ON freshness_events(m,published_at,id);
CREATE TABLE IF NOT EXISTS sidecar_seen(m TEXT NOT NULL,scope TEXT NOT NULL,path TEXT NOT NULL,generation INTEGER NOT NULL,PRIMARY KEY(m,scope,path));
CREATE TABLE IF NOT EXISTS attempt_observed(m TEXT NOT NULL,id TEXT NOT NULL,at REAL NOT NULL,PRIMARY KEY(m,id));
CREATE TABLE IF NOT EXISTS stage_observed(m TEXT NOT NULL,scope TEXT NOT NULL,generation INTEGER NOT NULL,id TEXT NOT NULL,at REAL NOT NULL,PRIMARY KEY(m,scope,generation,id));
CREATE TABLE IF NOT EXISTS scan_started(m TEXT NOT NULL,scope TEXT NOT NULL,generation INTEGER NOT NULL,at REAL NOT NULL,PRIMARY KEY(m,scope,generation));
