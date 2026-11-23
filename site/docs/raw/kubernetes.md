# Kubernetes Fleets (MobileWorld)

[Home](../../../README.md) · [Quickstart](index.md) · [Evaluation](evaluation.md) · [Training](training.md) · [Rollout](rollout.md) · **Kubernetes** · [Concepts](concepts.md)

Scale a single **MAI-Gym** manager to lease Android environments across multiple Kubernetes nodes. Evaluation and RL runners communicate exclusively with the manager, while the manager provisions sandboxes and routes requests directly to each MobileWorld backend's pod IP.

```text
mai-gym eval / train  ──▶  MAI-Gym Manager (:8090)  ──▶  SandboxSet / SandboxClaim  ──▶  MobileWorld Pods (:6800)
```

The walkthrough below provisions a 2-session warm pool and evaluates `SetAlarmTask` against it.

---

## 1. Verify cluster prerequisites

The Kubernetes launcher uses the Alibaba Cloud ACS sandbox template API and `agents.kruise.io/v1alpha1` custom resources (`Sandbox`, `SandboxSet`, and `SandboxClaim`). Ensure your cluster has:

- A compatible `agents.kruise.io` controller installed.
- A sandbox template that boots the MobileWorld emulator, restores Android assets, and serves the backend on a known port ([MobileWorld image & snapshot preparation](../../../skills/envs/mobileworld.md)).
- Network reachability from the manager host to pod IPs on the backend port.

Verify permissions using the kubeconfig the manager will run with (replace `mai-gym-eval` with a dedicated namespace for this fleet):

```bash
kubectl api-resources --api-group=agents.kruise.io
kubectl get namespace mai-gym-eval
kubectl auth can-i create sandboxclaims.agents.kruise.io -n mai-gym-eval
```

---

## 2. Start the manager and verify readiness

Save the configuration below as `fleet.yaml` on the manager host. Replace `<mobileworld-template-id>` with your cluster's template ID and set `service_port` to the backend port **inside the pod**.

```yaml
devices:
  - name: mobileworld-fleet
    env_type: mobileworld
    device_type: mobile
    infra_type: k8s
    max_sessions: 2
    sessions_per_backend: 1
    launch:
      acquisition: pool
      namespace: mai-gym-eval
      template_id: <mobileworld-template-id>
      service_port: 6800
      pool_name: mobileworld-eval
      pool_size: 2
manager:
  host: 127.0.0.1
  port: 8090
  auto_prewarm: true
```

- `pool_size` sets the warm replica count maintained by the `SandboxSet`.
- `max_sessions` caps the total sessions registered on the manager.
- `auto_prewarm: true` claims backends at startup so runners can immediately discover tasks.
- By default, the manager uses `~/.kube/config`; set `launch.kubeconfig` to use another kubeconfig file.

Start the manager:

```bash
uv run mai-gym serve --config fleet.yaml
```

In another terminal on the manager host, verify that pods are claimed, `by_device.mobileworld-fleet.ready` reaches `2`, and `/tasks` includes `SetAlarmTask`:

```bash
kubectl -n mai-gym-eval get sandboxsets,sandboxclaims,pods
curl -fsS http://127.0.0.1:8090/sessions
curl -fsS http://127.0.0.1:8090/tasks
```

> [!TIP]
> If sessions remain unready, inspect `SandboxClaim` status and pod events in `mai-gym-eval` (`kubectl -n mai-gym-eval describe pods`).

---

## 3. Run an evaluation against the fleet

Keep your model configuration from the [Evaluation Guide](evaluation.md) and point `benchmarks.mobileworld.manager_url` to the fleet manager:

```yaml
benchmarks:
  mobileworld:
    manager_url: http://127.0.0.1:8090
    num_envs: 1
    tasks: [SetAlarmTask]
```

```bash
uv run mai-gym eval --config eval.yaml --benchmarks mobileworld --trials 1
```

> [!NOTE]
> If the runner executes on a different machine than the manager, bind `manager.host` appropriately and pass the manager's reachable address in `manager_url`. For larger suites, increase `num_envs` up to your manager's session capacity.

---

## Capacity Acquisition Modes

Choose `launch.acquisition` based on your workload pattern:

| Mode | Best for | How backends are acquired |
| :--- | :--- | :--- |
| `direct` | Occasional runs that can tolerate cold-start time | Creates an individual `Sandbox` on demand from `template_id`. |
| `pool` | Repeated evaluations or high-throughput RL training | Claims ready backends from a named `SandboxSet` maintained pre-warmed by the controller. |
| `adopt` | Pre-provisioned backends in a dedicated namespace | Discovers eligible running, unowned sandboxes with a pod IP (capacity bounded by available instances). |

### Demand-based fleet autoscaling

To scale warm pool capacity automatically based on active session demand, combine `acquisition: pool` with `lazy_acquire` and `fleet_autoscale` (using `sessions_per_backend: 1`):

```yaml
    launch:
      acquisition: pool
      lazy_acquire: true
      fleet_autoscale:
        enabled: true
        min_replicas: 2   # Pre-warmed replicas retained after idle timeout
```

Monitor live demand and allocation through the manager API:

- `GET /fleet/requests` — currently requested fleet capacity
- `GET /sessions` — active, ready, and leased session counts

---

## Retiring a Fixed Pool

> [!WARNING]
> Stopping the manager process alone can leave `SandboxSet`, `SandboxClaim`, and `Sandbox` resources running in the cluster for reuse on restart.

Follow this checklist when decommissioning a fleet:

1. Wait for active evaluation or training jobs to finish.
2. Confirm `summary.leased` is `0` in `curl -fsS http://127.0.0.1:8090/sessions`.
3. Stop the `mai-gym serve` process.
4. Delete the `SandboxSet` and any remaining `SandboxClaim` / `Sandbox` resources in the dedicated namespace (`mai-gym-eval`).
