# Modified by MAI-Gym contributors.
import logging
import threading

import ray


logger = logging.getLogger(__name__)


class RolloutHealthMonitor:

    def __init__(self, server_group, args):
        self._server_group = server_group

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True
        self._is_checking_enabled = False

    def start(self) -> bool:
        if not self._server_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._server_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
            )
            self._kill_engine(rollout_engine_id=rollout_engine_id)
        else:
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")

    def _kill_engine(self, rollout_engine_id: int):
        logger.info(f"Killing server group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._server_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._server_group.nodes_per_engine,
        ):
            engine = self._server_group.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self._server_group.all_engines[i] = None
