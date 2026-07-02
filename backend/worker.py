import os
import time

from main import app_logger, get_next_queued_job, process_queued_job


POLL_INTERVAL_SEC = float(os.getenv("WORKER_POLL_INTERVAL_SEC", "5"))
WORKER_ONCE = os.getenv("WORKER_ONCE", "0").lower() in ("1", "true", "yes")


def run_forever():
    app_logger.info("[Worker] Started")
    while True:
        item = get_next_queued_job()
        if not item:
            if WORKER_ONCE:
                app_logger.info("[Worker] No queued job found")
                return
            time.sleep(POLL_INTERVAL_SEC)
            continue

        job_id, job = item
        app_logger.info(f"[Worker] Processing queued job {job_id}")
        try:
            process_queued_job(job_id, job)
        except Exception as e:
            app_logger.error(f"[Worker] Unhandled failure for job {job_id}: {e}", exc_info=True)
        if WORKER_ONCE:
            return


if __name__ == "__main__":
    run_forever()
