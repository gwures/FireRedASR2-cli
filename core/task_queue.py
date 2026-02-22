import time
import logging
from typing import Dict, Optional, List
from multiprocessing import Manager, Queue

logger = logging.getLogger("fireredasr2s.task_queue")


class TaskQueue:
    def __init__(self):
        self.manager = Manager()
        self.input_queue = Queue()
        self.status_dict: Dict[str, Dict] = self.manager.dict()

    def create_task(self, file_path: str, enable_punc: bool = True) -> Dict:
        from pathlib import Path
        task_id = Path(file_path).stem
        if task_id.endswith('_converted'):
            task_id = task_id[:-10]
        
        task = {
            "id": task_id,
            "file_path": file_path,
            "status": "queued",
            "created_at": time.time(),
            "enable_punc": enable_punc
        }
        self.status_dict[task_id] = task
        self.input_queue.put(task_id)
        logger.info(f"Created task: {task_id} (punc={enable_punc})")
        return task

    def get_task(self, task_id: str) -> Optional[Dict]:
        if task_id in self.status_dict:
            return self.status_dict[task_id]
        return None

    def get_queue(self) -> Queue:
        return self.input_queue

    def get_status_dict(self) -> Dict:
        return self.status_dict

    def shutdown(self):
        try:
            self.manager.shutdown()
            logger.info("TaskQueue shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
