from . import task_easy, task_hard, task_medium

TASK_REGISTRY = {
    "easy": task_easy,
    "medium": task_medium,
    "hard": task_hard,
}

__all__ = ["TASK_REGISTRY", "task_easy", "task_medium", "task_hard"]
