from __future__ import annotations

from typing import Any

from core.memory.episodic_memory import collection_name as episodic_collection_name
from core.memory.procedural_memory import clear_user_strategy_stats
from core.memory.semantic_memory import collection_name as semantic_collection_name
from core.store import delete_collection


def clear_user_memory(*, user_id: str) -> dict[str, Any]:
    """
    Remove episodic + semantic vector collections and procedural (strategy) stats for a user.
    Does not delete the main document index (e.g. collection 'docs').
    """
    epi = episodic_collection_name(user_id)
    sem = semantic_collection_name(user_id)
    delete_collection(collection=epi)
    delete_collection(collection=sem)
    n_proc = clear_user_strategy_stats(user_id=user_id)
    return {
        "user_id": user_id,
        "deleted_vector_collections": [epi, sem],
        "procedural_rows_deleted": n_proc,
    }
