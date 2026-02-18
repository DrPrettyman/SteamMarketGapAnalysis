"""Train/test split strategies for implicit feedback data.

Implicit feedback requires special splitting strategies — we can't just
randomly split rows because that would leak temporal information and not
reflect how recommendations work in practice.

Strategies:
    - Leave-one-out: For each user, hold out one randomly-selected game.
    - Temporal split: Hold out the most recently played game per user.
    - Cold-start split: Evaluate separately for games with many vs. few interactions.
"""

import logging

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def leave_one_out_split(
    user_items: sparse.csr_matrix,
    random_state: int = 42,
) -> tuple[sparse.csr_matrix, list[dict]]:
    """Leave-one-out split: hold out one item per user for testing.

    For each user with ≥2 interactions, randomly select one item as the
    test item.  The remaining items become the training set.

    Args:
        user_items: Sparse interaction matrix (n_users × n_items).
        random_state: Random seed.

    Returns:
        Tuple of (train_matrix, test_data) where test_data is a list of dicts
        with ``user_idx``, ``train_items``, and ``test_items``.
    """
    rng = np.random.RandomState(random_state)
    n_users, n_items = user_items.shape

    train = user_items.copy().tolil()
    test_data: list[dict] = []

    for u in range(n_users):
        items = user_items[u].nonzero()[1]
        if len(items) < 2:
            continue

        # Randomly select one item to hold out
        test_idx = rng.choice(len(items))
        test_item = items[test_idx]

        train[u, test_item] = 0
        train_items = [int(i) for i in items if i != test_item]

        test_data.append(
            {
                "user_idx": int(u),
                "train_items": train_items,
                "test_items": [int(test_item)],
            }
        )

    train = train.tocsr()
    train.eliminate_zeros()

    logger.info(
        "Leave-one-out split: %d test users (of %d total), train nnz=%d",
        len(test_data),
        n_users,
        train.nnz,
    )

    return train, test_data


def cold_start_split(
    test_data: list[dict],
    user_items: sparse.csr_matrix,
    threshold: int = 100,
) -> tuple[list[dict], list[dict]]:
    """Split test data into warm (≥ threshold interactions) and cold-start items.

    This allows separate evaluation of well-known vs. new/niche games.

    Args:
        test_data: Test data from :func:`leave_one_out_split`.
        user_items: Original interaction matrix (to count item interactions).
        threshold: Minimum interaction count to be "warm".

    Returns:
        Tuple of (warm_test_data, cold_test_data).
    """
    item_counts = np.asarray((user_items > 0).sum(axis=0)).flatten()

    warm: list[dict] = []
    cold: list[dict] = []

    for entry in test_data:
        test_item = entry["test_items"][0]
        if item_counts[test_item] >= threshold:
            warm.append(entry)
        else:
            cold.append(entry)

    logger.info(
        "Cold-start split: %d warm, %d cold (threshold=%d)",
        len(warm),
        len(cold),
        threshold,
    )

    return warm, cold


def popularity_baseline(
    user_items: sparse.csr_matrix,
    n: int = 10,
) -> list[int]:
    """Generate a popularity-based recommendation baseline.

    Simply returns the N most-interacted-with items across all users.
    This is the simplest baseline — any recommender should beat this.

    Args:
        user_items: Sparse interaction matrix.
        n: Number of items to recommend.

    Returns:
        List of item indices sorted by popularity (descending).
    """
    item_popularity = np.asarray((user_items > 0).sum(axis=0)).flatten()
    return np.argsort(item_popularity)[::-1][:n].tolist()
