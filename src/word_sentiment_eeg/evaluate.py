"""Fixed ridge probes, group splits and word-form permutation controls.

Pure NumPy implementation permits offline validation using the bundled runtime.
All feature filtering, scaling and PCA are fitted on training rows only.
"""
from collections import Counter
import warnings

import numpy as np
import pandas as pd

CLASSES = ["negative", "neutral", "positive"]


def metrics(y, pred, classes):
    matrix = np.array([[np.sum((y == a) & (pred == b)) for b in classes] for a in classes])
    support = matrix.sum(axis=1)
    tp = matrix.diagonal()
    recall = np.divide(tp, support, out=np.zeros(len(classes), float), where=support > 0)
    denom = matrix.sum(axis=0) + support
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(classes), float), where=denom > 0)
    return {"accuracy": float(np.mean(y == pred)), "balanced_accuracy": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)), "confusion": matrix.tolist(),
            "class_order": list(classes), "class_recall": recall.tolist(), "class_f1": f1.tolist()}


def make_splits(table, protocol, folds, seed):
    """All readers of a sentence stay together. Purge shared word forms optionally.

    The joint protocol also holds out a deterministic subject group. Its test
    coverage is intentionally partial (reader group crossed with sentence fold).
    """
    if protocol not in {"sentence", "sentence_word_form", "subject_sentence_word_form"}:
        raise ValueError(f"Unknown split protocol {protocol}")
    rng = np.random.default_rng(seed)
    sentences = np.array(sorted(table.sentence_key.unique()))
    sentences = rng.permutation(sentences)
    subject_groups = np.array_split(rng.permutation(sorted(table.subject.unique())), folds)
    for fold, held_sentences in enumerate(np.array_split(sentences, folds)):
        test = table.sentence_key.isin(held_sentences).to_numpy()
        train = ~test
        if protocol == "subject_sentence_word_form":
            held_readers = set(subject_groups[fold])
            test &= table.subject.isin(held_readers).to_numpy()
            train &= ~table.subject.isin(held_readers).to_numpy()
        before = int(train.sum())
        if "word_form" in protocol:
            train &= ~table.word_form.isin(set(table.loc[test, "word_form"])).to_numpy()
        tr, te = np.flatnonzero(train), np.flatnonzero(test)
        assert not set(table.iloc[tr].sentence_key) & set(table.iloc[te].sentence_key)
        if "word_form" in protocol:
            assert not set(table.iloc[tr].word_form) & set(table.iloc[te].word_form)
        if protocol.startswith("subject_"):
            assert not set(table.iloc[tr].subject) & set(table.iloc[te].subject)
        yield fold, tr, te, before - len(tr)


def transform_fit(train, test, max_components=None):
    """Finite training values only; flat channels disappear without hard-coded Cz."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    median = np.nan_to_num(median)
    a = np.where(np.isfinite(train), train, median)
    b = np.where(np.isfinite(test), test, median)
    mean, sd = a.mean(axis=0), a.std(axis=0)
    keep = sd > 1e-8
    if not keep.any():
        return np.zeros((len(a), 1)), np.zeros((len(b), 1)), {"features_kept": 0, "components": 0}
    a, b = (a[:, keep] - mean[keep]) / sd[keep], (b[:, keep] - mean[keep]) / sd[keep]
    info = {"features_kept": int(keep.sum()), "components": int(keep.sum())}
    if max_components is not None and a.shape[1] > max_components:
        # Covariance eigendecomposition depends only on training EEG.
        _, vectors = np.linalg.eigh(a.T @ a / max(1, len(a)))
        projection = vectors[:, -min(max_components, len(a) - 1):]
        a, b = a @ projection, b @ projection
        info["components"] = a.shape[1]
    return a, b, info


def fit_ridge_operator(train, test, alpha, weights):
    a = np.column_stack([train, np.ones(len(train))])
    b = np.column_stack([test, np.ones(len(test))])
    penalty = np.eye(a.shape[1]) * alpha
    penalty[-1, -1] = 0  # unpenalized intercept
    lhs = a.T @ (weights[:, None] * a) + penalty
    # p × n rather than n_test × n_train: no quadratic trial kernel allocation.
    return np.linalg.solve(lhs, a.T * weights), b


def nuisance_features(train, test):
    # Counts use unique training text occurrences, not repeated reader copies.
    counts = Counter(train.drop_duplicates(["sentence_key", "word_index"]).word_form)
    def matrix(table):
        return np.column_stack([
            table.word_form.str.len().to_numpy(float),
            np.log1p([counts.get(w, 0) for w in table.word_form]),
            table.relative_position.to_numpy(float), table.sentence_length.to_numpy(float),
            table[["n_fixations", "trt_ms", "ffd_ms", "gd_ms"]].to_numpy(float),
        ])
    return matrix(train), matrix(test)


def run_probe(table, eeg, config, protocol, classes):
    """Returns honest skipped-fold status, per-trial predictions, and a null screen.

    Permute lexical labels at word-form level, preserving every repeated reader
    and occurrence. This is a lexical-label randomization screen, not a causal
    sentiment test. It does not resolve correlated sentence or semantic effects.
    """
    table = table.reset_index(drop=True)
    y = np.array([classes.index(c) for c in table.lexical_class])
    class_indices = list(range(len(classes)))
    rng = np.random.default_rng(config["seed"] + 811)
    forms = sorted(table.word_form.unique())
    by_form = table.groupby("word_form").lexical_class.agg(lambda s: len(set(s)))
    permutation_applicable = not (by_form != 1).any()
    form_labels = np.array([y[np.flatnonzero(table.word_form.to_numpy() == w)[0]] for w in forms])
    permuted = [dict(zip(forms, rng.permutation(form_labels))) for _ in range(config["permutations"])] if permutation_applicable else []
    results, predictions = [], []
    null_predictions = [[] for _ in permuted]
    null_targets = [[] for _ in permuted]
    for fold, tr, te, purged in make_splits(table, protocol, config["folds"], config["seed"]):
        record = {"fold": fold, "n_train": len(tr), "n_test": len(te), "purged_train_rows": purged}
        record["split_membership"] = {
            "train_row_ids": [int(v) for v in table.row_id.iloc[tr]],
            "test_row_ids": [int(v) for v in table.row_id.iloc[te]],
        }
        train_counts = [int(np.sum(y[tr] == c)) for c in class_indices]
        test_counts = [int(np.sum(y[te] == c)) for c in class_indices]
        types = [table.iloc[tr].loc[y[tr] == c, "word_form"].nunique() for c in class_indices]
        record.update(train_class_counts=train_counts, test_class_counts=test_counts,
                      train_word_forms_per_class=[int(n) for n in types])
        if (min(train_counts, default=0) < config["minimum_train_per_class"]
                or min(test_counts, default=0) < config["minimum_test_per_class"]
                or min(types, default=0) < config["minimum_train_word_forms_per_class"]):
            results.append({**record, "status": "insufficient_class_coverage"})
            continue
        print(f"  {protocol} fold {fold + 1}: train {len(tr):,}, test {len(te):,}, purged {purged:,}", flush=True)
        train, test = table.iloc[tr], table.iloc[te]
        eeg_a, eeg_b, info = transform_fit(eeg[tr], eeg[te], config["max_eeg_components"])
        n_a, n_b = nuisance_features(train, test)
        n_a, n_b, _ = transform_fit(n_a, n_b)
        # One reader cannot turn the same word occurrence into independent labels.
        occurrence_count = train.groupby("item_id").item_id.transform("count").to_numpy()
        weights = 1.0 / occurrence_count
        weights /= weights.mean()
        targets = np.eye(len(classes))[y[tr]]
        operators = {}
        matrices = {"eeg": (eeg_a, eeg_b), "nuisance": (n_a, n_b),
                    "eeg_plus_nuisance": (np.column_stack([eeg_a, n_a]), np.column_stack([eeg_b, n_b]))}
        fold_preds = {}
        for name, (a, b) in matrices.items():
            op, design = fit_ridge_operator(a, b, config["ridge_alpha"], weights)
            operators[name] = (op, design)
            fold_preds[name] = np.argmax(design @ (op @ targets), axis=1)
        weighted_counts = np.bincount(y[tr], weights=weights, minlength=len(classes))
        fold_preds["majority"] = np.full(len(te), int(np.argmax(weighted_counts)))
        record.update(status="completed", preprocessing=info,
                      scores={name: metrics(y[te], p, class_indices) for name, p in fold_preds.items()})
        results.append(record)
        for j, idx in enumerate(te):
            predictions.append({"row_id": int(table.row_id.iloc[idx]), "fold": fold,
                                "true": int(y[idx]), **{name: int(p[j]) for name, p in fold_preds.items()}})
        # Batch the null targets for one BLAS call per fold, no refitting/PCA per permutation.
        if permuted:
            op, design = operators["eeg"]
            target_blocks = [np.eye(len(classes))[[mapping[w] for w in train.word_form]] for mapping in permuted]
            scores = design @ (op @ np.concatenate(target_blocks, axis=1))
            for p, mapping in enumerate(permuted):
                null_predictions[p].extend(scores[:, p * len(classes):(p + 1) * len(classes)].argmax(axis=1).tolist())
                null_targets[p].extend(mapping[w] for w in test.word_form)
    pred = pd.DataFrame(predictions)
    n_complete = sum(r["status"] == "completed" for r in results)
    summary = {"protocol": protocol, "classes": classes, "folds": results,
               "status": "completed" if n_complete == config["folds"] else ("partial" if n_complete else "insufficient_data"),
               "completed_folds": n_complete, "total_folds": config["folds"],
               "eligible_rows": len(table), "evaluated_rows": len(pred),
               "prediction_unit": "reader_word_occurrence; repeat readers are dependent",
               "inference_limit": "Lexical-label screen; word identity, semantics and correlated context remain possible confounds."}
    if len(pred):
        summary["scores"] = {name: metrics(pred.true.to_numpy(), pred[name].to_numpy(), class_indices)
                             for name in ("eeg", "nuisance", "eeg_plus_nuisance", "majority")}
        null = [metrics(np.array(a), np.array(b), class_indices)["balanced_accuracy"]
                for a, b in zip(null_targets, null_predictions) if a]
        observed = summary["scores"]["eeg"]["balanced_accuracy"]
        summary["word_form_permutation_screen"] = {
            "status": "computed" if permutation_applicable else "not_applicable_word_form_has_multiple_sense_labels",
            "n": len(null), "null_balanced_accuracies": null,
            "p_uncorrected": float((1 + sum(v >= observed for v in null)) / (1 + len(null))) if null else None,
            "validity": "exploratory; word-form labels exchangeable null; no correction across tasks/protocols",
        }
    return summary, pred
