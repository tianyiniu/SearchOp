"""Step 6: send each test question to one of the train-time question groups.

A test question is described and embedded exactly as the training questions
were (the dataset's describe_<dataset>_<version>.py, then embed_questions.py), placed in the space the
groups were built in (cluster_questions.representation) and given to the group
whose medoid is nearest. That is the rule k-medoids itself used for the
training questions, so the script first applies it to them and reports how
often it returns their stored group (it should be all of them, or very nearly).

Two other routes are written beside the main one, for comparison only:

    knn        majority group of the nearest training questions in the same space
    question   a linear classifier from the embedding of the RAW question to the
               group; it needs no description of the test question, so no
               strong-model call at test time

Also kept per question: the distance to every medoid and the margin between
the nearest two, so a "send unsure questions to the global champion" rule can
be tried afterwards without routing again.

    python scripts/route_questions.py \\
        --test-vectors outputs/describe_v3/vectors_600_test.npz \\
        --out outputs/describe_v3/routes_600_test.json

No model is called. The multi-hot columns that were dropped as near-constant
are decided on the TRAINING questions and the same columns are dropped here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_questions as CQ  # noqa: E402

ROOT = CQ.ROOT


def place(vec, keep: np.ndarray, rep: str, mh_weight: float,
          center: np.ndarray | None = None) -> np.ndarray:
    """cluster_questions.representation, with the kept multi-hot columns and the
    description centre given (both are taken from the training questions,
    never from the test questions)."""
    if rep == "description":
        return CQ.representation(vec, rep, mh_weight, 1.0, center=center)
    mh = CQ.unit(vec["multihot"].astype(np.float64)[:, keep])
    te = CQ.unit(vec["template"].astype(np.float64))
    if rep == "multihot":
        return mh
    if rep == "template":
        return te
    if rep == "both":
        return np.concatenate([mh * np.sqrt(mh_weight), te * np.sqrt(1.0 - mh_weight)], axis=1)
    raise ValueError(rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/describe_v3/clusters_600_train.json")
    ap.add_argument("--train-vectors", type=Path, default=ROOT / "outputs/describe_v3/vectors_600_train.npz")
    ap.add_argument("--test-vectors", type=Path, default=ROOT / "outputs/describe_v3/vectors_600_test.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mh-max-mean", type=float, default=0.9,
                    help="as in cluster_questions.py; must be the value the groups were built with")
    ap.add_argument("--knn", type=int, default=10, help="neighbours for the knn route")
    args = ap.parse_args()

    clusters = json.loads(args.clusters.read_text())
    rep, mh_weight = clusters["rep"], clusters["mh_weight"]
    train = np.load(args.train_vectors, allow_pickle=True)
    test = np.load(args.test_vectors, allow_pickle=True)
    if list(train["multihot_names"]) != list(test["multihot_names"]):
        raise SystemExit("the two vector files use different label vocabularies")
    if str(train["embed_model"]) != str(test["embed_model"]):
        raise SystemExit(f"embedding models differ: {train['embed_model']} vs {test['embed_model']}")

    means = train["multihot"].astype(np.float64).mean(axis=0)
    keep = (means <= args.mh_max_mean) & (means > 0)
    center = CQ.description_center(train) if rep == "description" else None
    x_train, x_test = place(train, keep, rep, mh_weight, center), place(test, keep, rep, mh_weight, center)
    # the vectors the groups were built from, recomputed by the clustering code itself
    if not np.allclose(x_train, CQ.representation(train, rep, mh_weight, args.mh_max_mean)):
        raise SystemExit("the training vectors placed here differ from cluster_questions.representation")

    train_ids = [str(i) for i in train["ids"]]
    row_of = {q: i for i, q in enumerate(train_ids)}
    group_ids = [c["cluster"] for c in clusters["clusters"]]
    medoids = np.stack([x_train[row_of[c["medoid"]]] for c in clusters["clusters"]])
    y_train = np.array([clusters["labels"][q] for q in train_ids])

    def to_medoids(x: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - CQ.unit(x) @ CQ.unit(medoids).T, 0.0, 2.0)

    # self-check: the rule must give the training questions their stored groups
    back = np.array(group_ids)[to_medoids(x_train).argmin(axis=1)]
    agree = float((back == y_train).mean())
    print(f"nearest medoid returns the stored group for {agree:.1%} of the {len(train_ids)} training questions")
    if agree < 0.95:
        raise SystemExit("that is too low: the vectors, --mh-max-mean or the clusters file do not match "
                         "the ones the groups were built with")

    dist = to_medoids(x_test)
    order = dist.argsort(axis=1)
    sims = CQ.unit(x_test) @ CQ.unit(x_train).T
    clf = LogisticRegression(max_iter=2000).fit(train["question"].astype(np.float64), y_train)
    by_question = clf.predict(test["question"].astype(np.float64))

    routes = {}
    for i, q in enumerate(str(t) for t in test["ids"]):
        if q in row_of:
            raise SystemExit(f"test question {q} is also a training question")
        near = sims[i].argsort()[::-1][: args.knn]
        votes = Counter(int(y_train[j]) for j in near)
        top = max(votes.values())
        # ties go to the group of the nearest neighbour among the tied groups
        knn = next(int(y_train[j]) for j in near if votes[int(y_train[j])] == top)
        routes[q] = {"group": int(group_ids[order[i, 0]]),
                     "second": int(group_ids[order[i, 1]]),
                     "margin": round(float(dist[i, order[i, 1]] - dist[i, order[i, 0]]), 6),
                     "distances": {str(g): round(float(d), 6) for g, d in zip(group_ids, dist[i])},
                     "knn": knn, "question": int(by_question[i])}

    main_route = [r["group"] for r in routes.values()]
    out = {"clusters": str(args.clusters.resolve()), "rep": rep, "mh_weight": mh_weight,
           "mh_max_mean": args.mh_max_mean, "knn_k": args.knn, "train_agreement": agree,
           "embed_model": str(test["embed_model"]), "routes": routes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    n = len(routes)
    sizes = Counter(main_route)
    share = Counter(int(v) for v in y_train)
    print(f"{n} test questions routed -> {args.out}")
    print("group   test questions   share   share of training questions")
    for g in group_ids:
        print(f"{g:>5}   {sizes.get(g, 0):>14}   {sizes.get(g, 0) / n:>5.1%}   {share[g] / len(y_train):>5.1%}")
    for other in ("knn", "question"):
        same = sum(r[other] == r["group"] for r in routes.values()) / n
        print(f"the {other!r} route agrees with nearest medoid on {same:.1%} of the test questions")
    margins = np.array([r["margin"] for r in routes.values()])
    print(f"margin between the nearest two medoids: median {np.median(margins):.3f}, "
          f"lowest quarter below {np.quantile(margins, 0.25):.3f}")


if __name__ == "__main__":
    main()
