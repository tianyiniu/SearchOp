"""Steps 3 and 4 of the per-type search: group the train questions by reasoning
shape and pick a representative subset per group.

Clustering is k-medoids over a chosen representation from embed_questions.py
(multihot, template, both, or description: the whole record embedded as one
passage, centred on the training mean), so every group centre is a real question. No
debate is run: everything here costs seconds.

Choosing k without spending model calls. For each k we report:
  silhouette    how separated the groups are in the chosen space (higher better)
  stability     mean adjusted Rand index between the full clustering and
                re-clusterings of bootstrap subsamples (higher better; groups
                that move when the data is resampled are not reasoning types)
  min_size      smallest group; a group needs enough questions to search on
  recover_acc   5-fold cross-validated accuracy of a linear classifier from the
                RAW question embedding to the group label, next to the
                majority-class rate. This is the test-time router: if the groups
                cannot be recovered from the question text, they cannot be used.
  disc_nmi      mutual information with the discipline label (diagnostic: if it
                is high, the descriptions carried topic and need harder scrubbing)
The chosen k is the largest one whose stability and min_size clear the
thresholds; ties go to silhouette. --k overrides.

The representative subset per group starts at the medoid, then alternates
farthest-point picks (cover the group's spread) with random picks (do not
over-sample its outliers) until --per-cluster questions are chosen. The rest
of the group is written as its held-out set, for validating a program found on
the subset.

    python3 scripts/cluster_questions.py --rep both --out outputs/clusters_train.json
    python3 scripts/cluster_questions.py --rep multihot --k 6 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent


# --- representations --------------------------------------------------------------

def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def informative_multihot(vec: np.lib.npyio.NpzFile, max_mean: float) -> np.ndarray:
    """Drop multi-hot columns that are on for nearly every question (or none):
    they cannot separate groups and only add a shared component to every
    vector."""
    mh = vec["multihot"].astype(np.float64)
    means = mh.mean(axis=0)
    keep = (means <= max_mean) & (means > 0)
    dropped = [str(n) for n, k in zip(vec["multihot_names"], keep) if not k]
    if dropped:
        print(f"multi-hot columns dropped as near-constant (mean > {max_mean} or 0): {dropped}")
    return mh[:, keep]


def description_center(vec: np.lib.npyio.NpzFile) -> np.ndarray:
    """The mean unit description vector of these (training) questions."""
    return unit(vec["description"].astype(np.float64)).mean(axis=0)


def representation(vec: np.lib.npyio.NpzFile, rep: str, mh_weight: float,
                   max_mean: float, center: np.ndarray | None = None) -> np.ndarray:
    if rep == "multihot":
        return unit(informative_multihot(vec, max_mean))
    if rep == "template":
        return unit(vec["template"].astype(np.float64))
    if rep == "both":         # unit blocks side by side, cosine on the pair
        mh = unit(informative_multihot(vec, max_mean)) * np.sqrt(mh_weight)
        te = unit(vec["template"].astype(np.float64)) * np.sqrt(1.0 - mh_weight)
        return np.concatenate([mh, te], axis=1)
    if rep == "description":  # the whole record as one passage (embed_questions.description_text)
        if "description" not in vec:
            raise SystemExit("these vectors have no 'description' embedding; re-run embed_questions.py")
        # Every description shares most of its text (headers, step and challenge
        # definitions), so the raw vectors crowd into one direction (mean pairwise
        # cosine 0.94). Subtracting the training questions' mean vector and
        # renormalising spreads them out. Test questions are centred on the
        # TRAINING mean, passed as `center` (route_questions.py).
        x = unit(vec["description"].astype(np.float64))
        return unit(x - (description_center(vec) if center is None else center))
    raise ValueError(rep)


def cosine_distances(x: np.ndarray) -> np.ndarray:
    d = 1.0 - unit(x) @ unit(x).T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0)


# --- k-medoids -------------------------------------------------------------------

def kmedoids(dist: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10,
             max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Alternating k-medoids (assign to nearest medoid, re-centre each group on
    its own medoid), k-means++-style starts, best of n_init by total distance.
    Returns (labels, medoid indices, cost)."""
    n = dist.shape[0]
    best = None
    for _ in range(n_init):
        med = [int(rng.integers(n))]
        while len(med) < k:                                   # D^2-weighted seeding
            d2 = dist[:, med].min(axis=1) ** 2
            if d2.sum() <= 0:
                med.append(int(rng.choice(np.setdiff1d(np.arange(n), med))))
                continue
            med.append(int(rng.choice(n, p=d2 / d2.sum())))
        med = np.array(med)
        for _ in range(max_iter):
            lab = dist[:, med].argmin(axis=1)
            new = med.copy()
            for j in range(k):
                idx = np.flatnonzero(lab == j)
                if idx.size:
                    new[j] = idx[dist[np.ix_(idx, idx)].sum(axis=1).argmin()]
            if set(new.tolist()) == set(med.tolist()):
                break
            med = new
        lab = dist[:, med].argmin(axis=1)
        cost = float(dist[np.arange(n), med[lab]].sum())
        if best is None or cost < best[2]:
            best = (lab, med, cost)
    return best


# --- metrics for choosing k -------------------------------------------------------

def stability(dist: np.ndarray, k: int, full_labels: np.ndarray, rng: np.random.Generator,
              n_boot: int, frac: float) -> float:
    n = dist.shape[0]
    aris = []
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        lab, _, _ = kmedoids(dist[np.ix_(idx, idx)], k, rng, n_init=3)
        aris.append(adjusted_rand_score(full_labels[idx], lab))
    return float(np.mean(aris))


def recoverability(q_emb: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float:
    """Cross-validated accuracy of a linear router from the raw question
    embedding to the group label."""
    if np.bincount(labels).min() < 5:
        return float("nan")
    cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 31)))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    pred = cross_val_predict(clf, q_emb, labels, cv=cv)
    return float((pred == labels).mean())


# --- subset -------------------------------------------------------------------------

def pick_subset(dist: np.ndarray, members: np.ndarray, medoid: int, n_pick: int,
                rng: np.random.Generator) -> list[int]:
    """Medoid first, then alternate farthest-point and random picks."""
    members = list(members)
    if len(members) <= n_pick:
        return members
    chosen = [medoid]
    pool = [m for m in members if m != medoid]
    turn = 0
    while len(chosen) < n_pick and pool:
        if turn % 2 == 0:
            mind = dist[np.ix_(pool, chosen)].min(axis=1)
            pick = pool[int(mind.argmax())]
        else:
            pick = pool[int(rng.integers(len(pool)))]
        chosen.append(pick)
        pool.remove(pick)
        turn += 1
    return chosen


# --- main ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", type=Path, default=ROOT / "outputs/describe_v3/vectors_2k_train.npz")
    ap.add_argument("--templates", type=Path, default=ROOT / "outputs/describe_v3/templates_train.jsonl")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_2k_train.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/describe_v3/clusters_2k_train.json")
    ap.add_argument("--rep", choices=["multihot", "template", "both", "description"], default="description")
    ap.add_argument("--mh-weight", type=float, default=0.5, help="weight of the multi-hot block in --rep both")
    ap.add_argument("--mh-max-mean", type=float, default=0.9,
                    help="drop multi-hot columns present on more than this fraction of questions")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--k", type=int, default=None, help="skip selection and use this k")
    ap.add_argument("--min-size", type=int, default=100, help="smallest group allowed when choosing k")
    ap.add_argument("--min-stability", type=float, default=0.6)
    ap.add_argument("--n-boot", type=int, default=10)
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--per-cluster", type=int, default=100, help="subset size per group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    vec = np.load(args.vectors)
    ids = vec["ids"].tolist()
    n = len(ids)
    x = representation(vec, args.rep, args.mh_weight, args.mh_max_mean)
    dist = cosine_distances(x)
    q_emb = vec["question"].astype(np.float64)
    disc = np.array([str(d) for d in vec["discipline"]])
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    recs = {}
    for line in args.templates.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                recs[r["id"]] = r
    print(f"{n} questions, representation {args.rep}, k in [{args.k_min}, {args.k_max}]")

    # --- sweep k
    ks = [args.k] if args.k else list(range(args.k_min, args.k_max + 1))
    table = []
    majority = np.bincount(np.unique(disc, return_inverse=True)[1]).max() / n
    print(f"\n{'k':>3} {'silhouette':>10} {'stability':>9} {'min_size':>8} {'recover_acc':>11} "
          f"{'majority':>8} {'disc_nmi':>8}")
    for k in ks:
        lab, med, cost = kmedoids(dist, k, rng)
        sizes = np.bincount(lab, minlength=k)
        row = {
            "k": k, "silhouette": float(silhouette_score(dist, lab, metric="precomputed")),
            "stability": stability(dist, k, lab, rng, args.n_boot, args.boot_frac),
            "min_size": int(sizes.min()),
            "recover_acc": recoverability(q_emb, lab, rng),
            "majority": float(sizes.max() / n),
            "disc_nmi": float(normalized_mutual_info_score(disc, lab)),
            "labels": lab, "medoids": med,
        }
        table.append(row)
        print(f"{k:>3} {row['silhouette']:>10.3f} {row['stability']:>9.3f} {row['min_size']:>8d} "
              f"{row['recover_acc']:>11.3f} {row['majority']:>8.3f} {row['disc_nmi']:>8.3f}")

    if args.k:
        chosen = table[0]
    else:
        ok = [r for r in table if r["min_size"] >= args.min_size and r["stability"] >= args.min_stability]
        if not ok:
            print(f"\nno k clears min_size>={args.min_size} and stability>={args.min_stability}; "
                  "falling back to the most stable k")
            ok = [max(table, key=lambda r: r["stability"])]
        kmax = max(r["k"] for r in ok)
        chosen = max((r for r in ok if r["k"] == kmax), key=lambda r: r["silhouette"])
    k, lab, med = chosen["k"], chosen["labels"], chosen["medoids"]
    print(f"\nchosen k = {k}")

    # --- describe each group and pick its subset
    clusters = []
    for j in range(k):
        members = np.flatnonzero(lab == j)
        sub = pick_subset(dist, members, int(med[j]), args.per_cluster, rng)
        held = [int(m) for m in members if m not in set(sub)]
        steps = Counter(s for m in members for s in recs[ids[m]]["steps"])
        challenges = Counter(ch for m in members for ch in recs[ids[m]]["challenges"])
        know = Counter(recs[ids[m]]["knowledge"] for m in members)
        discs = Counter(disc[m] for m in members)
        clusters.append({
            "cluster": j, "size": int(members.size), "medoid": ids[med[j]],
            "medoid_template": recs[ids[med[j]]]["template"],
            "steps": {s: round(c / members.size, 2) for s, c in steps.most_common()},
            "challenges": {r: round(c / members.size, 2) for r, c in challenges.most_common()},
            "knowledge": {r: round(c / members.size, 2) for r, c in know.most_common()},
            "disciplines": {d: round(c / members.size, 2) for d, c in discs.most_common(5)},
            "subset": [ids[m] for m in sub],
            "held_out": [ids[m] for m in held],
            "members": [ids[m] for m in members],
        })

    out = {
        "rep": args.rep, "mh_weight": args.mh_weight, "k": k, "seed": args.seed,
        "per_cluster": args.per_cluster,
        "sweep": [{kk: v for kk, v in r.items() if kk not in ("labels", "medoids")} for r in table],
        "labels": {ids[i]: int(lab[i]) for i in range(n)},
        "clusters": clusters,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    # --- human-readable report
    lines = [f"# Question groups: rep={args.rep}, k={k}, n={n}\n",
             "| k | silhouette | stability | min size | recover acc | majority | discipline NMI |",
             "|---|---|---|---|---|---|---|"]
    for r in out["sweep"]:
        lines.append(f"| {r['k']} | {r['silhouette']:.3f} | {r['stability']:.3f} | {r['min_size']} | "
                     f"{r['recover_acc']:.3f} | {r['majority']:.3f} | {r['disc_nmi']:.3f} |")
    for c in clusters:
        lines += [f"\n## Cluster {c['cluster']}: {c['size']} questions, subset {len(c['subset'])}, "
                  f"held out {len(c['held_out'])}",
                  f"- medoid template: {c['medoid_template']}",
                  f"- steps: {c['steps']}", f"- challenges: {c['challenges']}",
                  f"- knowledge: {c['knowledge']}", f"- disciplines: {c['disciplines']}",
                  "- sample questions:"]
        for qid in [c["medoid"]] + [q for q in rng.permutation(c["members"]).tolist()[:5] if q != c["medoid"]]:
            row = rows[qid]
            lines.append(f"  - [{row['discipline']} / {row['field']}] {row['question'][:200].strip()}")
            lines.append(f"    - {recs[qid]['template']}  (challenges {recs[qid]['challenges']})")
    rep_path = args.out.with_suffix(".md")
    rep_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {rep_path}")
    for c in clusters:
        print(f"  cluster {c['cluster']}: {c['size']:>4} q  top steps "
              f"{list(c['steps'])[:3]}  top challenge {list(c['challenges'])[:1]}  "
              f"disciplines {list(c['disciplines'])[:2]}")


if __name__ == "__main__":
    main()
