"""Step 2 of the per-type search: turn each question's description into vectors.

Four representations, saved side by side so the clustering step can compare
them:

  multihot   20 interpretable dimensions from the describer's record: which
             reasoning steps appear (10), which reasoning challenges (7), and
             the knowledge type (3). No model needed.
  template   embedding of the scrubbed template paragraph, from
             microsoft/harrier-oss-v1-0.6b (last-token pooling, L2-normalised,
             1024 dims). What the template says about the solution path,
             in a form that can be compared by cosine.
  question   embedding of the raw question and options, same model. Not used
             for grouping (it carries topic). Kept for the test-time router
             and for the "can the groups be recovered from the raw question"
             check in the clustering step.
  description  embedding of the whole record written out as one passage
             (description_text): the reasoning steps and the reasoning
             challenges as their definition sentences, in the describer's
             order, the knowledge type, and the template. Same model.

The embedding model runs on whatever CUDA_VISIBLE_DEVICES exposes, or CPU.
The 0.6B model needs ~2 GB; on CPU the train split takes a few minutes.

    python3 scripts/embed_questions.py \\
        --describer describe_supergpqa_v3 \\
        --templates outputs/describe_v3/templates_train.jsonl \\
        --dataset datasets/supergpqa_2k_train.json \\
        --out outputs/describe_v3/vectors_2k_train.npz
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import debate_mcq as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "microsoft/harrier-oss-v1-0.6b"
# Harrier wants an instruction on queries. Every text here is compared with
# texts of its own kind, so all sides get the same symmetric-similarity prefix.
INSTRUCT = "Instruct: Retrieve semantically similar text\nQuery: "

# The describer's vocabulary, prompt version and question layout: set by use_describer from the
# dataset's own describer file (--describer; one file per dataset since 2026-10-07, so the SuperGPQA
# and HLE vocabularies can never be mixed up).
STEPS: dict[str, str] = {}
CHALLENGES: dict[str, str] = {}
KNOWLEDGE: tuple = ()
PROMPT_VERSION = ""
MULTIHOT_NAMES: list[str] = []


def question_text(row: dict) -> str:
    """The question as a solver sees it: with its lettered options if it has an option list, as
    written otherwise (the HLE describer's question_text; the SuperGPQA describer renders every
    question with its options, which is the same thing for its rows)."""
    return D.render_question(row["question"], row["options"]) if row["options"] else row["question"]


def use_describer(name: str) -> None:
    """Take the vocabulary and prompt version of the describer module `name` (describe_supergpqa_v3,
    describe_hle_v4, ...)."""
    global STEPS, CHALLENGES, KNOWLEDGE, PROMPT_VERSION, MULTIHOT_NAMES
    d = importlib.import_module(name)
    STEPS, CHALLENGES, KNOWLEDGE, PROMPT_VERSION = d.STEPS, d.CHALLENGES, d.KNOWLEDGE, d.PROMPT_VERSION
    MULTIHOT_NAMES = ([f"step:{s}" for s in STEPS] + [f"challenge:{c}" for c in CHALLENGES]
                      + [f"knowledge:{k}" for k in KNOWLEDGE])


def multihot(rec: dict) -> np.ndarray:
    v = np.zeros(len(MULTIHOT_NAMES), dtype=np.float32)
    for s in rec["steps"]:
        v[MULTIHOT_NAMES.index(f"step:{s}")] = 1.0
    for c in rec["challenges"]:
        v[MULTIHOT_NAMES.index(f"challenge:{c}")] = 1.0
    v[MULTIHOT_NAMES.index(f"knowledge:{rec['knowledge']}")] = 1.0
    return v


KNOWLEDGE_TEXT = {"recall": "The answer is mainly recalled.",
                  "derive": "The answer is mainly derived.",
                  "both": "The answer is both recalled and derived."}


def description_text(rec: dict, exclude: frozenset[str] = frozenset()) -> str:
    """The record as one natural-language passage: each label is written as
    its definition sentence, never as its key. Challenges named in `exclude`
    are left out (the rest keep their order); if none is left, so is the
    challenges paragraph."""
    steps = "\n".join(f"{i}. {STEPS[s]}" for i, s in enumerate(rec["steps"], 1))
    kept = [c for c in rec["challenges"] if c not in exclude]
    challenges = "\n".join(f"{i}. {CHALLENGES[c]}" for i, c in enumerate(kept, 1))
    return ("Reasoning steps, in the order a careful solver takes them:\n" + steps
            + ("\n\nReasoning challenges, the most likely ways to get it wrong:\n" + challenges
               if kept else "")
            + "\n\nKnowledge: " + KNOWLEDGE_TEXT[rec["knowledge"]]
            + "\n\nReasoning template: " + rec["template"])


class Embedder:
    def __init__(self, device: str, max_length: int = 2048):
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL, dtype="auto").to(device).eval()
        self.device, self.max_length = device, max_length

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), unit="batch", leave=False):
            batch = self.tok([INSTRUCT + t for t in texts[i:i + batch_size]], padding=True,
                             truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            hidden = self.model(**batch).last_hidden_state
            # last-token pooling that is correct under either padding side
            mask = batch["attention_mask"]
            if bool((mask[:, -1] == 1).all()):                 # left padding
                pooled = hidden[:, -1]
            else:                                              # right padding
                last = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0)


def load_records(path: Path) -> dict[str, dict]:
    recs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("prompt_version") == PROMPT_VERSION and "error" not in r:
                recs[r["id"]] = r
    return recs


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--describer", required=True)
    use_describer(pre.parse_known_args()[0].describer)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--describer", required=True,
                    help="the dataset's describer module, whose records --templates holds "
                         "(describe_supergpqa_v3, describe_hle_v4, ...)")
    ap.add_argument("--templates", type=Path, default=ROOT / "outputs/describe_v3/templates_train.jsonl")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_2k_train.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/describe_v3/vectors_2k_train.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N described questions (smoke test)")
    ap.add_argument("--exclude-challenge", action="append", default=[], choices=list(CHALLENGES),
                    metavar="KEY", help="leave this challenge out of the description text (repeatable); "
                                        "the multi-hot columns and stored records are unchanged")
    args = ap.parse_args()

    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    recs = load_records(args.templates)
    ids = [i for i in rows if i in recs]
    missing = len(rows) - len(ids)
    if args.limit is not None:
        ids = ids[: args.limit]
    print(f"{len(ids)} questions with descriptions ({missing} without, skipped)")

    mh = np.stack([multihot(recs[i]) for i in ids])
    emb = Embedder(args.device)
    print(f"embedding templates on {args.device}")
    t_emb = emb([recs[i]["template"] for i in ids], args.batch_size)
    print("embedding raw questions")
    q_emb = emb([question_text(rows[i]) for i in ids],
                args.batch_size)
    print("embedding full descriptions")
    exclude = frozenset(args.exclude_challenge)
    if exclude:
        print(f"left out of the description text: {sorted(exclude)}")
    d_texts = [description_text(recs[i], exclude) for i in ids]
    d_emb = emb(d_texts, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, ids=np.array(ids), multihot=mh, template=t_emb, question=q_emb,
        description=d_emb, multihot_names=np.array(MULTIHOT_NAMES),
        templates=np.array([recs[i]["template"] for i in ids]),
        description_texts=np.array(d_texts),
        discipline=np.array([rows[i]["discipline"] for i in ids]),
        field=np.array([rows[i]["field"] for i in ids]),
        difficulty=np.array([rows[i]["difficulty"] for i in ids]),
        embed_model=np.array(EMBED_MODEL), describer=np.array(args.describer),
        prompt_version=np.array(PROMPT_VERSION),
        excluded_challenges=np.array(sorted(exclude), dtype=str),
    )
    print(f"wrote {args.out}: multihot {mh.shape}, template {t_emb.shape}, question {q_emb.shape}, "
          f"description {d_emb.shape}")
    print(f"example description text:\n{d_texts[0]}\n")
    print("multi-hot column means (how often each label appears):")
    for name, m in zip(MULTIHOT_NAMES, mh.mean(axis=0)):
        print(f"  {name:28s} {m:.2f}")


if __name__ == "__main__":
    main()
