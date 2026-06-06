# Label Studio project snapshots

Portable JSON snapshots of the annotation projects, so the **Label Studio layer is reproducible**
on any clone — without shipping the sqlite DB (which would leak the account email, password hash
and API token, and embeds non-portable absolute paths).

Each `NN_<title>.json` holds one project: its **labeling config** + every **task** with its
`data`, `predictions` (model pseudo-labels) and `annotations` (expert boxes). Image references
are `http://localhost:8081/corseacare/<file>`, which the shipped image server reproduces — so
after `make annotate` they resolve to the published dataset images. `index.json` lists the set.

| file | tasks | predictions | annotations | what it is |
|------|------:|------------:|------------:|------------|
| `03_CorSeaCare.json` | 11 | 7 | 11 | the original hand-corrected training set (same boxes as `annotations/`) |
| `04_CorSeaCare_preannot_verif.json` | 4 | 4 | 4 | pre-annotation verification round |
| `05_csc_260606_1.json` | 27 | 27 | 0 | active-learning batch — model pseudo-labels awaiting review |
| `08_csc_260606_2.json` | 21 | 21 | 0 | active-learning batch (pending) |
| `09_csc_260606_3.json` | 21 | 21 | 0 | active-learning batch (pending) |
| `10_csc_260606_4.json` | 21 | 21 | 1 | confidence-split review (sure + borderline), in progress |

## Restore on another machine

```bash
make annotate                                 # starts Label Studio (:8080) + image server (:8081)
export LABEL_STUDIO_API_KEY=<token>           # see token note below
make ls-restore                               # recreate every project (tasks + predictions + annotations)
# or selectively:  uv run python scripts/restore_label_studio.py --only CorSeaCare --skip-existing
```

> **Token note (Label Studio ≥ 1.23):** the old *Access Token* (legacy) is disabled by default.
> Create a **Personal Access Token** (Account & Settings → Personal Access Token) and export it as
> `LABEL_STUDIO_API_KEY`, or re-enable legacy tokens in *Organization → API tokens*. The restore
> script tries both the legacy (`Token`) and Bearer schemes automatically.

## Re-export after annotating

```bash
make ls-export        # re-snapshot the local LS DB into this folder (commit the diff)
```

No secrets are exported (no users, tokens, or absolute paths) — safe for a public repo.
