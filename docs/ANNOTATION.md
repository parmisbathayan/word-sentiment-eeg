# Word annotation protocol

The lexical target and the contextual target answer separate questions. Preserve
both. Annotation uses text only, never the EEG or test prediction.

1. Keep the dataset/task, sentence, original word index and stable `item_id`.
   All readers of this item share the same text annotation.
2. The first run maps exact normalized words to NRC VAD v1 scores. Missing
   entries are unknown; scores between the class ranges are ambiguous.
3. Nearby negation, contrast or mixed polarities trigger text review. They do
   not automatically reverse or assign sentiment. A whole-sentence label is
   never copied to a word.
4. `contextual_class` starts as `unreviewed`. Optional values after review are
   `negative`, `neutral`, `positive`, `uncertain`, or `exclude`.
5. A contextual label describes the sentiment contribution of the target word
   in the given sentence. It does not describe the whole review, the subject's
   feelings, or grammatical positivity/negativity.
6. For negation scope, idioms, sarcasm, ambiguous word sense or insufficient
   context, use `uncertain` with a note rather than guessing. For words that
   belong to an indivisible multiword expression, note the span; do not pretend
   the whole phrase has an independently measurable single-word label.
7. Record reviewer and rationale. Ideally obtain two independent annotations
   and adjudicate disagreements before contextual modeling. Freeze the scheme
   before inspecting EEG outcomes.

| Sentence and target | Lexical label | Contextual annotation |
|---|---|---|
| “The film was **wonderful**.” | positive if matched | positive after review |
| “The film was not **wonderful**.” | remains positive | negative/uncertain depending on intended scheme; needs review |
| “A **wonderful** waste of time.” | remains positive | ironic/uncertain; needs review |
| “A **killer** performance.” | possible dictionary mismatch | sense-dependent; review |
| An unlisted proper name | unknown | do not automatically call it neutral |

The default run is a conservative **lexical** screen. It excludes context flags
from probes but saves every case for audit. Human contextual annotations do not
replace this target. A later contextual experiment needs a separately frozen
question, sufficient independently annotated items and matching split controls.

To incorporate a completed review file, put it under this project's existing
Drive result folder and supply `--reviews` plus a **new run tag**. Required
columns: `item_id,contextual_class,reviewer,notes`. Repeated IDs and unknown IDs
are errors. Blank/unreviewed rows are ignored. Contextual `uncertain`/`exclude`
rows are excluded from lexical probes, but retained in the audit.
