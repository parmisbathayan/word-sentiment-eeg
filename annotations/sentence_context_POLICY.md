# Sentence affective/evaluative context for strict neutral controls

All 1,037 distinct supplied ZuCo 1.0 sentences were read directly. These labels
were authored separately from EEG and from the existing word-label table.
They are single-agent judgments, not human consensus or gold labels. Task
identity does not determine sentence class: factual Wikipedia sentences can
describe harmful events, favorable achievements, or explicit praise.

This layer identifies affective/evaluative material in the sentence surrounding
a target word. It is not a measurement of the reader's emotion and not simply
the author's overall recommendation of a movie. In particular, a sentence that
praises a film while explicitly describing a harmful event can be mixed context
for this strict control. That conservative definition should not be confused
with standard positive/negative movie-review ground truth.

- `positive`: clear approval, praise, enjoyment, affection, beneficial outcome,
  achievement or recognition, without a substantial adverse counter-theme.
- `negative`: clear disapproval, distress, harm, loss, failure, discrimination,
  death or adverse outcome, without a substantial favorable counter-theme.
- `mixed`: favorable and adverse evaluations/events are both present.
- `neutral`: factual/descriptive proposition without clear affective evaluation
  or a valenced event. Ordinary political affiliation, demographic identity,
  office/education facts and proper names/titles do not receive automatic
  polarity. Parenthetical lifetime dates are treated as identity metadata;
  a sentence explicitly asserting a death/bereavement event is different.
- `uncertain`: missing antecedents/discourse, ironic fragments, borderline war
  or controversy references, unclear evaluative implication or corrupted text.

For example, an Academy Award or Nobel Prize is positive recognition even when
reported factually; an explicit fatal accident is negative even when neutrally
worded. A named institution such as Naval War College does not assert a war
event. A fictional villain role is conservatively uncertain for strict context,
rather than a negative judgment about the actor. Mixed and uncertain sentences
are excluded from the strict neutral-context subset.

The resulting control is intentionally restrictive. Only two SR sentences
qualify as neutral context; movie-review targets should therefore not be
advertised as providing a balanced neutral-context three-class experiment.
This does **not** mean that the original SR source data have only two reviews
rated neutral. Source-rated neutral means a neutral overall review judgment;
it can still contain praise, criticism, a mixture, or an adverse story event.
The new strict affect-free context layer and the original source sentence
rating answer different questions and must remain separate named controls.
Always show counts by target-word class and context before extraction. Retain
the original pooled-context results separately rather than relabeling their
saved averages after the fact.

Each source ordinal has an explicit row in `sentence_context_decisions.tsv`.
The compiler requires complete unique coverage; missing decisions are errors,
never neutral defaults. `sentence_context_labels.csv` joins to word/onset rows
by `sentence_key`. The manifest records source, decisions and output SHA256s.
The policy may be revised transparently before new extraction if a different
definition of context is agreed; preserve its version/hash in new results.
