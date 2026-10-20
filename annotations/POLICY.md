# Agent annotation policy v2 — frozen before EEG outcomes

Scope: all English text in ZuCo 1.0 SR, NR and TSR, including factual Wikipedia
material. Review each distinct sentence and every token; reuse that annotation
for repeated readers/tasks/presentations while preserving their source keys.

The contextual target is the affective/evaluative valence carried by the word
in its current sense and construction: positive, negative, neutral or uncertain.
It is not whole-sentence sentiment, author stance, the reader's emotional state,
or a prediction of EEG. A harmful event can carry negative lexical meaning in
an otherwise factual biography. A person/place/title does not inherit the
valence of the event or an evaluative word inside its proper name.

Keep a separate usual lexical label when the contextual construction changes
it. Negation is interpreted, not mechanically inverted. Negators and ordinary
function words remain neutral as sentiment targets and receive modifier roles.
“Not good” usually carries negative evaluation on “good”; “not bad” may be
weak approval or uncertainty, depending on the whole sentence. “Not only good”
does not reverse good. “Hardly” and comparisons require the same scope review.

Words describing neutral entities do not inherit a surrounding opinion. In
“a terrible film,” terrible is negative and film remains neutral. Evaluative
words in clearly named titles, organizational names, or quoted terminology
can be neutral in use even when their ordinary lexical meaning is affective.

For irony/sarcasm, label an unambiguous intended evaluation and retain an irony
flag, or use uncertain if it cannot reliably be resolved from the supplied
sentence. Missing discourse, mixed word senses, corrupted tokens and indivisible
idioms may also require uncertain. Store the reason and any phrase span. A
bare assertion that a sentence is sarcastic is not evidence of intended polarity.

The agent reads every sentence and supplies a sparse record of all non-neutral
token decisions and lexical overrides. Unlisted tokens in a **reviewed**
sentence are explicit agent decisions of neutral; an unreviewed/missing sentence
is an error, not a neutral fallback. Compilation expands the decisions into a
complete per-token CSV. Exact surface-word checks prevent silent indexing drift.

These are single-agent annotations, not human consensus or gold labels.
Uncertainty and disagreement should be available for professor review. Preserve
the original sentence labels as provenance only and do not use them as a target
or an instruction while reviewing individual words.
