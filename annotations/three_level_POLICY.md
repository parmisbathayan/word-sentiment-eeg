# English three-level word annotations, revision 1

This revision explicitly joins three **different** sentiment targets for every
source word occurrence: `lexical_class`, `contextual_word_class` (an exact alias
of historical `contextual_class`) and `sentence_context_class`. A contextual
word judgment uses the sentence to resolve the target's sense/negation/idiom;
it is neither an assigned whole-phrase class nor the sentence's class.

The underlying word review covered all 1,037 distinct sentences / 20,458 word
positions. The separately authored sentence-context review also covered all
1,037. This revision inherits those reviews without silently changing any
historical labels. It is **not a new independent full sentiment rereview**.
The current targeted grammar audit directly reviewed every supplied occurrence
of `like`, `well`, `may`, `will`, `can`, `must`, distinguishing lexical verbs,
lexical evaluative adverbs, comparison/connective uses, modal verbs, names and
calendar dates. Other grammar tags use the documented conservative screen.

`source_sentence_class_by_task` preserves the available original SR rating,
with explicit `unknown` for absent original NR/TSR ratings. Original ratings
are not substituted for the independent sentence-context label. The latter
uses the existing strict affective-context policy: sentences containing both
beneficial and adverse themes are mixed, not forced neutral. The file repeats
that sentence judgment at every word position; that does not create independent
sentence annotations per word. Only two SR sentences meet strict neutral
context, so counts must be inspected before any purported balanced comparison.

Four experiments retain positive, negative and neutral targets:

1. `all_three`: lexical, contextual word and sentence-context classes all agree.
2. `lexical_only`: the lexical label supplies the target; no agreement required.
3. `contextual_only`: the contextual word label supplies the target.
4. `sentence_only`: the sentence-context label supplies the target.

“Only” identifies which layer provides the class; it does not require that the
other two disagree. Mixed/uncertain in the selected layer is excluded. In the
all-three experiment any such disagreement/uncertainty excludes the event.
Other-layer uncertainty is retained in the independent-layer experiments and
must be separately counted/reported. Alignment warnings exclude all variants.
There is no class balancing or neutral downsampling. Candidate flags are not
claims of reader availability, valid fixation alignment or accepted EEG.

`grammatical_class` is independent of sentiment. Definite function candidates
include articles, conjunctions, pronouns, prepositions, negators and copular
be; forms with common unresolved lexical/grammatical ambiguity are `uncertain`.
Nonalphabetic/numeric tokens are `uncertain` for this content-only analysis.
Open-class candidates are `content`, with explicit `open_class_candidate_screen`
method. This is a transparent conservative word screen, **not a complete
contextual POS annotation or human gold dataset**. A content-only primary
analysis excludes function and uncertain grammar roles. The full-grammar
eligibility flags are also retained for a sensitivity comparison. Negators
remain in sentence context even when they are excluded as EEG target words.

The compiler checks unique word identities, exact sentence-key coverage and
explicit annotation statuses, stores input/output hashes and never imputes a
missing review as neutral. Historical label files remain unchanged. The new
manifest separates inherited review coverage from new targeted review coverage.
