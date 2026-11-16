# Verified SST versus agent-label examples

Checked against the original SST archive with exact sentence/word joining in the cloud, not hypothetical examples. Agent labels are the saved v2review and are not human-consensus gold labels. [Verification run](https://github.com/parmisbathayan/word-sentiment-eeg/actions/runs/43840923878).

1. **convincing**, sentence key `c22c02ee4ac1d562622b542f`, word index6. Sentence: “Weiss and Speck never make a convincing case for the relevance of these two20th-century footnotes.” Original text has a space between “two” and “20th-century”. Original SST phrase ID5749, score0.63889, five-class+1/positive. Agent lexicalpositive, contextualnegative; excluded by lexical/context agreement. The phrase denies that a convincing case was made. This verifies the label-selection difference, not survival of all fixation-quality/matching rules.
2. **Sweet**, sentence key `c5949bfbc89066104e1d0362`, word index2, inside the movie title *Bread, My Sweet*. SST phrase ID26683, score0.73611, five-class+1/positive. Agent lexicalpositive, contextualneutral because it names a film; excluded by agreement. Whether this semantic judgment improves EEG decoding is not established.

The same audit found `improve` SSTneutral versus agent lexicalpositive/contextualnegative, and `Killed` SSTneutral versus agent lexicalnegative/contextualneutral. Those do not illustrate SSTpositive/negative words being excluded by the agent, so they were not used as examples of that direction.
