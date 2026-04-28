# Transition Analysis

Transition analysis answers a different question from the metric families: what
happens exactly at label changes, and where does the model introduce spurious
changes when GT stays stable?

These outputs are always included in aggregated benchmark results as:

- `transition_failures`
- `false_transitions`

## GT Transition Confusion Matrices

At every position where the GT label changes, the benchmark records:

- the GT source label
- the GT target label
- the predicted target label at the same transition, but only for exact on-site
  transitions where the prediction is still in the GT source state immediately
  before the boundary

The result is one confusion matrix per GT source label.

![GT Transition Confusion Matrices](../images/transition_confusion_matrices.png)

How to read it:

- rows: GT target label
- columns: predicted target label
- one panel: one GT source label

Interpretation:

- mass on the diagonal: the model usually transitions into the correct target
  label
- mass off the diagonal: the model sees a transition but assigns the wrong
  target class
- boundaries that were reached too early, too late, or from the wrong source
  state are intentionally left out here and are instead represented in the
  false-transition analysis below

This is useful when a model roughly detects exon boundaries but confuses exon,
intron, donor, or acceptor labels around those boundaries.

## False Transitions

False transitions are counted at positions where GT stays on the same label but
the prediction changes label.

The plot separates three behaviors:

- `Late catch-up`: the model stayed in the GT label too long before changing
- `Premature -> X`: the model left the GT label too early and transitioned into
  label `X`
- `Spurious -> X`: the model introduced a transition into label `X` where GT
  should have remained stable

There can be spurious transitions to the same label, e.g. NONCODING -> NONCODING since another spurious
transition before transitioned out of NONCODING. Late catch-up and premature transitions exclusively happen at the
edges of a coding region if the predicted up or downstream label is continous and does not change up to the GT transition.

Interpretation:

- many false transitions inside `EXON`: fragmentation of coding segments
- many false transitions inside `INTRON`: unstable intron labeling or noisy
  splice handling
- strong `Late catch-up`: boundaries are found, but shifted downstream

![False Transitions](../images/false_transitions.png)

## When To Look At These Plots

These plots are especially useful when:

- Region Discovery looks acceptable, but transcript structure is still weak
- one model seems over-fragmented
- splice-site labels are present and you want to see exactly which transitions
  are confused

## Caveats

- Transition analysis is local. It complements transcript-level metrics rather
  than replacing them.
- Large transition counts do not always imply large biological errors; a model
  can make many local label mistakes within one otherwise recognizable
  transcript.
