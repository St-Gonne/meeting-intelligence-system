# What broke, what changed, what remains

These lessons come from repeated private use and synthetic failure tests. This release shares the resulting code and generalized observations, not the meetings or people involved.

| Problem encountered | What the code now does | What remains |
|---|---|---|
| A valid, growing audio container could contain silence after source loss | Guarded capture, explicit source-loss evidence, segment manifests and interruption warnings | More device/OS testing; sleep cannot be treated as safe capture |
| Transcribing retained audio looked like evidence that the whole meeting was captured | Recording completeness is shown separately from transcript/report/brief state | Better ways to measure live coverage without recording extra private signals |
| Failed or partial starts were easy to miss | Separate failed-start and active-interruption alerts, terminal fallback and durable state | Field tests of whether people notice and understand the warning |
| Phone delivery could be confused with successful processing | Immutable revision checks, settlement, grouping review, normalization and processing are separate | Simpler setup and more recorder formats |
| A retry could expand beyond the recording the operator meant | Exact selection, source revalidation and per-operation confirmation | More clean-machine and failure-injection coverage |
| Speaker labels or context encouraged invented identities and action owners | Anonymous normal processing, untrusted-input boundaries, strict owner normalization and explicit Voice ID confirmation | Better attribution evaluation, especially overlap and more than two people |
| Hindi-English speech exposed transcription weaknesses | Adopted a separate Qwen phone runtime, kept lineage and explicit failure instead of silent fallback | A shareable consented accuracy benchmark; no published WER claim yet |
| A large model could silently lose part of a long input | Non-truncating request policy, completion and token-budget checks | Better segmentation and quality-preserving long-meeting strategies |
| Local models competed for unified memory | Shared job-level GPU admission, supervised children, unload and endpoint checks | Portable single/multiple service setup and measured memory/latency results |
| “Learning from usage” could turn into unjustified automatic changes | Content-free local event records and bounded review windows | Useful recommendations from sparse usage; no self-modifying product claim |

There is no public claim that all capture hardware, languages, group calls or new-owner Voice ID are solved. Good contributions should make one of these limits measurable and then improve it without hiding failures.
