# Using compacted context and generating compactions

Lane A evaluates whether a model can use provided compact states. Lane B later trains a model to generate those states. They are separate problems. `TeacherInterface` is generic and disabled; no external model/API is invoked.

Original chunks live in an immutable, SHA256-checked external archive. Pointers like `<mem:000123>` resolve exactly. Fixtures contain identifiers, filenames, signatures, constraints, numbers, failed approaches, tool results and compiler errors, separated by long work logs. A compaction can preserve selected fields plus raw archive pointers. Repeated work/compact boundaries deliberately drop fields at ratios 2x/4x/8x.

`tools.context_lab` generates 100 tasks and 3,200 deterministic retention checks for no compaction, one/repeated compaction, and exact retrieval. Dictionary-field retention is an oracle infrastructure metric, not neural-model performance. Exact archive retrieval verifies byte recovery, but does not prove a model can find or use the right block. Lost fields and recovered fields are reported separately.

Model evaluation must use answer likelihood/exact correctness over the same tasks and context budgets, measuring degradation at every boundary. Oracle retrieval is an upper bound; learned retrieval selection needs its own evaluation. Raw storage is required for exact recovery; compact states alone are explicitly lossy. Initial pretraining uses 1k; test 2k then 4k, targeted 8k, optional 16k. No 256k training is attempted.

`tools.eval_context --checkpoint ... --context 8192` measures answer NLL and teacher-forced answer-token accuracy at each boundary. It records prompt truncation and labels oracle retrieval. Teacher-forced token accuracy is not generated task pass rate. The initial lab validates fixtures and a small 1k neural smoke; 8k quality remains a later targeted experiment.
