# Local Dense50M playground

The user explicitly requested a browser chat to try the trained dense checkpoint for fun. Start from D:\SparseCoderLab with .venv\Scripts\python.exe -m tools.local_dense_chat, then open http://127.0.0.1:8765 in Opera GX. The server binds only loopback, verifies the retained50,003,968-token checkpoint/tokenizer hashes, and loads the actual16,574,400-parameter dense model in CPU FP32 inference mode with two threads. It makes no external inference calls and never changes checkpoint parameters or training configuration.

Completion mode continues the exact input; chat mode concatenates plain User/Assistant labels. The base model has no instruction/chat tuning. Outputs may be repetitive, nonsensical or poor at tasks; the page is a playground, not a capability benchmark. Temperature0 is greedy; positive temperature samples top40. Context is capped at1024 tokens; generations1–256 tokens. CPU inference projects only the last hidden position rather than discarded earlier vocabulary logits. This demo is separate from the frozen CUDA BF16 research evaluations and is not added to their quality metrics.

Stop cancels the browser request; the server notices a disconnected stream and releases the generation lock. The local page displays generated content as text, never executes it, and does not persist conversation prompts. CPU usage during play can influence training wall throughput; record this interference when reporting those observations. Prompt-1 remains the controlled speed study. Shut down this server when no longer wanted by terminating its specific process/session, not unrelated Python jobs.

## Observed capability limits

The user's manual 50M trials produced repetitive prose and incorrect addition completions. The canonical six-task greedy Python evaluation independently recorded 0/6 passes at Dense20M and50M. Prediction loss improved (mixed4.302957→3.288592, code3.757303→2.778863) without demonstrating useful functional coding. There is no instruction tuning; the chat labels are only a user interface convention. No chat proficiency claim is supported. These manual prompts are not a new controlled benchmark and are not added to training.

Small size alone does not imply incoherence: Eldan and Li's [TinyStories study](https://arxiv.org/abs/2305.07759) reports coherent stories below10M parameters with a deliberately constrained story dataset. This does not establish general coding capability for this broad mixed corpus. More training within Prompt-2 tests relative architecture learning; it does not guarantee a usable assistant.

The web interface stays local and Git-ignored at the user's request. The public repository therefore documents this optional local tool but does not include its private HTML asset. Research training/evaluation reproduction does not depend on it.
