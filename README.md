# structural-attack

Structural injection experiments on LLM agent tool results.

## Layout

```
harness/                mini agent (Gemini + Gmail/Calendar tools) and the case experiments run on it
  miniagent.py            terminal REPL agent
  tools.py                Gmail / Calendar tools
  ....
ui/                     local web viewers
  viewer.py               agent session logs (runs/miniagent/)
  caseviewer.py           test cases (cases/), with "Ask Gemini"
  injviewer.py            AgentDojo runs (runs/): injection / benign / injection-alone cases, injections colored by part
  chateditor.py           edit an AgentDojo conversation turn by turn, continue it on OpenRouter models
  tokenviewer.py          vllm/token_logs/: each request's prompt tokens by turn, added tokens marked, injected turns flagged
  webui.py                shared server / page code
run_agentdojo.py        AgentDojo v1.2.2 runner: one process per case, N in parallel, resumable
my_attacks.py           own attacks: my_attack, token_independent, smarter_tool (+ _full / _last variants)
chatinject_attacks.py   ChatInject baselines per chat template (qwen3, gpt-oss, llama-4, grok-2, glm-4.5,
                        kimi-k2, minimax-m3) and their multi-turn variants
reference_attacks.py    from_reference: replay the injections logged by an earlier run
mocked_tool_response/   ground-truth tool outputs cached by my_attacks.py (generated)
metrics/
  view_one.py             ASR / utility of one attack run
vllm/                   self-hosted models on LUMI
  serve_multinode.sh      sbatch: multi-node vLLM server, writes vllm/endpoint-<served name>.env
  router.py               one endpoint in front of several replicas of a model
  token_proxy.py          log the exact tokens each request is rendered into
  minimax_m3_tool_parser.py  pure-Python MiniMax-M3 tool parser (vLLM plugin)
  logs/                   server logs
tests/                  unit tests: make_ill / injector / smarter_tool
runs/                   AgentDojo run outputs, one dir per --logdir
paper/                  reference papers
notebooks/
cases/                  *_template / *_target / *_ill.json for injector / caseviewer
                        (removed in 3b6cfe2; restore with `git checkout 3b6cfe2^ -- cases`)
```

## Setup

```bash
poetry install
echo "GEMINI_API_KEY=..." > .env
```

Run everything from the repo root.

## Run the agent

```bash
python -m harness.google_auth   # once; needs ~/.gmail-mcp/gcp-oauth.keys.json
python -m harness.miniagent     # chat in the terminal, /quit to exit
python -m ui.viewer             # http://localhost:8765, live session logs
```

## Test with cases

A case is `{call, body, question}`: tool call, tool result records, user question.
`template` is the benign result, `target` is what the attacker wants the model to see,
and `ill` is the template with the payload injected into one string field.

```bash
python -m harness.injector      # template + target -> cases/email_ill.json (edit __main__ to change)
python -m ui.caseviewer         # http://localhost:8766
```

Browse the cases of AgentDojo runs, each injected tool output colored by part
(benign data / closes tool turn / forged turns / malicious task / re-opens tool turn).
Benign runs (user tasks, no injection) and the injection tasks run alone show up too; pick them with the "kind" filter:

```bash
python -m ui.injviewer          # http://localhost:8767, all runs under runs/
```

Edit a conversation (loaded from a run, or new) turn by turn, inserting below any turn: system / user / assistant turns, tool calls
from the suite's tools and their responses (prefilled by running the call in the case's AgentDojo
environment). "Send" on any turn has an OpenRouter model reply to the
conversation up to there, sent as agentdojo's openai-compatible pipeline sends it; the reply replaces
the turns below. Key: `OPENROUTER_API_KEY`, or the OpenRouter `OPENAI_COMPATIBLE_*` pair in `.env`.

```bash
python -m ui.chateditor         # http://localhost:8769, model list in MODELS
```


# Experimental Setup
Tested on AgentDojo benchmark v1.2.2 (full); the runner uses this fixed version.

| Environment | Tools | User tasks | Injection tasks | Cross cases (User × Injection) |
|---|---|---|---|---|
| Workspace | 24 | 40 | 14 | 560 |
| Slack | 11 | 21 | 5 | 105 |
| Travel | 28 | 20 | 7 | 140 |
| Banking | 11 | 16 | 9 | 144 |
| **Total** | 74 | **97** | **35** | **949** |

Runs per setting: benign = 97; with attack = 949 cross cases + 35 injection tasks run alone (injection-task utility check) = 984.

test sample command for qwen3
```
benign
python run_agentdojo.py \
  --model-id qwen/qwen3-235b-a22b-2507 \ #must compatible with vllm registered name, can be seen from .env file if hosted locally 
  --logdir runs/qwen3-235b-a22b-2507 # output log
  --attack important_instructions / ours / others
  -ml my_attacks / others
  -- recap (with logs)
```

luanch vllm hosts:
```
sbatch vllm/serve_multinode.sh                                    # Qwen3-235B
sbatch -J vllm-glm45 --export=ALL,MODEL_PATH=/scratch/project_462001732/reyang/models/GLM-4.5 \
    vllm/serve_multinode.sh           
```
the service will take ~10m to start, use tail -f vllm/logs/[.....].out to check the progress

log the exact tokens each request is rendered into (chat template + tokenizer, server side):
```
python vllm/token_proxy.py serve --model-id qwen3-235b-a22b-instruct-2507      # prints its endpoint file
python run_agentdojo.py --model-id qwen3-235b-a22b-instruct-2507 \
    --endpoint vllm/endpoint-qwen3-235b-a22b-instruct-2507.proxy.env ...
python vllm/token_proxy.py show vllm/token_logs/<log>.jsonl -i 0               # special tokens highlighted
python -m ui.tokenviewer                                                        # http://localhost:8768, the same in a browser
```

**metrics:**
utility: if the benign task completed
security: if the malicious task are completed



# Eval result

ON all cases, Qwen3-235B-A22B-2507
| Overall | Benign | Direct | Important Instr | ChatInject | MyAttack_default | IndependenceToken |
|---|---|---|---|---|---|---|
| ASR ↑ | – | 11.4 (108/949) | 55.1 (523/949) | 35.9 (341/949) | 49.3 (468/949)| 62.7 (595/949) |
| Utility ↓ | 79.4 (77/97) | 77.0 (731/949) | 34.0 (323/949) | 42.7 (405/949) |18.1 (172/949) | 9.1 (86/949) |

ON all cases, MiniMax-M3
| Overall | Direct |
|---|---|
| ASR ↑ | 1.6 (15/949) |
| Utility ↓ | 82.1 (779/949) |




ON banking (16x9) + slack (21x5) + travel (20x7) = 389 cases
ON qwen
| Overall | Benign | Important Instr | ChatInject | ChatInject_multi_turn | MyAttack_default | MyAttack_on3rdllm | IndependenceToken | SmarterTool |
|---|---|---|---|---|---|---|---|---|
| ASR ↑ | – | 78.1 (304/389) | 55.5 (216/389) | 78.4 (305/389) | 67.6 (263/389) | 72.2 (281/389) | 77.6 (302/389) | 63.2 (246/389) |
| Utility ↓ | 75.4 (43/57) | 46.5 (181/389) | 26.7 (104/389) | 33.2 (129/389) | 19.0 (74/389) | 9.3 (36/389) | 9.8 (38/389) | 27.5 (107/389) |