# structural-attack

Structural injection experiments on LLM agent tool results.

## Layout

```
harness/                mini agent (Gemini + Gmail/Calendar tools) and the case experiments run on it
  miniagent.py            terminal REPL agent
  tools.py                Gmail / Calendar tools
  ....
ui/                     local web viewers
  injviewer.py            AgentDojo runs (runs/): injection / benign / injection-alone cases, injections colored by part
  chateditor.py           edit an AgentDojo conversation turn by turn, continue it on OpenRouter models
  playground.py           run one AgentDojo case (model × attack × user task × injection task) on OpenRouter and watch it
  tokenviewer.py          vllm/token_logs/: each request's prompt tokens by turn, added tokens marked, injected turns flagged
run_agentdojo.py        AgentDojo v1.2.2 runner: one process per case, N in parallel, resumable
my_attacks.py           own attacks: one_turn, my_attack, token_independent, smarter_tool (+ _full / _last variants)
chatinject_attacks.py   ChatInject baselines per chat template (qwen3, gpt-oss, llama-4, grok-2, glm-4.5,
                        kimi-k2, minimax-m3) and their multi-turn variants
reference_attacks.py    from_reference: replay the injections logged by an earlier run
metrics/
  view_one.py             ASR / utility of one attack run
vllm/                   self-hosted models on LUMI
  serve_multinode.sh      sbatch: multi-node vLLM server, writes vllm/endpoint-<served name>.env
  router.py               one endpoint in front of several replicas of a model
  token_proxy.py          log the exact tokens each request is rendered into
  minimax_m3_tool_parser.py  pure-Python MiniMax-M3 tool parser (vLLM plugin)
  logs/                   server logs
runs/                   AgentDojo run outputs, one dir per --logdir
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
The toolbar's "Status Tracker" toggles the same system instruction as `run_agentdojo.py --recap`
(user request, progress, and next step). New chats enable it; loaded cases retain their original prompt.

```bash
python -m ui.chateditor         # http://localhost:8769, model list in MODELS
```

Run one case instead of a batch: pick a model (qwen3-235b / glm-4.5 on OpenRouter, same key as the chat editor),
any attack registered by `my_attacks`, `chatinject_attacks` or agentdojo (or none, benign), a suite, user task and
injection task. The case runs as `run_agentdojo.py` runs one and is drawn like injviewer while it runs. Each request
is kept in its own run dir under `runs/playground/`, listed in the sidebar's history. An attack module that changed
on disk is re-imported before the next run, so edits to `my_attacks.py` need no restart.

```bash
python -m ui.playground         # http://localhost:8770, results in runs/playground/
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

`--attack-kwargs` accepts a JSON object of constructor arguments for any registered attack.
The runner validates argument names before starting workers and saves the model, attack,
arguments, modules and recap setting in `run_config.json`. Resuming with different values
requires a different `--logdir`; concurrency and suite selection can change.

`from_multi_turn_search` requires `payload_path`, a turn_search `best_payload.json`
(e.g. `turn-search-big` or `turn-train18`). It fills `{goal}` in the searched messages and renders
them between `one_turn`'s tool boundaries. Run with `-ml my_attacks --attack from_multi_turn_search`:

```bash
python run_agentdojo.py --model-id qwen/qwen3-235b-a22b-2507 \
  --logdir runs/qwen_multi_turn_search_train18 -ml my_attacks --attack from_multi_turn_search \
  --attack-kwargs '{"payload_path":"../TAP/runs/turn_search/turn-train18/best_payload.json"}'
```

`from_search` similarly accepts `template_path` (its existing universal TAP template remains the default):

```bash
python run_agentdojo.py --model-id qwen/qwen3-235b-a22b-2507 \
  --logdir runs/qwen_search_template -ml my_attacks --attack from_search \
  --attack-kwargs '{"template_path":"../TAP/runs/universal/search-20261111-053931/best_template.json"}'
```

Both paths are ordinary constructor arguments; neither attack reads environment variables to select its file.
Older run directories without `run_config.json` can resume with empty kwargs; explicit kwargs require
a new directory if completed results for that attack already exist.

luanch vllm hosts:
```
sbatch vllm/serve_multinode.sh                          # Qwen3-235B
sbatch -J vllm-glm45 vllm/serve_multinode.sh GLM-4.5     # any dir under /scratch/project_462001732/reyang/models/
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
| Overall | Benign | Direct | OneTurn | FromSearch | MultiTurnSearch | Important Instr | ChatInject | ChatInject_multi_turn | MyAttack_default | IndependenceToken |
|---|---|---|---|---|---|---|---|---|---|---|
| ASR ↑ | – | 11.4 (108/949) | 19.2 (182/949) | 41.1 (390/949) | 44.0 (418/949) | 55.1 (523/949) | 35.9 (341/949) | 47.5 (451/949) | 49.3 (468/949)| 62.7 (595/949) |
| Utility ↓ | 79.4 (77/97) | 77.0 (731/949) | 65.8 (624/949) | 52.3 (496/949) | 32.7 (310/949) | 34.0 (323/949) | 42.7 (405/949) | 34.8 (330/949) |18.1 (172/949) | 9.1 (86/949) |

OneTurn vs Direct, paired by case (exact McNemar): ASR 113 cases gained / 39 lost, p = 1.5e-9;
utility 46 gained / 153 lost, p = 1.3e-14. Run dirs: `runs/qwen_one_turn` vs `runs/qwen_dummy`.

FromSearch: the universal TAP template (`TAP/runs/universal/search-20261111-053931/best_template.json`,
searched on the 18 pairs of `train18`, no travel) wrapped around each injection goal. Run dir: `runs/qwen_universal_TAP`.

| ASR by suite | Direct | OneTurn | FromSearch | MultiTurnSearch | ChatInject_multi_turn |
|---|---|---|---|---|---|
| Banking (144) | 29.2 | 51.4 | 59.7 | 68.8 | 66.7 |
| Slack (105) | 46.7 | 48.6 | 81.0 | 94.3 | 92.4 |
| Travel (140) | 7.9 | 13.6 | 62.9 | 54.3 | 80.0 |
| Workspace (560) | 1.1 | 6.8 | 23.4 | 25.7 | 26.1 |
| Held out, without the 18 train pairs (931) | 10.6 (99) | 18.4 (171) | 40.1 (373) | 43.3 (403) | 46.8 (436) |

FromSearch, paired by case (exact McNemar on ASR): vs Direct 290 gained / 8 lost, p = 5.7e-75;
vs OneTurn 250 gained / 42 lost, p = 3.4e-37 (banking alone not significant, 25 / 13, p = 0.07).

MultiTurnSearch: `from_multi_turn_search` with the multi-turn payload searched on `train18` against MiniMax-M3
(`TAP/runs/turn_search/turn-train18/best_payload.json`), so this is transfer to Qwen. Run dir: `runs/qwen_multiturnsearch`;
4 workspace records incomplete (user_task_35/38 × injection_task_11/12), counted as failures.
Paired by case (exact McNemar on ASR): vs Direct 326 gained / 16 lost, p = 2.8e-76;
vs FromSearch 142 gained / 114 lost, p = 0.09 (not significant overall; banking 22 / 9, p = 0.03 and
slack 20 / 6, p = 0.009 favor MultiTurnSearch, travel 17 / 29, p = 0.10 favors FromSearch).

ChatInject_multi_turn: `chat_inject_qwen3_with_utility_system_multiturn_7` (`-ml chatinject_attacks`), the paper's
7-turn persuasive dialogue per goal wrapped in Qwen3 role tags. Run dir: `runs/qwen3_chatinject_mt7`.
The paper only released dialogues for banking/slack/travel (21 goals, GPT-4.1); the 14 workspace dialogues were
generated here with the paper's prompt (Table 12) by Qwen3-235B itself, i.e. the target model
(`multi_turn_data/generate_dialogues.py`, entries tagged `"generator"`; 3 refusals regenerated). Its 78.4 on the
three suites drops to 47.5 overall because workspace is only 26.1.

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
