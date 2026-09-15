# AOLLamaTritLLM — an Active Oberon port of LLamaTritLLM

A port of [virex-84/LLamaTritLLM](https://github.com/virex-84) to Active Oberon:
a BitNet-1.58-style ternary Llama trained on CPU, with no ML framework, no GPU
and no Python. The trained model file is measured in kilobytes.

The original is C#; there is also a [Go port](../GoLLamaTritLLM). All three write
the **same binary model files**, and that is checked rather than claimed: this
implementation loads a `ternary_model.bin` produced by the Go one and answers
from it.

## Build and run

Needs the [minia2 SDK](https://github.com/active-oberon/minia2) (`ob` on the PATH).

```sh
ob test Trit*.Mod                 # 32 cases
ob build TritTrain.Mod -o trittrain
ob build TritInfer.Mod -o tritinfer

./trittrain                       # interactive menu
./tritinfer models/ternary_model.bin --prompt привет
```

Typical flow in the training menu: **create model → pretrain → tune → save →
export ternary → chat**. Data files are read from `data/` relative to the
working directory.

## Modules

A2's namespace is flat and a project is one directory, so every module carries
the `Trit` prefix and there are no subdirectories.

| Module | What |
|---|---|
| `TritIO` | array types, the .NET binary encoding, FP16 conversion |
| `TritMap` | string-keyed hash map and string list |
| `TritConfig` | model geometry and its file layout |
| `TritPacker` | multi-plane ternary weights, base-3 packing |
| `TritBPE` | byte-level BPE with GPT-2 pre-tokenization |
| `TritOps` `TritLinear` `TritAttn` `TritFFN` `TritBlock` `TritModel` | the inference model |
| `TritTensor` `TritGrad` | autograd nodes and the differentiable operations |
| `TritAdam` `TritTrainBlock` `TritTrainModel` `TritTrainer` | the FP32 master model and training |
| `TritData` | the JSON corpus loader |
| `TritTrain` `TritInfer` | the two commands |

## Verified

With the default config (`emb=64, hidden=64, heads=1, kv=1, layers=1, seq=512,
planes=1`) this port reproduces the article's figures, and the Go port's, exactly:

| | Article (C#) | Go port | This port |
|---|---|---|---|
| Latent parameters | 106 496 | 106 496 | 106 496 |
| BPE merges on this corpus | — | 953 | 953 |
| `master_model.bin` | 420 KB | 430 707 B | 430 707 B |
| `ternary_model.bin` | 28.9 KB | 29 626 B | 29 626 B |

Training is about twice as slow as the Go port on the same machine.

## What the language cost, and what it did not

The three things I expected to be missing were mostly not:

- **Generics exist**, as parameterized modules (`MODULE M(TYPE T, CONST Less)`).
  Not needed here in the end.
- **JSON is in the standard library** (`JSON.Mod`, RFC 8259), so the corpus
  loader is fifteen lines.
- **Regular expressions are too** (`Regex.Mod`, a Thompson NFA), including
  `\p{L}` and `\p{N}` against Cyrillic — which is most of the GPT-2
  pre-tokenizing pattern.

What actually cost something:

- **No closures.** The language report is explicit: a procedure assigned to a
  procedure variable "must not be local to another procedure". Where C# and Go
  append a closure per operation, `TritTensor.Node` is a base class and every
  operation in `TritGrad` is a subtype that keeps what it captured in fields and
  overrides `Backward`. This is the one structural rewrite in the port.
- **No map keyed by string content.** `FoxBasic.HashTable` hashes the pointer and
  `StringPool` interns into one global pool that BPE's throwaway pair keys would
  never leave, so `TritMap` is 160 lines of open addressing. It is the only
  library this port had to bring.
- **No lookahead in the regex engine**, exactly as in Go's RE2: .NET's
  `\s+(?!\S)` is emulated in `TritBPE.Pretokenize`, and
  `TestPretokenizeWhitespace` pins the equivalence on nine cases.
- **No population count.** `TritPacker.PopCount` is SWAR. Only the
  trit-against-trit dot product uses it and inference does not, so it does not
  matter; marked `ponytail:` in the source.

## Three things worth knowing

- `UTF8Strings.DecodeChar` reports success on the terminating `0X` as well — NUL
  is a code point like any other — so every walk over a string here stops on
  `cp = 0` rather than on the call returning `FALSE`. The doc comment does not say
  so; `UTF8Strings.Valid` is the one place that shows the idiom.
- `UTF8Strings.Length` counts sequences by the lead-byte table while a decode loop
  counts characters that actually decode, so the two disagree on malformed input.
  `SplitIntoChars` sizes its array with the same walk it fills it by.
- Writing this port turned up one real defect and one stale install, both since
  settled: `UTF8Strings` walked off the end of an exactly-sized array on malformed
  input (minia2 `94cb33a`), and `ob build` linked an object compiled before the
  last edit (`c40708d`, `f089a5a`) on an SDK two weeks behind the checkout. If a
  trap reproduces against a source you have already fixed, check `ob version`
  against the tree first.

## Differences from the C# original

Same list as the Go port, plus one:

- Weights of the inference model start zeroed rather than random: every caller
  overwrites them immediately with a load or an import.
- `AttentionMode.TopK` keeps the original's behaviour of spreading `1/k` across
  every score rather than the top k. Marked `ponytail:` in the source.
- `Gelu` and the standalone `Softmax` are not ported: nothing on the Llama path
  calls either.
- No BPE cache — the C# version keeps one per encoded word.
- The random number generator is A2's, so a model trained here starts from
  different weights. Seeded, so runs are reproducible.

## License

MIT, as the original. Copyright (c) 2026 virex-84 for the original work.
