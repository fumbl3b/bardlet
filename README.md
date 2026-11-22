# Ye Olde Bard

A tiny GPT, written and trained from scratch on a laptop, that you can chat with
through an old-school terminal game.

```
                          Y E   O L D E   B A R D

       _.-------._         +--[ DIALOGUE ]---------------------------------+
     .'           '.       | THOU> Good morrow! What news from court?      |
    /               \      |                                               |
  ,|                 |,    | BARD> I think you, sir, sir, I am caited by.  |
 {{|   ___     ___   |}}   |       Here's beggar, his name, his consul,    |
 {{|   <o>     <o>   |}}   |       the bad that tears a harden to me.      |
 {{|        |        |}}   |                                               |
 {{|       ._.       |}}   |                                               |
  {|    ,~~~~~~~,    |}    |                                               |
   \    '-(___)-'    /     |                                               |
    \     \:::/     /      |                                               |
     '.    ':'    .'       |                                               |
  .-~~~'-._____.-'~~~-.    |                                               |
{~{~{~{~{~~{_}~~}~}~}~}~}  |                                               |
 '~{~{~{~{~~~~~}~}~}~}~'   +-----------------------------------------------+
```

## Run it

With [uv](https://docs.astral.sh/uv/) there is nothing to install; it fetches PyTorch the first time:

```
git clone https://github.com/fumbl3b/bardlet.git && cd bardlet
uv run bard.py
```

Or with your own Python: `pip install torch`, then `python bard.py`.

Type a line and press Enter. ESC cuts the Bard off mid-speech, or quits when it is your turn.

| Option | What it does |
|---|---|
| `--temperature 0.5` | Safer, more grammatical replies. Higher values (try `1.2`) are wilder. Default `0.8`. |
| `--mute` | No sound effects. |
| `--no-intro` | Skip the boot screen. |

The window needs to be at least 72 x 24. Sound works on macOS and on Linux with
`paplay` or `aplay`. On Windows, also run `pip install windows-curses`.

`chat.py` is the same chat as plain text, with no animation.

## What to expect

The Bard sounds like Shakespeare and does not understand a word you say. The model
only knows how to continue a play script, so the game writes your line into a hidden
script under one character's name and lets the model write the next character's
speech. It can see the last 256 characters and nothing before that.

## The model

| | |
|---|---|
| Type | Character-level, decoder-only transformer (GPT-style), trained from scratch |
| Parameters | 4,833,345 |
| Size | 6 layers, 4 heads per layer, width 256, context 256 characters |
| Vocabulary | 65 characters |
| Training data | Tiny Shakespeare, about 1 MB of text |
| Loss | 1.21 train, 1.49 validation (measured on the saved `model.pt`) |
| Hardware | MacBook Air M3 |

## Files

| File | What it is |
|---|---|
| `bard.py` | The game. |
| `chat.py` | Plain-text chat, and the model code that `bard.py` loads. |
| `model.pt` | The trained weights (about 26 MB). |
| `gpt5.py` | The training script that produced `model.pt`. |
| `gpt.py` to `gpt4.py` | The steps along the way: tokenizer, bigram model, single attention head, first full transformer. |
| `input.txt` | The training text. |

To train your own, install PyTorch and run `python gpt5.py`. It saves the best
checkpoint to `model.pt` every 500 steps. Swap in any `input.txt` of about 1 MB or more.

## Credits

The training text is the Tiny Shakespeare file from Andrej Karpathy's
[char-rnn](https://github.com/karpathy/char-rnn), and the model follows the design
taught in his nanoGPT lessons.
