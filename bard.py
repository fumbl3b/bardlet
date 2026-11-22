# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Ye Olde Bard: an old-school terminal game front end for the tiny Shakespeare GPT.

    uv run bard.py                    (no setup; uv fetches PyTorch)
    python bard.py                    (if PyTorch is already installed)

Options: --mute, --no-intro, --temperature 0.8, --model model.pt
Needs chat.py and model.pt in the same folder.
On Windows also: pip install windows-curses
"""
import argparse
import curses
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import wave

MIN_ROWS, MIN_COLS = 24, 72

# EEE = an eye, MMMMMMMMM = the mouth; both are swapped in per frame.
PORTRAIT = [
    r"         _.-------._       ",
    r"       .'           '.     ",
    r"      /               \    ",
    r"    ,|                 |,  ",
    r"   {{|   ___     ___   |}} ",
    r"   {{|   EEE     EEE   |}} ",
    r"   {{|        |        |}} ",
    r"   {{|       ._.       |}} ",
    r"    {|    ,~~~~~~~,    |}  ",
    r"     \    MMMMMMMMM    /   ",
    r"      \     \:::/     /    ",
    r"       '.    ':'    .'     ",
    r"    .-~~~'-._____.-'~~~-.  ",
    r"  {~{~{~{~{~~{_}~~}~}~}~}~}",
    r"   '~{~{~{~{~~~~~}~}~}~}~' ",
]
PORTRAIT_W = max(len(row) for row in PORTRAIT)
EYE_OPEN, EYE_SHUT = "<o>", "<->"
MOUTH_SHUT = "'-.___.-'"
MOUTH_TALK = ["'-(___)-'", "'-(   )-'", "'-(ooo)-'", "'-(___)-'"]


class Sound:
    """Square-wave blips, written as tiny WAV files and played in the background."""

    RATE = 22050

    def __init__(self, muted):
        self.player = None if muted else self._find_player()
        self.dir = None
        self.files = {}
        self.procs = []
        self.last = 0.0
        if self.player:
            self.dir = tempfile.mkdtemp(prefix="bard-sfx-")
            self._tone("tick", [(990, 0.02)])
            self._tone("boot", [(262, 0.08), (330, 0.08), (392, 0.08), (523, 0.18)])
            self._tone("send", [(660, 0.04), (880, 0.07)])
            self._tone("end", [(330, 0.05), (247, 0.10)])
            for i, freq in enumerate((196, 220, 247, 262)):
                self._tone(f"talk{i}", [(freq, 0.035)])

    @staticmethod
    def _find_player():
        if sys.platform == "win32":
            return "winsound"
        for cmd in (["afplay"], ["paplay"], ["aplay", "-q"]):
            if shutil.which(cmd[0]):
                return cmd
        return None

    def _tone(self, name, notes, volume=0.2):
        frames = bytearray()
        for freq, seconds in notes:
            n = int(self.RATE * seconds)
            for i in range(n):
                level = 1 if (i * freq / self.RATE) % 1 < 0.5 else -1
                fade = min(1.0, i / 40, (n - i) / 40)  # avoids clicks
                frames += struct.pack("<h", int(level * fade * volume * 32767))
        path = os.path.join(self.dir, name + ".wav")
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.RATE)
            f.writeframes(bytes(frames))
        self.files[name] = path

    def play(self, name, min_gap=0.0):
        if not self.player:
            return
        now = time.monotonic()
        if now - self.last < min_gap:
            return
        self.last = now
        path = self.files[name]
        try:
            if self.player == "winsound":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                self.procs = [p for p in self.procs if p.poll() is None]
                self.procs.append(subprocess.Popen(
                    self.player + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        except OSError:
            self.player = None  # give up on sound quietly

    def talk(self):
        self.play(f"talk{random.randrange(4)}", min_gap=0.07)

    def cleanup(self):
        for p in self.procs:
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                p.kill()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


class Game:
    def __init__(self, scr, args, sound):
        self.scr = scr
        self.args = args
        self.sound = sound
        self.entries = []      # [who, text] pairs; who is "you" or "bard"
        self.script = []       # the hidden play script, as token ids
        self.typed = ""
        self.talking = False
        self.mouth = 0
        self.blink_until = 0.0
        self.next_blink = time.monotonic() + 3
        self.params = 0

        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        for i, colour in enumerate((curses.COLOR_GREEN, curses.COLOR_CYAN, curses.COLOR_YELLOW,
                                    curses.COLOR_RED, curses.COLOR_MAGENTA, curses.COLOR_WHITE), 1):
            curses.init_pair(i, colour, -1)
        self.GREEN, self.CYAN, self.YELLOW, self.RED, self.MAGENTA, self.WHITE = (
            curses.color_pair(i) for i in range(1, 7))

    # ---------- drawing ----------

    def put(self, y, x, text, attr=0):
        h, w = self.scr.getmaxyx()
        if not (0 <= y < h) or x >= w or x < 0:
            return
        text = text[: w - x - (1 if y == h - 1 else 0)]  # never touch the last cell
        try:
            self.scr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def box(self, top, left, bottom, right, title, attr):
        width = right - left + 1
        label = f"[ {title} ]"
        self.put(top, left, "+--" + label + "-" * max(0, width - len(label) - 4) + "+", attr)
        for y in range(top + 1, bottom):
            self.put(y, left, "|", attr)
            self.put(y, right, "|", attr)
        self.put(bottom, left, "+" + "-" * (width - 2) + "+", attr)

    def portrait_attr(self, row, col, ch):
        if row >= 12:
            return self.WHITE | curses.A_BOLD                      # ruff collar
        if row == 5 and (9 <= col <= 11 or 17 <= col <= 19):
            return self.CYAN | curses.A_BOLD                       # eyes
        if row == 9 and 10 <= col <= 18:
            return self.RED | curses.A_BOLD                        # mouth
        if ch in "{}" or (row == 3 and ch == ","):
            return self.YELLOW                                     # hair
        if (row == 8 and 10 <= col <= 18) or (row in (10, 11) and 12 <= col <= 16):
            return self.YELLOW                                     # moustache and beard
        return self.WHITE

    def draw_portrait(self, top, left):
        now = time.monotonic()
        if not self.talking and now >= self.next_blink:
            self.blink_until = now + 0.15
            self.next_blink = now + random.uniform(2.5, 6)
        eye = EYE_SHUT if now < self.blink_until else EYE_OPEN
        mouth = MOUTH_TALK[self.mouth % len(MOUTH_TALK)] if self.talking else MOUTH_SHUT
        for r, row in enumerate(PORTRAIT):
            row = row.replace("EEE", eye).replace("MMMMMMMMM", mouth)
            for c, ch in enumerate(row):
                if ch != " ":
                    self.put(top + r, left + c, ch, self.portrait_attr(r, c, ch))
        caption = "* SPEAKING *" if self.talking else "~ THE BARD ~"
        attr = (self.MAGENTA | curses.A_BOLD) if self.talking else self.YELLOW
        self.put(top + len(PORTRAIT) + 1, left + (PORTRAIT_W - len(caption)) // 2, caption, attr)

    def dialogue_lines(self, width):
        lines = []
        for i, (who, text) in enumerate(self.entries):
            label = "THOU> " if who == "you" else "BARD> "
            attr = (self.CYAN if who == "you" else self.GREEN) | curses.A_BOLD
            body = text.strip("\n")
            if self.talking and i == len(self.entries) - 1:
                body += "_"
            first = True
            for para in (body or "...").split("\n"):
                for piece in textwrap.wrap(para, max(10, width - len(label))) or [""]:
                    lines.append(((label if first else " " * len(label)) + piece, attr))
                    first = False
            lines.append(("", 0))
        return lines

    def draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if h < MIN_ROWS or w < MIN_COLS:
            self.put(0, 0, f"Make the window at least {MIN_COLS}x{MIN_ROWS} (now {w}x{h}).", self.RED)
            self.scr.refresh()
            return

        title = "  Y E   O L D E   B A R D  "
        self.put(0, 0, " " * (w - 1), self.YELLOW | curses.A_REVERSE)
        self.put(0, (w - len(title)) // 2, title, self.YELLOW | curses.A_REVERSE | curses.A_BOLD)

        self.draw_portrait(2, 2)

        left, right, top, bottom = PORTRAIT_W + 4, w - 2, 2, h - 6
        self.box(top, left, bottom, right, "DIALOGUE", self.GREEN)
        inner_w, inner_h = right - left - 3, bottom - top - 1
        if self.entries:
            for i, (text, attr) in enumerate(self.dialogue_lines(inner_w)[-inner_h:]):
                self.put(top + 1 + i, left + 2, text, attr)
        else:
            self.put(top + 2, left + 2, "Speak, friend, and the Bard shall answer."[:inner_w], self.GREEN)
            self.put(top + 4, left + 2, "(He knoweth only Shakespeare,"[:inner_w], self.GREEN | curses.A_DIM)
            self.put(top + 5, left + 2, " and understandeth little.)"[:inner_w], self.GREEN | curses.A_DIM)

        self.box(h - 5, 1, h - 3, w - 2, "SPEAK", self.CYAN)
        if self.talking:
            self.put(h - 4, 3, "The Bard speaketh... (ESC to cut him short)", self.MAGENTA)
        else:
            cursor = "_" if int(time.monotonic() * 2) % 2 else " "
            room = w - 12
            self.put(h - 4, 3, "THOU> " + self.typed[-room:] + cursor, self.CYAN | curses.A_BOLD)

        sound = "on" if self.sound.player else "off"
        footer = (f" ENTER speak | ESC quit | {self.params:,} params"
                  f" | temp {self.args.temperature} | sound {sound}")
        self.put(h - 1, 0, footer, self.YELLOW | curses.A_DIM)
        self.scr.refresh()

    # ---------- model ----------

    def load_model(self):
        import torch
        from chat import load
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.chars = load(self.args.model, self.device)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.params = sum(p.numel() for p in self.model.parameters())

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def intro(self):
        self.scr.erase()
        row = [1]

        def line(text, pause, attr=None):
            self.put(row[0], 2, text, attr or self.GREEN)
            self.scr.refresh()
            self.sound.play("tick")
            row[0] += 1
            time.sleep(pause)

        if self.args.no_intro:
            self.load_model()
            return
        line("GLOBE SOFTWORKS BIOS v15.99", 0.35, self.GREEN | curses.A_BOLD)
        row[0] += 1
        line("MEMORY CHECK ............ 640K OK", 0.3)
        line("QUILL DRIVER ............ OK", 0.25)
        line("INKWELL ................. FULL", 0.25)
        line(f"LOADING {os.path.basename(self.args.model).upper()} ...", 0)
        self.load_model()
        line(f"  {self.params:,} PARAMETERS OK", 0.4)
        row[0] += 1
        line("SUMMONING THE BARD ...", 0.2, self.YELLOW | curses.A_BOLD)
        self.sound.play("boot")
        time.sleep(0.8)

    def speak(self):
        """Generate the Bard's reply one character at a time, animating as it goes."""
        torch, block = self.torch, self.model.block_size
        self.entries.append(["bard", ""])
        self.talking = True
        self.scr.nodelay(True)
        reply = ""
        while len(reply) < self.args.max_chars and not reply.endswith("\n\n"):
            idx = torch.tensor([self.script[-block:]], dtype=torch.long, device=self.device)
            token = self.model.next_token(idx, self.args.temperature).item()
            self.script.append(token)
            ch = self.chars[token]
            reply += ch
            self.entries[-1][1] = reply
            if ch.strip():
                self.mouth += 1
                self.sound.talk()
            self.draw()
            try:
                if self.scr.get_wch() == "\x1b":
                    break
            except curses.error:
                pass
        if not reply.endswith("\n\n"):
            self.script += self.encode("\n\n")  # close the speech in the script too
        self.talking = False
        self.sound.play("end")

    def read_line(self):
        """A tiny line editor that keeps the idle animation running. None means quit."""
        self.scr.nodelay(False)
        self.scr.timeout(80)
        self.typed = ""
        while True:
            self.draw()
            try:
                key = self.scr.get_wch()
            except curses.error:
                continue  # timeout: just redraw
            if key in ("\n", "\r", curses.KEY_ENTER):
                if self.encode(self.typed.strip()):
                    return self.typed.strip()
            elif key == "\x1b":
                return None
            elif key in ("\x7f", "\b", curses.KEY_BACKSPACE):
                self.typed = self.typed[:-1]
            elif isinstance(key, str) and key.isprintable() and len(self.typed) < 200:
                self.typed += key

    def loop(self):
        while True:
            line = self.read_line()
            if line is None:
                return
            self.typed = ""
            self.sound.play("send")
            self.entries.append(["you", line])
            self.script += self.encode(f"{self.args.you}:\n{line}\n\n{self.args.bot}:\n")
            self.speak()


def run(scr, args, sound):
    game = Game(scr, args, sound)
    game.intro()
    game.loop()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Ye Olde Bard: chat with a tiny Shakespeare GPT.")
    parser.add_argument("--model", default=os.path.join(here, "model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--you", default="ROMEO", help="hidden script name for your lines")
    parser.add_argument("--bot", default="JULIET", help="hidden script name for the Bard's lines")
    parser.add_argument("--mute", action="store_true", help="no sound effects")
    parser.add_argument("--no-intro", action="store_true", help="skip the boot screen")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Can't find {args.model}. Train a model first (python gpt5.py) or pass --model.")
    sys.path.insert(0, here)                 # so `from chat import load` works from anywhere
    os.environ.setdefault("ESCDELAY", "25")  # make the ESC key respond at once

    sound = Sound(args.mute)
    try:
        curses.wrapper(run, args, sound)
    except KeyboardInterrupt:
        pass
    finally:
        sound.cleanup()
    print("Exeunt.")


if __name__ == "__main__":
    main()
