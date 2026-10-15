"""
prepare_dataset.py
Builds the labelled dataset (data/pvp_chat_labelled.csv) from the two raw Kaggle files:
  - chatlogs.csv               (League of Legends tribunal chat logs)
  - dota2_chat_messages.csv    (DOTA 2 match chat)

Neither raw file has per-message sentiment labels, so messages are weak-labelled with a
gaming-toxicity lexicon (same idea as the paper's "manually constructed dictionary"):
  0 = neutral / positive    1 = negative attitude    2 = derogatory (slurs, sexist/racist abuse)

Usage:  python prepare_dataset.py --raw_dir <folder with the 2 CSVs> --out data/pvp_chat_labelled.csv
"""
import argparse, re
import pandas as pd

# ---------- Class 2: derogatory (slurs, racist / sexist / homophobic / ableist abuse, death wishes)
DEROG = [
    r"n+[i1!]+gg+(?:a|er|uh|ah|az|ers|as)?s?", r"nigs?", r"negro(?:es)?",
    r"f+a+g+(?:g+o+t+|o+t+)?s?", r"fagg?ots?", r"homos?", r"gay(?:s|boy|lord)?", r"lesbo?s?", r"tranny",
    r"retard(?:s|ed)?", r"ret[ae]rd\w*", r"tard(?:s)?", r"autis(?:t|tic|m)\w*", r"mongo(?:l|loid)?\w*",
    r"spastic", r"downie", r"down\s*syndrome",
    r"cunts?", r"bitch(?:es|y)?", r"b[i1]tch\w*", r"whores?", r"sluts?", r"hoes?", r"pussy", r"pussies",
    r"kys", r"kill\s*(?:ur|your)\s*sel(?:f|ves)", r"hang\s*(?:ur|your)\s*sel(?:f|ves)",
    r"(?:get|hope\s*(?:u|you)\s*get)\s*cancer", r"cancer\s*(?:kid|boy|player|ass)",
    r"rape[ds]?", r"raping", r"jews?", r"chinks?", r"gooks?", r"pakis?", r"spics?",
    r"monkeys?(?!\s*king)", r"terrorists?",
    r"go\s*back\s*to\s*(?:the\s*)?kitchen", r"(?:play|go)\s*(?:barbie|with\s*dolls)", r"like\s*a\s*girl",
    r"girls?\s*(?:cant|can't|cannot|should)",
    r"(?:ur|your|yo)\s*(?:mom|mum|mother|mama)", r"(?:mom|mum|mother)\s*(?:fucker|is\s*a)", r"motherfuck\w*",
    r"sons?\s*of\s*(?:a\s*)?bitch\w*", r"suck\s*(?:my\s*)?(?:dick|cock)", r"cock\s*sucker\w*", r"dickhead\w*",
]
# ---------- Class 1: negative attitude (insults, blame, flaming, profanity, rage)
NEG = [
    r"noo+b(?:s|y|z|ish)?", r"n00b\w*", r"nubs?", r"idiots?", r"stupid\w*", r"dumb\w*", r"morons?",
    r"trash(?:y)?", r"garbage", r"useless", r"feed(?:er|ers|ing|s)?", r"inting", r"int(?:er|ers)",
    r"report(?:ed|ing)?", r"uninstall\w*", r"loser\w*", r"pathetic", r"worst", r"brain\s*dead", r"braindead",
    r"shit\w*", r"sh1t\w*", r"fuck\w*", r"fck\w*", r"fk", r"fuk\w*", r"fking", r"wtf", r"stfu", r"gtfo",
    r"omfg", r"ffs", r"damn", r"dafuq", r"asshole\w*", r"ass", r"dicks?", r"dogs?", r"clown\w*", r"scrub\w*",
    r"(?<!my )bad(?:dest)?", r"terrible", r"awful", r"suck(?:s|ing|ed)?", r"hate", r"shut\s*up", r"die",
    r"l2p", r"learn\s*to\s*play", r"no\s*brain", r"cry(?:ing)?", r"cancer", r"toxic",
    r"bronze\s*\w*", r"low\s*elo", r"\w+\s*diff",
]
RX2 = re.compile(r"\b(?:" + "|".join(DEROG) + r")\b", re.I)
RX1 = re.compile(r"\b(?:" + "|".join(NEG) + r")\b", re.I)

def weak_label(text: str) -> int:
    t = str(text)
    if RX2.search(t): return 2
    if RX1.search(t): return 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", default="data/pvp_chat_labelled.csv")
    ap.add_argument("--dota_rows", type=int, default=3_000_000, help="DOTA file has 21M rows; read only the first N")
    ap.add_argument("--n_total", type=int, default=33_000, help="paper used ~33k messages")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    lol = pd.read_csv(f"{a.raw_dir}/chatlogs.csv", usecols=["message"]).rename(columns={"message": "text"})
    lol["game"] = "LoL"
    dota = pd.read_csv(f"{a.raw_dir}/dota2_chat_messages.csv", usecols=["text"], nrows=a.dota_rows)
    dota["game"] = "DOTA2"
    df = pd.concat([lol, dota], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()

    # keep Latin-script messages with at least one real word (drops Russian/Chinese chat & pure emoticons)
    df = df[df.text.str.fullmatch(r"[\x20-\x7E]+") & df.text.str.contains(r"[A-Za-z]{2,}")]
    # drop exact duplicates ("gg", "lol" ...) so the same message can't sit in both train and test
    df["norm"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates("norm").drop(columns="norm")

    df["label"] = df.text.map(weak_label)
    print("Pool after cleaning:", len(df), df.label.value_counts().sort_index().to_dict())

    # sample to the paper's class balance (Fig. 2: ~44% / 31% / 25%)
    share = {0: 0.44, 1: 0.31, 2: 0.25}
    parts = [df[df.label == k].sample(int(a.n_total * s), random_state=a.seed) for k, s in share.items()]
    out = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print("Saved", a.out, out.shape, out.label.value_counts().sort_index().to_dict(), out.game.value_counts().to_dict())

if __name__ == "__main__":
    main()
