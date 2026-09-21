"""
Corpus builder for the NLP agent.

Sources (all legally redistributable / openly licensed):
  1. Project Gutenberg  - curated catalog of ~70 public-domain books, each tried
                          against several mirrors so one dead URL doesn't kill a download.
  2. Gutendex API       - discovers *every* public-domain English book by chosen authors
                          (Dostoevsky, Conan Doyle, Mary Shelley by default).
  3. Wikipedia API      - full article text for topic lists + category/search expansion
                          (Harry Potter lore, Dostoevsky, Sherlock Holmes, Frankenstein).
  4. Harry Potter Wiki  - optional (--fandom), CC BY-SA fan encyclopedia via MediaWiki API.
  5. Hugging Face       - optional (--hf), large generic corpora streamed with `datasets`.

Everything is cached on disk, downloaded in parallel, retried with back-off, and described
in data/raw/manifest.json (source URL, license, size) so you always know where text came from.

Backward compatible with the previous module: DATASET_SOURCES, DOSTOEVSKY_NOTES_TEXT,
clean_gutenberg_header_footer, ensure_dostoevsky_notes, write_harry_potter_lore,
write_dostoevsky_sample and prepare_all_datasets keep their names.

CLI:
    python datasets.py                      # everything except Fandom / HF
    python datasets.py --fandom --hf tinystories
    python datasets.py --categories dostoevsky sherlock_holmes --no-wikipedia
    python datasets.py --offline            # only the embedded fallback corpora
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Wikimedia asks for a descriptive User-Agent with contact info.
USER_AGENT = "NLP-Agent/2.0 (text corpus builder; contact: open-source-nlp@example.com)"

# --------------------------------------------------------------------------------------
# Catalog of Project Gutenberg books (public domain in the US)
# NOTE: IDs are stored by category. Every download records the book's real "Title:" and
# "Author:" header in manifest.json, so a wrong ID is easy to spot and fix.
# --------------------------------------------------------------------------------------
GUTENBERG_CATALOG: Dict[str, Dict[str, int]] = {
    "dostoevsky": {
        "dostoevsky_notes": 600,
        "dostoevsky_crime": 2554,
        "dostoevsky_gambler": 2197,
        "dostoevsky_karamazov": 28054,
        "dostoevsky_idiot": 2638,
        "dostoevsky_demons": 8117,
        "dostoevsky_poor_folk": 2302,
        "dostoevsky_house_of_the_dead": 37536,
    },
    "sherlock_holmes": {
        "sherlock_holmes": 1661,  # The Adventures of Sherlock Holmes
        "sherlock_study_in_scarlet": 244,
        "sherlock_sign_of_four": 2097,
        "sherlock_memoirs": 834,
        "sherlock_return": 108,
        "sherlock_hound": 2852,
        "sherlock_valley_of_fear": 3289,
        "sherlock_last_bow": 2350,
        "sherlock_case_book": 69700,
    },
    "gothic": {
        "frankenstein": 84,
        "dracula": 345,
        "jekyll_hyde": 43,
        "dorian_gray": 174,
        "wuthering_heights": 768,
        "jane_eyre": 1260,
        "turn_of_the_screw": 209,
        "poe_works_vol1": 2147,
    },
    "russian_lit": {
        "war_and_peace": 2600,
        "anna_karenina": 1399,
        "dead_souls": 1081,
        "fathers_and_sons": 30723,
    },
    "philosophy": {
        "zarathustra": 1998,
        "beyond_good_and_evil": 4363,
        "republic": 1497,
        "meditations": 2680,
        "art_of_war": 132,
        "the_prince": 1232,
        "leviathan": 3207,
        "walden": 205,
        "metamorphosis": 5200,
    },
    "classics": {
        "pride_and_prejudice": 1342,
        "emma": 158,
        "great_expectations": 1400,
        "tale_of_two_cities": 98,
        "moby_dick": 2701,
        "alice_wonderland": 11,
        "ulysses": 4300,
        "les_miserables": 135,
        "monte_cristo": 1184,
        "three_musketeers": 1257,
        "don_quixote": 996,
        "huckleberry_finn": 76,
        "tom_sawyer": 74,
        "treasure_island": 120,
        "middlemarch": 145,
        "time_machine": 35,
        "war_of_the_worlds": 36,
    },
    "epics_drama": {
        "shakespeare_complete": 100,
        "iliad": 6130,
        "odyssey": 1727,
        "divine_comedy": 8800,
        "paradise_lost": 26,
    },
}

# Authors discovered dynamically through the Gutendex API: slug -> (search query, name token)
GUTENDEX_AUTHORS: Dict[str, Tuple[str, str]] = {
    "dostoevsky": ("dostoyevsky", "dostoyevsky"),  # Gutenberg spells it "Dostoyevsky"
    "doyle": ("doyle", "doyle"),
    "shelley": ("mary shelley", "shelley"),
}

# Wikipedia / MediaWiki topic definitions
WIKI_TOPICS: Dict[str, Dict[str, Any]] = {
    "harry_potter": {
        "titles": [
            "Harry Potter", "Harry Potter (character)", "Hogwarts", "Lord Voldemort",
            "Albus Dumbledore", "Severus Snape", "Sirius Black (character)", "Hermione Granger",
            "Ron Weasley", "Draco Malfoy", "Rubeus Hagrid", "Neville Longbottom",
            "Fictional universe of Harry Potter", "Magic in Harry Potter",
            "Places in Harry Potter", "Magical objects in Harry Potter",
            "Harry Potter and the Philosopher's Stone", "Harry Potter and the Chamber of Secrets",
            "Harry Potter and the Prisoner of Azkaban", "Harry Potter and the Goblet of Fire",
            "Harry Potter and the Order of the Phoenix", "Harry Potter and the Half-Blood Prince",
            "Harry Potter and the Deathly Hallows", "Harry Potter influences and analogues",
            "Themes in Harry Potter", "Harry Potter fandom", "Quidditch", "Horcrux",
        ],
        "categories": ["Category:Harry Potter characters", "Category:Harry Potter"],
        "search": ["Harry Potter wizarding world", "Hogwarts house"],
    },
    "dostoevsky": {
        "titles": [
            "Fyodor Dostoevsky", "Notes from Underground", "Crime and Punishment",
            "The Brothers Karamazov", "The Idiot", "Demons (Dostoevsky novel)",
            "The Gambler (novel)", "Poor Folk", "The House of the Dead (novel)",
            "The Double: A Petersburg Poem", "Existentialism", "Russian literature",
            "Nihilism", "Underground Man",
        ],
        "categories": ["Category:Novels by Fyodor Dostoevsky"],
        "search": [],
    },
    "sherlock_holmes": {
        "titles": [
            "Sherlock Holmes", "Arthur Conan Doyle", "The Adventures of Sherlock Holmes",
            "The Hound of the Baskervilles", "A Study in Scarlet", "The Sign of the Four",
            "The Memoirs of Sherlock Holmes", "The Return of Sherlock Holmes",
            "Professor Moriarty", "Dr. Watson", "Detective fiction",
        ],
        "categories": ["Category:Sherlock Holmes short stories", "Category:Sherlock Holmes novels"],
        "search": [],
    },
    "frankenstein": {
        "titles": [
            "Frankenstein", "Mary Shelley", "Victor Frankenstein", "Frankenstein's monster",
            "Gothic fiction", "Romanticism", "Science fiction",
        ],
        "categories": [],
        "search": [],
    },
}

# Optional large generic corpora (need `pip install datasets`)
HF_DATASETS: Dict[str, Dict[str, Optional[str]]] = {
    "tinystories": {"path": "roneneldan/TinyStories", "config": None, "split": "train", "field": "text",
                    "license": "CDLA-Sharing-1.0"},
    "wikitext103": {"path": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train",
                    "field": "text", "license": "CC BY-SA 3.0"},
}

# Mirrors used to try Gutenberg books. {id} = book id, {path} = digit-split directory.
_GUTENBERG_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://gutenberg.pglaf.org/{path}/{id}/{id}-0.txt",
    "https://aleph.gutenberg.org/{path}/{id}/{id}-0.txt",
]


def gutenberg_urls(book_id: int) -> List[str]:
    path = "/".join(str(book_id)[:-1]) or "0"
    return [t.format(id=book_id, path=path) for t in _GUTENBERG_URL_TEMPLATES]


# Kept for backward compatibility with the previous version of this module.
DATASET_SOURCES: Dict[str, str] = {
    "dostoevsky_notes": gutenberg_urls(600)[0],
    "dostoevsky_crime": gutenberg_urls(2554)[0],
    "dostoevsky_gambler": gutenberg_urls(2197)[0],
    "sherlock_holmes": gutenberg_urls(1661)[0],
    "frankenstein": gutenberg_urls(84)[0],
}

# --------------------------------------------------------------------------------------
# Embedded fallback corpora (used only when the network is unavailable)
# --------------------------------------------------------------------------------------
DOSTOEVSKY_NOTES_TEXT = """
I am a sick man.... I am a spiteful man. I am an unattractive man. I believe my liver is diseased.
However, I know nothing at all about my disease, and do not know for certain what ails me. I don't consult a doctor for it,
and never have, though I have a respect for medicine and doctors. Besides, I am extremely superstitious, sufficiently so to respect medicine, anyway.
No, I refuse to consult a doctor from spite. That you probably will not understand. Well, I understand it, though.
Of course, I can't explain who it is precisely that I am mortifying in this case by my spite: I am perfectly well aware that I cannot hurt the doctors by not consulting them;
I know better than anyone that by all this I am only hurting myself and no one else.
Still, if I don't consult a doctor it is from spite. My liver is bad, well then let it get even worse!

I have been living like this for a long time twenty years. Now I am forty. I used to be in the civil service, but no longer am.
I was a spiteful official. I was rude and took pleasure in being so. I did not take bribes, you see, so I was bound to find a recompense in that, at least.
When petitioners used to come for information to the table at which I sat, I used to grind my teeth at them, and felt intense enjoyment when I succeeded in making anyone unhappy.
I almost always succeeded. For the most part they were all timid people of course, they were petitioners.
But of the uppish persons there was one officer in particular I could not endure. He simply would not be humble, and clanked his sword in a disgusting way.
I carried on a feud with him for eighteen months over that sword. At last I got the better of him. He left off clanking it. That happened in my youth, though.

An intelligent man cannot become anything seriously, and it is only the fool who becomes anything.
Yes, a man in the nineteenth century must and morally ought to be pre-eminently a characterless creature;
a man of character, an active man is pre-eminently a limited creature.
That is my conviction of forty years. I am forty years old now, and you know forty years is a whole lifetime;
you know it is extreme old age. To live longer than forty years is bad manners, is vulgar, immoral.
Who does live beyond forty? Answer that, sincerely and honestly! I will tell you who do: fools and worthless fellows.
I tell all old men that to their faces, all these venerable old men, all these silver-haired and reverend seniors!
I tell the whole world so to its face! I have a right to say so, for I shall go on living to sixty myself! To seventy! To eighty! Hold on, let me catch my breath.

It is better to do nothing! Better conscious inertia! And so hurrah for underground!
Though I have said that I envy the normal man to the last drop of my bile, yet I should not care to be in his place such as he is now.
Though why am I lying! I am lying because I know myself that it is not underground that is better, but something different, quite different,
for which I am thirsting, but which I cannot find! Damn underground!

Reason only satisfies the reasoning side of man's nature, but will is a manifestation of the whole of life,
that is, of the whole human life including reason and all the impulses. And although our life, in this manifestation of it,
is often worthless, yet it is life and not simply extracting square roots.
Here I, for instance, quite naturally want to live, in order to satisfy all my capacities for life, and not simply my reasoning capacity.
What does reason know? Reason only knows what it has succeeded in learning, and human nature acts as a whole,
with everything that is in it, consciously or unconsciously, and, even if it goes wrong, it lives.

Gentlemen, there are cases when man purposefully wishes for himself what is harmful and stupid, simply to have the right of wishing for himself even what is absurd,
and not being bound to desire only what is sensible.
Man likes to make roads and to create, that is undeniable. But why has he also such a passionate love for destruction and chaos?
Shall I tell you why? Because he is instinctively afraid of attaining his goal and completing the edifice he is constructing.
How do you know that he does not only love that edifice from a distance, and is by no means in love with it at close quarters?
Perhaps the only goal on earth to which mankind is striving consists in this uninterrupted process of attaining,
or in other words, in life itself, and not in the thing to be attained.

Two times two makes four seems to me simply a piece of insolence. Two times two makes four is a pert fellow who stands with arms akimbo barring your path and spitting.
I admit that two times two makes four is an excellent thing, but if we are to give everything its due, two times two makes five is sometimes a very charming thing too.
And why are you so firmly, so triumphantly, convinced that only the normal and the positive in other words, only what is conducive to welfare is for the advantage of man?
Is not reason in error as regards advantage? Does not man, perhaps, love something besides well-being?
Perhaps he is just as fond of suffering? Perhaps suffering is just as great an advantage to him as well-being?
Man is sometimes extraordinarily, passionately fond of suffering, and that is a fact.
Suffering is the sole origin of consciousness. Though I did lay down at the beginning that consciousness is the greatest misfortune for man, yet I know man prizes it and would not give it up for any satisfaction.

I want now to tell you, gentlemen, whether you care to hear it or not, why I could not even become an insect.
I tell you solemnly, that I have many times tried to become an insect. But I was not even worthy of that.
I swear, gentlemen, that to be too conscious is an illness a real thorough-going illness.
For man's everyday needs it would have been quite enough to have the ordinary human consciousness, that is, half or a quarter of the amount which falls to the lot of a cultivated man of our unhappy nineteenth century, especially one who has the fatal ill-luck to inhabit Petersburg, the most theoretical and intentional town on the whole terrestrial globe.
It would have been quite enough, for instance, to have the consciousness of an average direct man and man of action.
I bet you think that I am writing all this from vanity, to be funny at the expense of men of action, and, furthermore, that out of sheer bad taste I am clanking my sword like my officer.
"""

HARRY_POTTER_LORE_TEXT = """
Since the Harry Potter series got darker further into the storyline, deaths play an important role in the development of the characters and the wizarding world.
As we approached closer to the fight, several characters lost their lives for Harry and for the bigger goal which was to kill Voldemort.
Yet, not all deaths had the same weight and meaning in the story; some deaths had stronger impacts on the readers and the characters.

Before we get to the theme of death in Harry Potter, one must ask the question: What is Death? Is Death always the end? I think of death as not only an ending but also a beginning.
Death is a beginning to a new adventure, and in the series, some deaths signified a new journey or a new time phase in the wizarding world.
After every death of a major character, the characters underwent changes and character development, and the changes were clearest for Harry Potter, the Chosen Boy.
In order to analyze the role of deaths in Harry Potter, we would look at the deaths that might have the most impacts on Harry as well as the readers.

Since the beginning, death was an important part of the storyline with the death of Lily and James Potter to protect their son against Voldemort.
The tragic death of Harry’s parents was a beginning to a temporarily peaceful stage for the wizarding world with the disappearance of Voldemort.
They sacrificed their lives for their son to live, but their death also made Harry become the seventh Horcrux. As an unexpected Horcrux, Harry was set out to be the Chosen One to defeat Voldemort.
Harry’s life also took a different turn as he was raised in the Muggle world by his aunt’s family, constantly being bullied and mistreated.
The death was not just the end but also a new beginning in this case. Moreover, Harry’s parents’ death was a symbol of love and the power of love.

As Harry got to Hogwarts, he encountered several deaths in the long-lasting fight against Voldemort. However, the death of Cedric was a turning point for Harry to learn what he was fighting against
and what was waiting for him later on. The death of Cedric was not only Voldemort’s first announcement of his return but also a point of guilt and haunt for Harry.
Cedric was killed as a spare to Harry; Voldemort viewed him as nothing but an inconvenience on his way to get Harry. This left a sense of guilt and trauma for Harry,
which led him to be more serious and determined in the fight against Voldemort. His growth and development after the death were seen through the formation of Dumbledore’s Army to help
teach fellow students self-defense in battle.

Yet, in the series, I believe the death of Sirius Black and the death of Dumbledore had the most impacts on Harry. They all acted as mentors to Harry on his journey,
so their deaths gave him more urges to revenge and to fight against Voldemort to protect his loved ones. In terms of Sirius Black, he was a best friend of Lily and James,
so Sirius was like a father figure to Harry. However, when Harry thought he finally had a family, Sirius died because of Voldemort. The death was traumatic and extremely painful to Harry
and left him with a life-long pain. For Dumbledore, he was always a mentor to Harry since his first year at Hogwarts. Dumbledore’s death by Snape left Harry full of resentment for the Potion Master and a strong urge to seek revenge. The death of the two mentors in his life gave Harry a real sense of his responsibility and what was in store for him for the last fights. The deaths play an important role to not only save Harry’s life but also to motivate him in his fight for the safety of the wizarding world.

Although all aforementioned deaths were sad and left the readers and Harry emotional, some deaths did not leave us with the same feeling but rather a sense of relief.
The death of Voldemort was the most obvious example of this. Voldemort’s death was an end to the dark days of a long-lasting and fatal fight. His defeat was also a beginning
to a new era when the wizarding world could live without constant fear of death. Moreover, for Harry, this put an end to his need for revenge and his life-long responsibility as a hero.
Harry could finally live a normal life. The death of Voldemort was not a pain but a relief and a signal for the brighter days for the wizarding world.

The Hogwarts School of Witchcraft and Wizardry stood high atop the misty mountains of Scotland, protected by ancient enchantments and hidden from Muggle eyes.
Four founders established the great castle: Godric Gryffindor, Salazar Slytherin, Rowena Ravenclaw, and Helga Hufflepuff.
Each house valued distinct qualities in its students. Gryffindor cherished courage, chivalry, and determination in the face of peril.
Ravenclaw celebrated wisdom, intellect, curiosity, and creative brilliance.
Hufflepuff was dedicated to loyalty, hard work, patience, and fair play for all witches and wizards.
Slytherin favored ambition, cunning, resourcefulness, and heritage.

The Triwizard Tournament brought students together from Durmstrang Institute and Beauxbatons Academy to compete in three harrowing magical tasks.
The Goblet of Fire, an impartial judge bound by powerful magical contract, selected champions based on bravery and skill.
The Dark Arts represent spells and enchantments intended to inflict harm, control, or conquer other magical beings.
The Three Unforgivable Curses—the Imperius Curse, the Cruciatus Curse, and the Killing Curse Avada Kedavra—result in immediate imprisonment in Azkaban.
A Patronus is a positive energy projection produced by recalling one's happiest memory, acting as a shield against Soul-dementors.
The Phoenix Fawkes represented rebirth and healing through tears of miraculous restoration.
The Elder Wand, the Resurrection Stone, and the Cloak of Invisibility together form the legendary Deathly Hallows.
Whoever conquers death with humility and selfless love becomes the true master of the Hallows.
"""

# --------------------------------------------------------------------------------------
# HTTP + bookkeeping helpers
# --------------------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})

_MANIFEST: List[Dict[str, Any]] = []
_MANIFEST_LOCK = threading.Lock()


def http_get(url: str, *, params: Optional[dict] = None, timeout: int = 30,
             retries: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    """GET with retries and exponential back-off. Returns None on permanent failure."""
    for attempt in range(retries):
        resp = None
        try:
            resp = _session.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            pass
        if resp is not None:
            if resp.status_code == 200:
                return resp
            if resp.status_code in (400, 401, 403, 404, 410):
                return None  # retrying will not help
        if attempt < retries - 1:
            time.sleep(backoff ** (attempt + 1))
    return None


def _record(**entry: Any) -> None:
    with _MANIFEST_LOCK:
        _MANIFEST.append(entry)


def _slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "untitled"


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def combine_files(paths: Iterable[Path], dest: Path,
                  separator: str = "\n\n" + "=" * 80 + "\n\n") -> Optional[Path]:
    """Concatenate text files into one training-ready corpus file."""
    parts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(paths) if p.exists()]
    if not parts:
        return None
    _save_text(dest, separator.join(parts))
    return dest


# --------------------------------------------------------------------------------------
# Project Gutenberg
# --------------------------------------------------------------------------------------
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*?\*\*\*", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*?\*\*\*", re.IGNORECASE)


def clean_gutenberg_header_footer(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[:end.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _gutenberg_meta(raw: str) -> Dict[str, str]:
    head = raw[:6000].replace("\r", "")
    title = re.search(r"^Title:\s*(.+)$", head, re.MULTILINE)
    author = re.search(r"^Author:\s*(.+)$", head, re.MULTILINE)
    return {"title": title.group(1).strip() if title else "", "author": author.group(1).strip() if author else ""}


def _looks_like_html(text: str) -> bool:
    return text.lstrip()[:200].lower().startswith(("<!doctype", "<html"))


def download_gutenberg_book(book_id: int, dest: Path, *, name: Optional[str] = None,
                            extra_urls: Iterable[str] = (), min_chars: int = 5000,
                            timeout: int = 30) -> Optional[Path]:
    """Try the direct URL(s) first, then every Gutenberg mirror. Cached on disk."""
    name = name or dest.stem
    if dest.exists() and dest.stat().st_size >= min_chars * 3:  # a real book, not a stub/fallback
        return dest

    for url in [*extra_urls, *gutenberg_urls(book_id)]:
        resp = http_get(url, timeout=timeout)
        if resp is None:
            continue
        raw = resp.content.decode("utf-8-sig", errors="replace")
        if len(raw) < min_chars or _looks_like_html(raw):
            continue
        cleaned = clean_gutenberg_header_footer(raw)
        if len(cleaned) < min_chars:
            continue
        meta = _gutenberg_meta(raw)
        _save_text(dest, cleaned)
        _record(name=name, source="Project Gutenberg", gutenberg_id=book_id, url=url,
                title=meta["title"], author=meta["author"], license="Public domain (US)",
                chars=len(cleaned), path=str(dest))
        print(f"[Gutenberg] {name}: {meta['title'] or book_id} ({len(cleaned):,} chars)")
        return dest
    print(f"[Gutenberg] FAILED {name} (id={book_id}) - all mirrors unavailable")
    return None


def gutendex_find_books(query: str, name_token: str, *, max_books: int = 40,
                        languages: str = "en") -> List[Dict[str, Any]]:
    """Ask the Gutendex API (JSON catalog of Project Gutenberg) for public-domain books."""
    books: List[Dict[str, Any]] = []
    url: Optional[str] = "https://gutendex.com/books/"
    params: Optional[dict] = {"search": query, "languages": languages, "copyright": "false"}
    while url and len(books) < max_books:
        resp = http_get(url, params=params)
        if resp is None:
            break
        try:
            data = resp.json()
        except ValueError:
            break
        for book in data.get("results", []):
            if not any(name_token in a.get("name", "").lower() for a in book.get("authors", [])):
                continue
            text_url = next((u for mime, u in book.get("formats", {}).items()
                             if mime.startswith("text/plain") and not u.endswith(".zip")), None)
            books.append({"id": book["id"], "title": book.get("title", ""), "url": text_url})
            if len(books) >= max_books:
                break
        url, params = data.get("next"), None
    return books


def download_gutenberg_categories(categories: Iterable[str], out_dir: Path,
                                  workers: int = 6) -> Dict[str, List[Path]]:
    book_dir = out_dir / "gutenberg"
    results: Dict[str, List[Path]] = {}
    jobs = []
    for cat in categories:
        if cat not in GUTENBERG_CATALOG:
            print(f"[Gutenberg] unknown category '{cat}' (available: {', '.join(GUTENBERG_CATALOG)})")
            continue
        results[cat] = []
        for slug, book_id in GUTENBERG_CATALOG[cat].items():
            jobs.append((cat, slug, book_id))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_gutenberg_book, bid, book_dir / f"{slug}.txt", name=slug): cat
                   for cat, slug, bid in jobs}
        for fut in as_completed(futures):
            path = fut.result()
            if path:
                results[futures[fut]].append(path)
    return results


def download_gutendex_authors(authors: Iterable[str], out_dir: Path, *, max_books: int = 40,
                              workers: int = 6) -> Dict[str, List[Path]]:
    """Discover + download every public-domain English book by the given authors."""
    known_ids = {bid for cat in GUTENBERG_CATALOG.values() for bid in cat.values()}
    book_dir = out_dir / "gutenberg" / "gutendex"
    results: Dict[str, List[Path]] = {}
    for author in authors:
        if author not in GUTENDEX_AUTHORS:
            print(f"[Gutendex] unknown author '{author}' (available: {', '.join(GUTENDEX_AUTHORS)})")
            continue
        query, token = GUTENDEX_AUTHORS[author]
        books = [b for b in gutendex_find_books(query, token, max_books=max_books) if b["id"] not in known_ids]
        print(f"[Gutendex] {author}: {len(books)} additional books found")
        paths: List[Path] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for b in books:
                dest = book_dir / f"{author}_{b['id']}_{_slug(b['title'], 40)}.txt"
                extra = [b["url"]] if b["url"] else []
                futures.append(pool.submit(download_gutenberg_book, b["id"], dest,
                                           name=f"{author}_{b['id']}", extra_urls=extra))
            paths = [p for p in (f.result() for f in futures) if p]
        results[f"{author}_gutendex"] = paths
    return results


# --------------------------------------------------------------------------------------
# MediaWiki (Wikipedia + Fandom)
# --------------------------------------------------------------------------------------
class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "table", "aside", "figure", "sup", "nav", "noscript"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK and not self._skip:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html_text: str) -> str:
    parser = _HTMLText()
    parser.feed(html_text)
    return "".join(parser.parts)


_TAIL_SECTIONS = {"references", "external links", "further reading", "see also", "notes",
                  "bibliography", "footnotes", "sources", "citations", "gallery"}


def _tidy_wiki_text(text: str) -> str:
    """Drop trailing reference-style sections, flatten '== Heading ==' markup."""
    lines: List[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^\s*=+\s*(.*?)\s*=+\s*$", line)
        if m:
            if m.group(1).strip().lower() in _TAIL_SECTIONS:
                break
            line = m.group(1)
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class MediaWiki:
    def __init__(self, api_url: str, name: str, license_name: str, delay: float = 0.25) -> None:
        self.api_url, self.name, self.license_name, self.delay = api_url, name, license_name, delay

    def _get(self, params: dict) -> Optional[dict]:
        time.sleep(self.delay)  # be polite to the servers
        resp = http_get(self.api_url, params={**params, "format": "json"})
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def page_text(self, title: str) -> Optional[Tuple[str, str]]:
        data = self._get({"action": "query", "prop": "extracts", "explaintext": 1,
                          "exsectionformat": "wiki", "redirects": 1, "titles": title})
        if data:
            for page in data.get("query", {}).get("pages", {}).values():
                if "missing" in page:
                    return None
                if page.get("extract"):
                    return page.get("title", title), _tidy_wiki_text(page["extract"])
        # Fallback for wikis without the TextExtracts extension: parse rendered HTML
        data = self._get({"action": "parse", "prop": "text", "redirects": 1, "page": title,
                          "disableeditsection": 1})
        if data and "parse" in data:
            html_text = data["parse"].get("text", {}).get("*", "")
            return data["parse"].get("title", title), _tidy_wiki_text(html_to_text(html_text))
        return None

    def category_members(self, category: str, limit: int = 200) -> List[str]:
        titles: List[str] = []
        params: Dict[str, Any] = {"action": "query", "list": "categorymembers", "cmtitle": category,
                                  "cmlimit": "max", "cmnamespace": 0}
        while len(titles) < limit:
            data = self._get(params)
            if not data:
                break
            titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
            if "continue" not in data:
                break
            params = {**params, **data["continue"]}
        return titles[:limit]

    def search(self, query: str, limit: int = 50) -> List[str]:
        data = self._get({"action": "query", "list": "search", "srsearch": query,
                          "srlimit": min(limit, 50), "srnamespace": 0})
        return [r["title"] for r in data.get("query", {}).get("search", [])] if data else []

    def all_pages(self, limit: int = 300, min_size: int = 3000) -> List[str]:
        """Substantial articles only (apminsize filters out stubs)."""
        titles: List[str] = []
        params: Dict[str, Any] = {"action": "query", "list": "allpages", "apnamespace": 0,
                                  "aplimit": "max", "apfilterredir": "nonredirects", "apminsize": min_size}
        while len(titles) < limit:
            data = self._get(params)
            if not data:
                break
            titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
            if "continue" not in data:
                break
            params = {**params, **data["continue"]}
        return titles[:limit]


WIKIPEDIA = MediaWiki("https://en.wikipedia.org/w/api.php", "Wikipedia", "CC BY-SA 4.0")
HP_FANDOM = MediaWiki("https://harrypotter.fandom.com/api.php", "Harry Potter Wiki (Fandom)", "CC BY-SA 3.0")


def collect_wiki_pages(topic: str, titles: List[str], wiki: MediaWiki, out_dir: Path, *,
                       min_chars: int = 500, workers: int = 4) -> List[Path]:
    """Download the given article titles in parallel; one text file per article."""
    seen, unique = set(), []
    for t in titles:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    def fetch(title: str) -> Optional[Path]:
        dest = out_dir / f"{_slug(title)}.txt"
        if dest.exists() and dest.stat().st_size >= min_chars:
            return dest
        result = wiki.page_text(title)
        if not result:
            return None
        real_title, text = result
        if len(text) < min_chars:
            return None
        _save_text(dest, f"# {real_title}\n\n{text}\n")
        _record(name=f"{wiki.name}:{real_title}", source=wiki.name, topic=topic,
                url=f"{wiki.api_url.rsplit('/', 1)[0]}/wiki/{real_title.replace(' ', '_')}",
                license=wiki.license_name, chars=len(text), path=str(dest))
        return dest

    paths: List[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in pool.map(fetch, unique):
            if path:
                paths.append(path)
    print(f"[{wiki.name}] {topic}: {len(paths)}/{len(unique)} articles")
    return paths


def collect_wikipedia_topic(topic: str, out_dir: Path, *, category_limit: int = 150,
                            workers: int = 4) -> List[Path]:
    spec = WIKI_TOPICS[topic]
    titles = list(spec.get("titles", []))
    for cat in spec.get("categories", []):
        titles += WIKIPEDIA.category_members(cat, limit=category_limit)
    for q in spec.get("search", []):
        titles += WIKIPEDIA.search(q, limit=50)
    return collect_wiki_pages(topic, titles, WIKIPEDIA, out_dir / "wikipedia" / topic, workers=workers)


def collect_fandom_harry_potter(out_dir: Path, max_pages: int = 300, workers: int = 4) -> List[Path]:
    titles = HP_FANDOM.all_pages(limit=max_pages)
    return collect_wiki_pages("harry_potter", titles, HP_FANDOM,
                              out_dir / "fandom" / "harry_potter", workers=workers)


# --------------------------------------------------------------------------------------
# Optional Hugging Face corpora
# --------------------------------------------------------------------------------------
def download_hf_dataset(key: str, out_dir: Path, max_rows: int = 50_000) -> Optional[Path]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[HF] `pip install datasets` to enable Hugging Face corpora")
        return None
    spec = HF_DATASETS[key]
    dest = out_dir / "hf" / f"{key}.txt"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(spec["path"], spec["config"], split=spec["split"], streaming=True)
        chars = 0
        with open(dest, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                if i >= max_rows:
                    break
                text = (row.get(spec["field"]) or "").strip()
                if text:
                    f.write(text + "\n\n")
                    chars += len(text)
    except Exception as exc:  # network / schema problems should not stop the pipeline
        print(f"[HF] {key} failed: {exc}")
        return None
    _record(name=f"hf:{key}", source="Hugging Face", url=f"https://huggingface.co/datasets/{spec['path']}",
            license=spec["license"], chars=chars, path=str(dest))
    print(f"[HF] {key}: {chars:,} chars")
    return dest


# --------------------------------------------------------------------------------------
# Original public functions (same names / behaviour, now with mirrors + fallbacks)
# --------------------------------------------------------------------------------------
def ensure_dostoevsky_notes(output_dir: Optional[Path] = None) -> Path:
    target_dir = output_dir or DATA_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "dostoevsky_notes.txt"

    # 50k chars = the real book. (The embedded fallback is ~7k, so an earlier offline run
    # no longer blocks a later online run from fetching the full text.)
    if target_file.exists() and len(target_file.read_text(encoding="utf-8", errors="ignore")) > 50_000:
        return target_file

    got = download_gutenberg_book(600, target_file, name="dostoevsky_notes", timeout=20)
    if got:
        return got

    content = DOSTOEVSKY_NOTES_TEXT.strip()
    _save_text(target_file, content)
    print(f"[Dataset] Generated local corpus for dostoevsky_notes ({len(content):,} chars) to {target_file}")
    return target_file


def write_harry_potter_lore(output_dir: Optional[Path] = None,
                            extra_files: Optional[Iterable[Path]] = None) -> Path:
    """Embedded lore essay + (optionally) any downloaded Wikipedia/Fandom articles appended."""
    target_dir = output_dir or DATA_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "harry_potter_lore.txt"

    parts = [HARRY_POTTER_LORE_TEXT.strip()]
    for path in sorted(extra_files or []):
        parts.append(path.read_text(encoding="utf-8", errors="replace").strip())
    corpus = "\n\n".join(parts)
    _save_text(target_file, corpus)
    print(f"[Dataset] Saved Harry Potter wizarding corpus ({len(corpus):,} chars) to {target_file}")
    return target_file


def write_dostoevsky_sample(output_dir: Optional[Path] = None) -> Path:
    target_dir = output_dir or DATA_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "dostoevsky_core.txt"
    _save_text(target_file, DOSTOEVSKY_NOTES_TEXT.strip())
    return target_file


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def write_manifest(out_dir: Path) -> Path:
    for entry in _MANIFEST:  # store paths relative to the data folder
        try:
            entry["path"] = str(Path(entry["path"]).relative_to(out_dir))
        except ValueError:
            pass
    total = sum(e.get("chars", 0) for e in _MANIFEST)
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total_chars": total,
               "files": sorted(_MANIFEST, key=lambda e: e["path"])}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def prepare_all_datasets(download_online: bool = True, *,
                         categories: Optional[List[str]] = None,
                         gutendex_authors: Optional[List[str]] = None,
                         gutendex_max_books: int = 40,
                         wikipedia: bool = True,
                         wiki_topics: Optional[List[str]] = None,
                         fandom: bool = False,
                         fandom_max_pages: int = 300,
                         hf: Optional[List[str]] = None,
                         hf_max_rows: int = 50_000,
                         workers: int = 6,
                         output_dir: Optional[Path] = None) -> Dict[str, Path]:
    out_dir = output_dir or DATA_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _MANIFEST.clear()
    datasets: Dict[str, Path] = {}
    hp_extra: List[Path] = []

    if download_online:
        # 1) curated Gutenberg catalog
        cats = categories if categories is not None else list(GUTENBERG_CATALOG)
        for cat, paths in download_gutenberg_categories(cats, out_dir, workers).items():
            combined = combine_files(paths, out_dir / f"corpus_gutenberg_{cat}.txt")
            if combined:
                datasets[f"gutenberg_{cat}"] = combined

        # 2) every extra book by chosen authors (Gutendex)
        authors = gutendex_authors if gutendex_authors is not None else list(GUTENDEX_AUTHORS)
        if authors:
            for key, paths in download_gutendex_authors(authors, out_dir, max_books=gutendex_max_books,
                                                        workers=workers).items():
                combined = combine_files(paths, out_dir / f"corpus_gutenberg_{key}.txt")
                if combined:
                    datasets[f"gutenberg_{key}"] = combined

        # 3) Wikipedia topics
        if wikipedia:
            for topic in (wiki_topics if wiki_topics is not None else list(WIKI_TOPICS)):
                if topic not in WIKI_TOPICS:
                    print(f"[Wikipedia] unknown topic '{topic}' (available: {', '.join(WIKI_TOPICS)})")
                    continue
                paths = collect_wikipedia_topic(topic, out_dir, workers=max(2, workers // 2))
                if topic == "harry_potter":
                    hp_extra += paths
                combined = combine_files(paths, out_dir / f"corpus_wikipedia_{topic}.txt")
                if combined:
                    datasets[f"wikipedia_{topic}"] = combined

        # 4) Harry Potter Wiki on Fandom (opt-in)
        if fandom:
            paths = collect_fandom_harry_potter(out_dir, max_pages=fandom_max_pages, workers=max(2, workers // 2))
            hp_extra += paths
            combined = combine_files(paths, out_dir / "corpus_fandom_harry_potter.txt")
            if combined:
                datasets["fandom_harry_potter"] = combined

        # 5) Hugging Face corpora (opt-in)
        for key in hf or []:
            if key not in HF_DATASETS:
                print(f"[HF] unknown dataset '{key}' (available: {', '.join(HF_DATASETS)})")
                continue
            path = download_hf_dataset(key, out_dir, max_rows=hf_max_rows)
            if path:
                datasets[f"hf_{key}"] = path

    # Original outputs (always produced; online data is folded in when available)
    datasets["dostoevsky_notes"] = ensure_dostoevsky_notes(out_dir) if download_online else _offline_notes(out_dir)
    datasets["harry_potter_lore"] = write_harry_potter_lore(out_dir, extra_files=hp_extra)
    datasets["dostoevsky_core"] = write_dostoevsky_sample(out_dir)

    manifest = write_manifest(out_dir)
    total_mb = sum(p.stat().st_size for p in datasets.values() if p.exists()) / 1e6
    print(f"[Dataset] Done: {len(datasets)} corpora, {total_mb:,.1f} MB of combined text. Manifest: {manifest}")
    return datasets


def _offline_notes(out_dir: Path) -> Path:
    target = out_dir / "dostoevsky_notes.txt"
    if not target.exists():
        _save_text(target, DOSTOEVSKY_NOTES_TEXT.strip())
    return target


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build the NLP agent text corpora.")
    p.add_argument("--out", type=Path, default=DATA_RAW_DIR, help="output folder (default: data/raw)")
    p.add_argument("--offline", action="store_true", help="only write embedded fallback corpora")
    p.add_argument("--categories", nargs="*", default=None,
                   help=f"Gutenberg categories (default all): {', '.join(GUTENBERG_CATALOG)}")
    p.add_argument("--no-gutendex", action="store_true", help="skip author-wide Gutendex discovery")
    p.add_argument("--gutendex-max", type=int, default=40, help="max extra books per author")
    p.add_argument("--no-wikipedia", action="store_true")
    p.add_argument("--fandom", action="store_true", help="also crawl the Harry Potter Wiki (CC BY-SA)")
    p.add_argument("--fandom-max", type=int, default=300)
    p.add_argument("--hf", nargs="*", default=None, choices=list(HF_DATASETS),
                   help="optional Hugging Face corpora")
    p.add_argument("--hf-max-rows", type=int, default=50_000)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)

    prepare_all_datasets(
        download_online=not a.offline,
        categories=a.categories,
        gutendex_authors=[] if a.no_gutendex else None,
        gutendex_max_books=a.gutendex_max,
        wikipedia=not a.no_wikipedia,
        fandom=a.fandom,
        fandom_max_pages=a.fandom_max,
        hf=a.hf,
        hf_max_rows=a.hf_max_rows,
        workers=a.workers,
        output_dir=a.out,
    )


if __name__ == "__main__":
    main()
