#!/usr/bin/env python3
"""
Balanced IndianKanoon scraper (category + query-based)

- 4 categories: criminal, property, cybercrime, domestic_violence
- Each category has its own queries and its own output files
- Per cycle:
    - 2 cases from criminal
    - 1 case from property
    - 1 case from cybercrime
    - 1 case from domestic_violence
- Uses per-category seen_ids files so you can stop/resume safely
- Only query-based scraping (no brute-force doc-id ranges)
- Keeps full judgment text in "full_text"
"""

import os
import re
import json
import time
import random
import logging
import argparse
from typing import Optional, List, Dict, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# -----------------------
# CONFIG DEFAULTS
# -----------------------
BASE = "https://indiankanoon.org"
DEFAULT_OUTPUT_ROOT = "/home/meet/LLM_Lawyer/backend/data/raw/kanoon"
DEFAULT_LOG_FILE = os.path.join(DEFAULT_OUTPUT_ROOT, "scraper.log")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# -----------------------
# CATEGORY CONFIG
# -----------------------
CATEGORY_CONFIG: Dict[str, Dict] = {
    "criminal": {
        "cases_per_cycle": 2,
        "queries": [
            "IPC 302",
            "IPC 304",
            "IPC 304B",
            "IPC 307",
            "IPC 376",
            "IPC 420",
            "IPC 406",
            "IPC 498A",

            "bail",
            "anticipatory bail",
            "regular bail",
            "default bail",

            "circumstantial evidence",
            "hostile witness",
            "medical evidence credibility",

            "FIR quashing",
            "charge framing",
            "discharge application",

            "dowry death",
            "domestic cruelty",

            "Section 27 Evidence Act",
            "Section 161 CrPC",
            "Section 164 CrPC",
            "Section 91 CrPC",
            "Section 482 CrPC",

            "criminal conspiracy",
            "criminal breach of trust",
            "cheating and forgery",
            "abetment",
            "attempt to murder",
            "sexual assault",
            "rape conviction set aside",

            "PMLA bail",
            "NDPS Section 27A",
            "NDPS Act seizure",
            "NDPS conviction",
            "illegal search and seizure",
            "chain of custody evidence",

            "POCSO case",
            "child witness credibility",
            "rape false implication",

            "corruption prevention act",
            "CBI trap case",
            "vigilance trap case",

            "kidnapping",
            "dacoity",
            "robbery",
            "arms act conviction",
            "possession of firearm"
        ],
    },
    "property": {
        "cases_per_cycle": 1,
        "queries": [
            "possession suit",
            "specific performance",
            "partition suit",
            "injunction order",
            "property dispute",
            "mutation order",
            "land acquisition",
            "title dispute",
            "encroachment",
            "civil revision petition"
        ],
    },
    "cybercrime": {
        "cases_per_cycle": 1,
        "queries": [
            "Information Technology Act",
            "Section 66",
            "Section 66A",
            "Section 66C",
            "Section 67",
            "electronic evidence",
            "digital signature",
            "electronic record admissible",
            "computer resource",
            "forgery electronic",
            "identity theft",
            "online cheating",
            "digital fraud judgment",
            "ATM fraud",
            "bank fraud",
            "fake website",
            "electronic transaction dispute"
        ],
    },
    "domestic_violence": {
        "cases_per_cycle": 1,
        "queries": [
            "matrimonial dispute",
            "dowry harassment",
            "cruelty husband",
            "Section 498A",
            "Section 12 DV Act",
            "maintenance wife",
            "shared household",
            "domestic abuse",
            "stridhan recovery",
            "marital discord",
            "residence order",
            "maintenance order",
            "Section 125 CrPC"
        ],
    }
}


def attach_paths_to_categories(root: str):
    """Fill in file paths (output, cases_dir, seen_file) for each category."""
    for name, cfg in CATEGORY_CONFIG.items():
        base_dir = os.path.join(root, name)
        cfg["output_master"] = os.path.join(base_dir, f"{name}.jsonl")
        cfg["cases_dir"] = os.path.join(base_dir, "cases")
        cfg["seen_file"] = os.path.join(base_dir, f"seen_ids_{name}.txt")


# -----------------------
# UTILITIES
# -----------------------
def ensure_dirs(*paths):
    for p in paths:
        d = p if p.endswith(os.sep) else os.path.dirname(p)
        if d == "":
            continue
        os.makedirs(d, exist_ok=True)


def write_jsonl(obj: dict, filepath: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(obj: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_seen_ids(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    s: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    s.add(int(line))
                except ValueError:
                    pass
    return s


def save_seen_id(path: str, doc_id: int):
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(doc_id) + "\n")


def safe_get(
    url: str,
    session: requests.Session,
    timeout: int = 15,
    max_retries: int = 3,
    backoff: float = 1.2,
    proxies=None,
) -> Optional[requests.Response]:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout, proxies=proxies)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 410):
                return None
            time.sleep(backoff * (attempt + 1) + random.random() * 0.5)
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    logging.debug(f"safe_get failed for {url}: {last_exc}")
    return None


# -----------------------
# PARSING / EXTRACTION
# -----------------------
def extract_text_from_pre(soup: BeautifulSoup) -> str:
    pres = soup.find_all("pre")
    if pres:
        return "\n\n".join(p.get_text("\n", strip=True) for p in pres)
    return ""


def extract_case_from_soup(url: str, soup: BeautifulSoup) -> dict:
    # --- Title ---
    title = None
    candidates = [
        ("h1", None),
        ("h2", {"class": "title"}),
        ("div", {"class": "doc_title"}),
        ("meta", {"property": "og:title"}),
        ("title", None),
    ]
    for tag, attrs in candidates:
        try:
            if tag == "meta":
                el = soup.find("meta", property="og:title")
                if el and el.get("content"):
                    title = el.get("content").strip()
                    break
            else:
                el = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
                if el:
                    title = el.get_text(strip=True)
                    break
        except Exception:
            continue
    if not title:
        el = soup.find(["b", "strong"])
        title = el.get_text(strip=True) if el else "Unknown Title"

    # --- Court, date, judges ---
    court = None
    date = None
    judges: List[str] = []

    docdesc = soup.find("div", class_=re.compile(r"(docdesc|judgments|docinfo|docheader)"))
    if docdesc:
        desc_text = docdesc.get_text(" ", strip=True)
        m = re.search(
            r"(Supreme Court of India|Supreme Court|[A-Za-z ]+ High Court)",
            desc_text,
            re.I,
        )
        if m:
            court = m.group(0).strip()
        m2 = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", desc_text)
        if m2:
            date = m2.group(0).strip()
        j_matches = re.findall(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+J(?:ustice)?\.?)",
            desc_text,
        )
        for jm in j_matches:
            jm_clean = jm.replace("J.", "").replace("Justice", "").strip()
            if jm_clean and jm_clean not in judges:
                judges.append(jm_clean)

    if not court:
        el = soup.find("span", id="court")
        if el:
            court = el.get_text(strip=True)
    if not date:
        el = soup.find("span", id="judgment_date")
        if el:
            date = el.get_text(strip=True)

    # Additional judge scan near "Author:" / "Bench:"
    for label in ["Author", "Bench", "Coram"]:
        el = soup.find(string=re.compile(rf"^{label}\s*:", re.I))
        if el and el.parent:
            text = el.parent.get_text(" ", strip=True)
            # strip "Author:" or "Bench:" etc
            text = re.sub(rf"^{label}\s*:\s*", "", text, flags=re.I)
            parts = [p.strip() for p in re.split(r"[,&]", text) if p.strip()]
            for p in parts:
                if p and p not in judges:
                    judges.append(p)

    # --- Summary / headnotes ---
    summary = ""
    summary_candidates = [
        ("div", {"id": "headnotes"}),
        ("div", {"class": "synopsis"}),
        ("div", {"class": "summary"}),
    ]
    for tag, attrs in summary_candidates:
        el = soup.find(tag, attrs=attrs)
        if el:
            summary = el.get_text(" ", strip=True)
            break

    # --- Full text ---
    full_text = ""
    strategies = [
        lambda s: s.find("div", id="judgment"),
        lambda s: s.find("div", class_="judgments"),
        lambda s: s.find("div", class_="content"),
        lambda s: s.find("div", class_="doc"),
    ]
    for strat in strategies:
        try:
            el = strat(soup)
            if el:
                full_text = el.get_text("\n", strip=True)
                break
        except Exception:
            continue

    if not full_text:
        full_text = extract_text_from_pre(soup)

    # --- Citations / sections ---
    citations: List[str] = []
    for tag in soup.find_all(["span", "a"], class_=re.compile(r"(citation|doc_citation|citation_ref)", re.I)):
        t = tag.get_text(" ", strip=True)
        if t:
            citations.append(t)

    if not citations:
        txt_top = soup.get_text(" ", strip=True)[:2000]
        cit_matches = re.findall(
            r"\b([A-Z][a-zA-Z]+\s+v\.?\s+[A-Z][a-zA-Z]+(?:\s+[\w\.]+){0,3})\b",
            txt_top,
        )
        citations.extend(cit_matches)

    sections: List[str] = []
    for s_tag in soup.find_all("a", href=True):
        txt = s_tag.get_text(" ", strip=True)
        if re.search(r"(Article|Art\.|Section|S\.)\s*\d", txt, re.I):
            sections.append(txt)

    def norm_list(lst: List[str]) -> List[str]:
        out: List[str] = []
        for v in lst:
            v2 = re.sub(r"\s+", " ", v).strip()
            if v2 and v2 not in out:
                out.append(v2)
        return out

    citations = norm_list(citations)
    sections = norm_list(sections)
    judges = norm_list(judges)

    return {
        "url": url,
        "title": title,
        "court": court or "",
        "date": date or "",
        "judges": judges,
        "citations": citations,
        "sections": sections,
        "summary": summary or "",
        "full_text": full_text or "",   # FULL judgment, uncut
    }


# -----------------------
# SEARCH HELPERS
# -----------------------
def extract_id_from_url(url: str) -> Optional[int]:
    m = re.search(r"/doc/(\d+)/", url)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def search_query_urls(
    query: str,
    pages: int,
    session: requests.Session,
    proxies: Optional[Dict] = None,
) -> List[str]:
    urls: List[str] = []
    for p in range(pages):
        q = requests.utils.requote_uri(query)
        search_url = f"{BASE}/search/?formInput={q}+doctypes:judgments&pagenum={p}"
        resp = safe_get(search_url, session, proxies=proxies)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=re.compile(r"^/doc/\d+/")):
            href = a["href"]
            if href.startswith("/doc/"):
                urls.append(BASE + href)
        time.sleep(random.uniform(0.6, 1.5))
    # keep order but dedupe
    return list(dict.fromkeys(urls))


def scrape_doc_id(
    doc_id: int,
    session: requests.Session,
    output_master: str,
    cases_dir: str,
    seen_file: str,
    proxies: Optional[Dict] = None,
    save_per_case: bool = False,
) -> Optional[int]:
    url = f"{BASE}/doc/{doc_id}/"
    resp = safe_get(url, session, proxies=proxies)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    data = extract_case_from_soup(url, soup)

    ensure_dirs(output_master, os.path.join(cases_dir, "dummy.txt"), seen_file)
    write_jsonl(data, output_master)

    if save_per_case:
        fn = os.path.join(cases_dir, f"{doc_id}.json")
        write_json(data, fn)

    save_seen_id(seen_file, doc_id)
    return doc_id


# -----------------------
# BALANCED CYCLIC SCRAPER
# -----------------------
def run_balanced_scraper(
    pages_per_query: int,
    sleep_between_cases: float,
    save_per_case: bool,
    proxies: Optional[Dict] = None,
):
    session = requests.Session()

    # per-category state
    category_order = ["criminal", "property", "cybercrime", "domestic_violence"]
    seen_ids: Dict[str, Set[int]] = {}
    query_indices: Dict[str, int] = {}

    # initialise paths and state
    for name, cfg in CATEGORY_CONFIG.items():
        ensure_dirs(
            cfg["output_master"],
            os.path.join(cfg["cases_dir"], "dummy.txt"),
            cfg["seen_file"],
        )
        seen_ids[name] = load_seen_ids(cfg["seen_file"])
        query_indices[name] = 0

    logging.info("Starting balanced cyclic scraping across categories.")
    logging.info(
        "Per cycle: criminal=2, property=1, cybercrime=1, domestic_violence=1."
    )
    logging.info("Use Ctrl+C to stop; resume later with the same command.")

    cycle = 409
    while True:
        cycle += 1
        logging.info(f"=== Cycle {cycle} ===")

        for cat_name in category_order:
            cfg = CATEGORY_CONFIG[cat_name]
            queries = cfg["queries"]
            q_count = len(queries)
            needed = cfg["cases_per_cycle"]
            logging.info(
                f"[{cat_name}] Need {needed} new cases this cycle "
                f"(seen so far: {len(seen_ids[cat_name])})"
            )

            attempts = 0  # how many distinct queries we tried this cycle

            with tqdm(total=needed, desc=f"{cat_name} cases", leave=False) as pbar:
                while needed > 0 and attempts < q_count:
                    q_idx = query_indices[cat_name] % q_count
                    query = queries[q_idx]
                    logging.info(f"[{cat_name}] Searching query: {query!r}")

                    urls = search_query_urls(query, pages_per_query, session, proxies)
                    ids = [extract_id_from_url(u) for u in urls]
                    ids = [i for i in ids if i is not None]
                    new_ids = [i for i in ids if i not in seen_ids[cat_name]]

                    if not new_ids:
                        logging.info(
                            f"[{cat_name}] No new doc IDs for query {query!r} "
                            f"(all seen or no results)."
                        )
                        attempts += 1
                        query_indices[cat_name] += 1
                        continue

                    logging.info(
                        f"[{cat_name}] Found {len(new_ids)} new doc IDs for query {query!r}."
                    )

                    for doc_id in new_ids:
                        if needed <= 0:
                            break
                        try:
                            scraped_id = scrape_doc_id(
                                doc_id,
                                session=session,
                                output_master=cfg["output_master"],
                                cases_dir=cfg["cases_dir"],
                                seen_file=cfg["seen_file"],
                                proxies=proxies,
                                save_per_case=save_per_case,
                            )
                            if scraped_id:
                                seen_ids[cat_name].add(scraped_id)
                                logging.info(
                                    f"[{cat_name}] Saved doc {scraped_id} "
                                    f"(total now {len(seen_ids[cat_name])})."
                                )
                                needed -= 1
                                pbar.update(1)
                                time.sleep(
                                    sleep_between_cases
                                    + random.uniform(0.0, sleep_between_cases)
                                )
                        except Exception as e:
                            logging.exception(
                                f"[{cat_name}] Exception scraping doc {doc_id}: {e}"
                            )
                            continue

                    query_indices[cat_name] += 1

                if needed > 0:
                    logging.warning(
                        f"[{cat_name}] Could not fill quota this cycle. "
                        f"Remaining needed: {needed}"
                    )


# -----------------------
# CLI
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Balanced IndianKanoon scraper (4 categories, cyclic)."
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for all categories.",
    )
    p.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Pages per search query (per call).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Base sleep between scraping two cases (seconds).",
    )
    p.add_argument(
        "--log",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Log file path.",
    )
    p.add_argument(
        "--save_per_case",
        action="store_true",
        help="If set, save one JSON file per case in category-specific cases_dir.",
    )
    p.add_argument(
        "--use_proxies",
        action="store_true",
        help="Enable proxies (set PROXIES env var like http://user:pass@host:port).",
    )
    return p


def main():
    args = build_argparser().parse_args()

    # Update paths based on chosen root
    attach_paths_to_categories(args.output_root)

    # logging
    ensure_dirs(args.log)
    logging.basicConfig(
        level=logging.INFO,
        filename=args.log,
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Starting balanced_indian_kanoon_scraper")

    proxies = None
    if args.use_proxies:
        prox = os.environ.get("PROXIES", "")
        if prox:
            proxies = {"http": prox, "https": prox}
            logging.info("Using proxy from PROXIES env var")

    try:
        run_balanced_scraper(
            pages_per_query=args.pages,
            sleep_between_cases=args.sleep,
            save_per_case=args.save_per_case,
            proxies=proxies,
        )
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt. Gracefully stopping scraper.")


if __name__ == "__main__":
    main()
