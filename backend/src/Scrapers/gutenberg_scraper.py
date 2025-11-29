#!/usr/bin/env python3
import aiohttp
import asyncio
import argparse
from bs4 import BeautifulSoup
from pathlib import Path
import random
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MeetScraper/1.0; +https://example.com/bot)"
}

SEM = asyncio.Semaphore(5)  # limit parallel requests



async def fetch(session, url):
    async with SEM:
        for attempt in range(5):
            try:
                async with session.get(url, headers=HEADERS, timeout=20) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        print(f"⚠️ Status {resp.status} for {url}, retrying...")
            except Exception as e:
                print(f"⚠️ Fetch error for {url}: {e}, retrying...")
            await asyncio.sleep(1 + random.random())
    print(f"❌ Failed to fetch {url} after retries")
    return None



def extract_txt_url(html):
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".txt") or href.endswith(".txt.utf-8") or "txt" in href:
            if href.startswith("http"):
                return href
            return "https://www.gutenberg.org" + href

    return None



async def download_text(session, txt_url):
    for attempt in range(5):
        try:
            async with session.get(txt_url, headers=HEADERS, timeout=20) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    print(f"⚠️ Status {resp.status} for TXT {txt_url}, retrying...")
        except Exception as e:
            print(f"⚠️ TXT fetch error: {e}, retrying...")

        await asyncio.sleep(1 + random.random())

    print(f"❌ Failed to download text: {txt_url}")
    return None



async def process_url(session, url, output_dir):
    try:
        print(f"🔍 Fetching metadata: {url}")
        html = await fetch(session, url)

        txt_url = extract_txt_url(html)
        if not txt_url:
            print(f"❌ No .txt link found: {url}")
            return

        print(f"⬇️ Downloading: {txt_url}")
        text = await download_text(session, txt_url)

        if not text:
            print(f"❌ Failed to save text for: {url}")
            return

        book_id = url.rstrip("/").split("/")[-1]
        filename = f"{book_id}.txt"
        filepath = output_dir / filename

        filepath.write_text(text, encoding="utf-8")
        print(f"✅ Saved: {filepath}")

    except Exception as e:
        print(f"❌ Unexpected error: {url} → {e}")



async def main(url_list, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(url_list):
            await process_url(session, url, output_dir)
            await asyncio.sleep(0.5 + random.random())  # mandatory anti-ban delay



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    urls = Path(args.urls).read_text().strip().splitlines()
    asyncio.run(main(urls, Path(args.out)))
