# Web scraping (example)

Not all Fed content is available via a convenient API. This folder holds a **minimal BeautifulSoup example** for learning; it does **not** crawl the Fed by default.

For the documented dataset refresh flow that pulls from **federalreserve.gov** (statements, minutes, press-conference transcripts, speeches, and testimony) plus controlled supplements, see `fed_official.py` and `scripts/rebuild_database.py`.

Before scraping any site:

- Read **robots.txt** and the site’s **terms of use**.
- Prefer official **APIs** (e.g. economic data providers) when available.
- Rate-limit requests and cache pages for reproducibility.

Run the example (offline HTML string):

```bash
cd fomc_tools
python scraping/example_parse.py
```
