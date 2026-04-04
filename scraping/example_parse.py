"""Parse static HTML with BeautifulSoup (minimal example)."""

from __future__ import annotations

from bs4 import BeautifulSoup

SAMPLE_HTML = """
<html><body>
<article class="release">
  <h1>Federal Reserve issues FOMC statement</h1>
  <time datetime="2020-01-29">January 29, 2020</time>
  <p>The Committee decided to maintain the target range for the federal funds rate.</p>
</article>
</body></html>
"""


def main() -> None:
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    title = soup.find("h1")
    para = soup.find("p")
    print("Title:", title.get_text(strip=True) if title else None)
    print("First paragraph:", para.get_text(strip=True) if para else None)


if __name__ == "__main__":
    main()
