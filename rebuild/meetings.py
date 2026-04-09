"""FOMC meeting calendar parsing + historic dates missing from cached Fed calendar page."""

from __future__ import annotations

import re
from pathlib import Path

# Official statement dates (YYYYMMDD), Jan 2015 – Dec 2019 — Fed calendar HTML in cache starts ~2021.
_MEETINGS_2015_2019: frozenset[str] = frozenset(
    [
        "20150128",
        "20150318",
        "20150429",
        "20150617",
        "20150729",
        "20150917",
        "20151028",
        "20151216",
        "20160127",
        "20160316",
        "20160427",
        "20160615",
        "20160727",
        "20160921",
        "20161102",
        "20161214",
        "20170201",
        "20170315",
        "20170503",
        "20170614",
        "20170726",
        "20170920",
        "20171101",
        "20171213",
        "20180131",
        "20180321",
        "20180502",
        "20180613",
        "20180801",
        "20180926",
        "20181108",
        "20181219",
        "20190130",
        "20190320",
        "20190501",
        "20190619",
        "20190731",
        "20190918",
        "20191030",
        "20191211",
    ]
)

# 2020 meetings not always linked the same way on archived calendar snippets; include full set.
_MEETINGS_2020_EXTRA: frozenset[str] = frozenset(
    [
        "20200129",
        "20200303",
        "20200315",
        "20200429",
        "20200610",
        "20200729",
        "20200916",
        "20201105",
        "20201216",
    ]
)

_MINUTES_RE = re.compile(r"fomcminutes(20\d{6})\.pdf", re.I)
_MONETARY_HREF_RE = re.compile(r"/newsevents/pressreleases/monetary(20\d{6})[a-z]\.htm", re.I)


def meeting_dates_from_calendar_html(html: str) -> set[str]:
    out: set[str] = set(_MINUTES_RE.findall(html))
    out.update(_MONETARY_HREF_RE.findall(html))
    return out


def all_meeting_dates(
    calendar_html_path: Path | None,
    *,
    min_ymd: str,
    max_ymd: str,
    extra_calendar_html: str | None = None,
) -> list[str]:
    found: set[str] = set(_MEETINGS_2015_2019)
    found.update(_MEETINGS_2020_EXTRA)
    if calendar_html_path and calendar_html_path.is_file():
        found.update(meeting_dates_from_calendar_html(calendar_html_path.read_text(encoding="utf-8", errors="replace")))
    if extra_calendar_html:
        found.update(meeting_dates_from_calendar_html(extra_calendar_html))
    bounded = [d for d in found if min_ymd <= d <= max_ymd]
    return sorted(bounded)
