"""Text web search (SearXNG), page fetching and evidence assembly.

Deliberately simple and deterministic:
- up to ``results_per_query`` results are kept per query;
- sources are ranked round-robin across queries (rank 1 of every query first);
- the first ``pages_to_fetch`` sources get one HTML fetch attempt each, and a
  failed attempt is recorded rather than replaced by the next source;
- page text is the opening excerpt of the extracted body, not a selected passage.
There is no reverse-image search and no PDF extraction.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Config


class SearchError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# URL and date filtering
# ----------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Key for de-duplication and exclusion: scheme/host lower-cased, no fragment, no trailing slash."""
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/") or ""
    return urlunsplit(("https" if p.scheme in ("http", "https") else p.scheme, host, path, p.query, ""))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def domain_excluded(url: str, domains) -> bool:
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)


def parse_date(value) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    return None


def filter_reason(result: dict, domains, excluded_urls: set[str], cutoff: date | None) -> str | None:
    url = result.get("url") or ""
    if not url.startswith(("http://", "https://")):
        return "not_http"
    if domain_excluded(url, domains):
        return "excluded_domain"
    if normalize_url(url) in excluded_urls:
        return "excluded_url"
    if cutoff is not None:
        published = parse_date(result.get("publishedDate"))
        if published is None:
            return "undated_with_cutoff"
        if published > cutoff:
            return "after_cutoff"
    return None


# ----------------------------------------------------------------------------
# SearXNG
# ----------------------------------------------------------------------------

class SearxClient:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=cfg.search_timeout)
        self.base = cfg.search_url.rstrip("/")

    def search(self, query: str) -> dict:
        params = {"q": query, "format": "json", "safesearch": 0, "language": self.cfg.search_language}
        try:
            r = self.http.get(f"{self.base}/search", params=params, timeout=self.cfg.search_timeout,
                              headers={"Accept": "application/json"})
        except httpx.HTTPError as e:
            raise SearchError(f"search backend unreachable: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise SearchError(f"search backend HTTP {r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
        except ValueError as e:
            raise SearchError("search backend did not return JSON (is format=json enabled?)") from e
        results = [{
            "url": x.get("url", ""),
            "title": x.get("title", ""),
            "content": x.get("content", ""),
            "engines": x.get("engines") or ([x["engine"]] if x.get("engine") else []),
            "publishedDate": x.get("publishedDate"),
        } for x in data.get("results", [])]
        return {"results": results[: self.cfg.results_per_query],
                "unresponsive_engines": data.get("unresponsive_engines", [])}


# ----------------------------------------------------------------------------
# HTML -> text
# ----------------------------------------------------------------------------

_SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form",
         "iframe", "template", "button", "select", "figure"}
_BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
          "main", "tr", "blockquote", "pre", "dd", "dt", "figcaption"}
_VOID = {"br", "img", "hr", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.title_parts: list[str] = []
        self.in_title = False
        self.body: list[str] = []
        self.main_depth = 0
        self.main: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return
        if tag in _SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag in ("article", "main"):
            self.main_depth += 1
        if tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in ("article", "main"):
            self.main_depth = max(0, self.main_depth - 1)
        if tag in _BLOCK:
            self._emit("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif not self.skip:
            self._emit(data)

    def _emit(self, s):
        if self.skip:
            return
        self.body.append(s)
        if self.main_depth:
            self.main.append(s)


def _clean(parts: list[str]) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text).  Uses <article>/<main> when it holds at least 300 characters."""
    p = _TextExtractor()
    p.feed(html)
    p.close()
    main = _clean(p.main)
    text = main if len(main) >= 300 else _clean(p.body)
    return re.sub(r"\s+", " ", "".join(p.title_parts)).strip(), text


def excerpt(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.8 else cut) + " …"


class PageFetcher:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=cfg.fetch_timeout, follow_redirects=True)

    def fetch(self, url: str, domains, excluded_urls: set[str]) -> dict:
        """One attempt.  Returns {status: ok|failed|unsupported|excluded, ...}; never raises."""
        try:
            with self.http.stream("GET", url, timeout=self.cfg.fetch_timeout, follow_redirects=True,
                                  headers={"User-Agent": self.cfg.user_agent,
                                           "Accept": "text/html,application/xhtml+xml"}) as r:
                final = str(r.url)
                if domain_excluded(final, domains) or normalize_url(final) in excluded_urls:
                    return {"status": "excluded", "final_url": final, "reason": "redirected to excluded source"}
                if r.status_code != 200:
                    return {"status": "failed", "final_url": final, "reason": f"HTTP {r.status_code}"}
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype == "application/pdf":
                    return {"status": "unsupported", "final_url": final, "reason": "PDF extraction not implemented"}
                if ctype not in ("text/html", "application/xhtml+xml", ""):
                    return {"status": "unsupported", "final_url": final, "reason": f"content-type {ctype}"}
                body = bytearray()
                for chunk in r.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.cfg.max_page_bytes:
                        break
                encoding = r.encoding or "utf-8"
        except httpx.HTTPError as e:
            return {"status": "failed", "final_url": url, "reason": f"{type(e).__name__}: {e}"[:300]}
        html = bytes(body).decode(encoding, errors="replace")
        title, text = html_to_text(html)
        if len(text) < 200:
            return {"status": "failed", "final_url": final, "reason": "no readable text (script-rendered or blocked page?)"}
        return {"status": "ok", "final_url": final, "title": title,
                "excerpt": excerpt(text, self.cfg.page_excerpt_chars), "text_chars": len(text)}


# ----------------------------------------------------------------------------
# Evidence collection
# ----------------------------------------------------------------------------

def collect_evidence(queries: list[str], cfg: Config, searcher: SearxClient, fetcher: PageFetcher,
                     excluded_urls=(), cutoff_date: str | None = None) -> dict:
    """Run every query, filter, de-duplicate, fetch the top pages, build evidence items E1..En."""
    cutoff = parse_date(cutoff_date or cfg.cutoff_date)
    if (cutoff_date or cfg.cutoff_date) and cutoff is None:
        raise ValueError(f"cutoff_date must be YYYY-MM-DD, got {cutoff_date or cfg.cutoff_date!r}")
    excluded = {normalize_url(u) for u in excluded_urls}
    per_query, errors = [], []
    filtered: dict[str, int] = {}
    ranked: list[list[dict]] = []

    for qi, q in enumerate(queries):
        entry = {"query": q, "status": "ok", "n_results": 0, "unresponsive_engines": []}
        try:
            res = searcher.search(q)
        except SearchError as e:
            entry.update(status="error", error=str(e))
            errors.append({"stage": "search", "query": q, "message": str(e)})
            per_query.append(entry)
            ranked.append([])
            continue
        entry["n_results"] = len(res["results"])
        entry["unresponsive_engines"] = res["unresponsive_engines"]
        kept = []
        for rank, r in enumerate(res["results"], 1):
            reason = filter_reason(r, cfg.exclude_domains, excluded, cutoff)
            if reason:
                filtered[reason] = filtered.get(reason, 0) + 1
                continue
            kept.append({**r, "query_index": qi, "rank": rank})
        entry["kept"] = len(kept)
        per_query.append(entry)
        ranked.append(kept)

    # Round-robin merge, first occurrence of a URL wins.
    sources, seen = [], {}
    for rank in range(max((len(k) for k in ranked), default=0)):
        for kept in ranked:
            if rank >= len(kept):
                continue
            r = kept[rank]
            key = normalize_url(r["url"])
            if key in seen:
                seen[key]["query_indices"].append(r["query_index"])
                continue
            src = {"url": r["url"], "title": r["title"], "snippet": excerpt(r["content"] or "", cfg.snippet_chars),
                   "published": str(parse_date(r.get("publishedDate")) or "") or None,
                   "engines": r["engines"], "query_indices": [r["query_index"]], "rank": r["rank"]}
            seen[key] = src
            sources.append(src)

    fetches = []
    for src in sources[: cfg.pages_to_fetch]:
        page = fetcher.fetch(src["url"], cfg.exclude_domains, excluded)
        fetches.append({"url": src["url"], **{k: v for k, v in page.items() if k != "excerpt"}})
        if page["status"] == "excluded":
            src["excluded_after_fetch"] = True
        elif page["status"] == "ok":
            src["page_excerpt"] = page["excerpt"]
            if page["final_url"] != src["url"]:
                src["final_url"] = page["final_url"]
            if not src["title"]:
                src["title"] = page["title"]
        else:
            errors.append({"stage": "fetch", "url": src["url"], "message": page["reason"]})
    sources = [s for s in sources if not s.get("excluded_after_fetch")]

    evidence, used, dropped = [], 0, 0
    for src in sources:
        text = src["snippet"]
        if src.get("page_excerpt"):
            text = (text + "\n" if text else "") + "Page excerpt: " + src["page_excerpt"]
        if not text.strip():
            continue
        if used + len(text) > cfg.evidence_budget_chars:
            dropped += 1
            continue
        used += len(text)
        evidence.append({"id": f"E{len(evidence) + 1}", "url": src["url"], "title": src["title"],
                         "published": src["published"], "engines": src["engines"],
                         "query_indices": src["query_indices"], "fetched": bool(src.get("page_excerpt")),
                         **({"final_url": src["final_url"]} if src.get("final_url") else {}),
                         "text": text})

    return {"queries": per_query, "fetches": fetches, "filtered": filtered,
            "n_sources": len(sources), "dropped_by_budget": dropped, "evidence_chars": used,
            "cutoff_date": str(cutoff) if cutoff else None, "evidence": evidence, "errors": errors}
