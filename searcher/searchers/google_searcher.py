"""
Google web searcher implementation using Serper API.
"""

import hashlib
import http.client
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class GoogleSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--max-qps",
            type=int,
            default=8,
            help="Maximum requests per second for Serper (default: 8)",
        )
        parser.add_argument(
            "--serper-api-key-env",
            type=str,
            default="SERPER_API_KEY",
            help="Environment variable name storing Serper API key (default: SERPER_API_KEY)",
        )
        parser.add_argument(
            "--serper-location",
            type=str,
            default="United States",
            help="Default location for non-Chinese queries (default: United States)",
        )
        parser.add_argument(
            "--serper-gl",
            type=str,
            default="us",
            help="Default country code for non-Chinese queries (default: us)",
        )
        parser.add_argument(
            "--serper-hl",
            type=str,
            default="en",
            help="Default language code for non-Chinese queries (default: en)",
        )

    def __init__(self, reranker, args):
        super().__init__(reranker)
        load_dotenv()
        self.max_qps = args.max_qps
        self.request_times = deque()
        self.default_location = args.serper_location
        self.default_gl = args.serper_gl
        self.default_hl = args.serper_hl

        self.serper_api_key = os.getenv(args.serper_api_key_env)
        if not self.serper_api_key:
            raise ValueError(
                f"Serper API key missing. Set env var '{args.serper_api_key_env}'."
            )

        self.docid_to_text: Dict[str, str] = {}

    def _wait_for_rate_limit(self):
        """Ensures at most max_qps requests per second."""
        now = time.time()
        while self.request_times and self.request_times[0] <= now - 1:
            self.request_times.popleft()

        if len(self.request_times) >= self.max_qps:
            sleep_time = self.request_times[0] + 1.1 - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            return self._wait_for_rate_limit()

        self.request_times.append(time.time())

    @staticmethod
    def _contains_chinese_basic(text: str) -> bool:
        return any("\u4E00" <= char <= "\u9FFF" for char in text)

    def _search_serper(self, query: str) -> dict:
        self._wait_for_rate_limit()
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

        if self._contains_chinese_basic(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
            }
        else:
            payload = {
                "q": query,
                "location": self.default_location,
                "gl": self.default_gl,
                "hl": self.default_hl,
            }

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(5):
            try:
                conn.request("POST", "/search", json.dumps(payload), headers)
                response = conn.getresponse()
                response_data = response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"Serper request failed with status {response.status}: {response_data.decode('utf-8', errors='ignore')}"
                    )
                return json.loads(response_data.decode("utf-8"))
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(
                        f"Serper request failed after retries: {exc}"
                    ) from exc
                time.sleep(0.5 * (attempt + 1))
            finally:
                conn.close()

    @staticmethod
    def _build_docid(result: Dict[str, Any], rank: int) -> str:
        canonical = result.get("link") or result.get("title") or f"rank-{rank}"
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        return f"google:{digest}"

    def _retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        raw_results = self._search_serper(query)
        organic_results = raw_results.get("organic", [])[:k]

        processed_results = []
        for idx, result in enumerate(organic_results, 1):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            source = result.get("source", "")
            date = result.get("date", "")

            text = "\n".join(
                [
                    f"title: {title}",
                    f"url: {link}",
                    f"source: {source}",
                    f"date: {date}",
                    f"snippet: {snippet}",
                ]
            )
            docid = self._build_docid(result, idx)
            processed_results.append(
                {
                    "docid": docid,
                    "score": 1.0 / idx,
                    "text": text,
                }
            )
            self.docid_to_text[docid] = text

        return processed_results

    def retrieve_batch(
        self, queries: List[str], qids: List[str], k: int = 10
    ) -> Dict[str, list[dict[str, Any]]]:
        results = {}
        for qid, query in zip(qids, queries):
            results[qid] = self._retrieve(query, k)
        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.docid_to_text:
            raise RuntimeError("No search history")

        text = self.docid_to_text.get(docid)
        if text is None:
            return None

        return {
            "docid": docid,
            "text": text,
        }

    @property
    def search_type(self) -> str:
        return "GOOGLE"
