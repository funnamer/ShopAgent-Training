"""Deterministic multi-field BM25 search for ShopSimulator.

The search index is a SQLite FTS5 database. Product text is pre-tokenized with
a small, deterministic Chinese-bigram tokenizer, so search does not depend on
Pyserini, Java, spaCy, Transformers, or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union


SEARCH_VERSION = "shopsimulator-multifield-bm25-v1"
INDEX_SCHEMA_VERSION = 1
DEFAULT_FIELD_WEIGHTS = {
    "title": 3.0,
    "brand": 2.0,
    "category": 2.0,
    "model": 2.5,
    "attributes": 1.5,
    "options": 1.2,
    "bullets": 0.8,
}
FIELDS = tuple(DEFAULT_FIELD_WEIGHTS)

_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*")
_MODEL = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+\-/]*[a-z])(?=[a-z0-9._+\-/]*\d)"
    r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*(?![a-z0-9])"
)

PathLike = Union[str, Path]


class SearchIndexError(RuntimeError):
    """Raised when a search index is absent, incompatible, or corrupted."""


@dataclass(frozen=True)
class SearchHit:
    asin: str
    score: float
    rank: int


def normalize_query(value: object) -> str:
    """Normalize text without language-model or machine-local dependencies."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("人民币", "元").replace("块钱", "元")
    text = re.sub(r"(?<=\d)\s*(?:rmb|cny|yuan)\b", "元", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,。；;：:！!？?、|]+", " ", text)
    return " ".join(text.split())


def search_tokens(value: object) -> Tuple[str, ...]:
    """Return deterministic Chinese bigrams plus Latin/number tokens."""
    text = normalize_query(value)
    tokens: List[str] = []
    for match in _HAN_RUN.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(_LATIN_OR_NUMBER.findall(text))
    return tuple(dict.fromkeys(token for token in tokens if token))


def _flatten(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: List[str] = []
        for key in sorted(value, key=str):
            flattened.append(str(key))
            flattened.extend(_flatten(value[key]))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [str(value)]


def product_fields(product: Mapping[str, object]) -> Dict[str, str]:
    """Build public search fields; task goals and reward data are excluded."""
    title = str(product.get("title") or product.get("Title") or "")
    brand = str(product.get("brand") or product.get("shop_name") or "")
    category = str(product.get("category") or product.get("product_category") or "")
    attributes = " ".join(
        _flatten(product.get("attribute") or product.get("Attributes") or [])
    )
    options = " ".join(
        _flatten(
            product.get("customization_options") or product.get("options") or {}
        )
    )
    bullets = " ".join(
        _flatten(
            product.get("small_description")
            or product.get("sub_title")
            or product.get("BulletPoints")
            or []
        )
    )
    model = " ".join(dict.fromkeys(_MODEL.findall(normalize_query(title))))
    return {
        "title": title,
        "brand": brand,
        "category": category,
        "model": model,
        "attributes": attributes,
        "options": options,
        "bullets": bullets,
    }


def _tokenized(value: object) -> str:
    return " ".join(search_tokens(value))


def _validated_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    if set(weights) != set(FIELDS):
        raise ValueError("field weights must contain exactly: " + ", ".join(FIELDS))
    result = {}
    for field in FIELDS:
        value = float(weights[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"weight for {field} must be finite and positive")
        result[field] = value
    return result


def product_data_fingerprint(products: Iterable[Mapping[str, object]]) -> str:
    """Hash exactly the public fields that influence retrieval."""
    digest = hashlib.sha256()
    for product in products:
        asin = str(product.get("asin", "")).strip()
        if not asin or asin == "nan":
            continue
        payload = {"asin": asin, **product_fields(product)}
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # They are cache-key components that invalidate stale hashes.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: PathLike) -> str:
    """Hash a product file once per process, invalidating on size/mtime change."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


def build_index(
    products: Iterable[Mapping[str, object]],
    output_path: PathLike,
    *,
    product_data_sha256: str,
    field_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Atomically build a deterministic SQLite FTS5 search index."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()

    weights = _validated_weights(field_weights or DEFAULT_FIELD_WEIGHTS)
    connection = sqlite3.connect(str(temporary))
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "asin UNINDEXED, title, brand, category, model, attributes, options, bullets,"
            " tokenize='unicode61 remove_diacritics 2')"
        )
        seen = set()
        count = 0
        for product in products:
            asin = str(product.get("asin", "")).strip()
            if not asin or asin == "nan" or asin in seen:
                continue
            fields = product_fields(product)
            connection.execute(
                "INSERT INTO products(asin, title, brand, category, model, attributes, options, bullets)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (asin, *(_tokenized(fields[field]) for field in FIELDS)),
            )
            seen.add(asin)
            count += 1

        manifest = {
            "search_version": SEARCH_VERSION,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "product_count": count,
            "product_data_sha256": str(product_data_sha256),
            "field_weights": weights,
            "query_normalizer": "nfkc-casefold-basic-punctuation-v1",
            "tokenizer": "han-bigram-latin-number-v1",
            "tie_breaker": "asin-ascending",
        }
        connection.execute("CREATE TABLE manifest(payload TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO manifest(payload) VALUES (?)",
            (json.dumps(manifest, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()

    temporary.replace(output)
    return manifest


def _fts_query(tokens: Tuple[str, ...]) -> str:
    return " OR ".join(f'"{token}"' for token in tokens)


class MultiFieldBM25Searcher:
    """Read-only search interface compatible with ShopSimulator's SimServer."""

    def __init__(
        self,
        index_path: PathLike,
        *,
        expected_product_sha256: Optional[str] = None,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise SearchIndexError(f"search index not found: {self.index_path}")

        uri = f"file:{self.index_path}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        try:
            row = self._connection.execute("SELECT payload FROM manifest").fetchone()
            if row is None:
                raise ValueError("manifest is empty")
            self.manifest = json.loads(row["payload"])
        except Exception as exc:
            self._connection.close()
            raise SearchIndexError(f"invalid search index manifest: {exc}") from exc

        if self.manifest.get("search_version") != SEARCH_VERSION:
            self.close()
            raise SearchIndexError("search version mismatch")
        if self.manifest.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            self.close()
            raise SearchIndexError("index schema version mismatch")
        if (
            expected_product_sha256 is not None
            and self.manifest.get("product_data_sha256") != expected_product_sha256
        ):
            self.close()
            raise SearchIndexError("product data does not match the search index")
        self.weights = _validated_weights(self.manifest["field_weights"])

    def search(self, query: object, k: int = 150) -> List[SearchHit]:
        tokens = search_tokens(query)
        if not tokens or int(k) <= 0:
            return []
        weight_values = [self.weights[field] for field in FIELDS]
        weight_placeholders = ", ".join("?" for _ in weight_values)
        rows = self._connection.execute(
            f"SELECT asin, bm25(products, {weight_placeholders}) AS score "
            "FROM products WHERE products MATCH ? "
            "ORDER BY score ASC, asin ASC LIMIT ?",
            (*weight_values, _fts_query(tokens), int(k)),
        ).fetchall()
        return [
            SearchHit(asin=row["asin"], score=-float(row["score"]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def close(self) -> None:
        self._connection.close()
