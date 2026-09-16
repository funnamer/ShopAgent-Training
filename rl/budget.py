"""Dependency-free copy of the ShopSimulator visible-budget policy."""

from __future__ import annotations

import math
import re


APPROXIMATE_BUDGET_TOLERANCE = 0.10
_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_NUMBER = r"(?:\d+(?:\.\d+)?\s*(?:[kK]|万|千|百)?|[零〇一二两三四五六七八九十百千万]+)"
_CONTEXT = r"(?:预算(?:价格)?|价格|售价|价位|价钱|心理价位)"
_RANGE = re.compile(
    rf"{_CONTEXT}[^，。！？]{{0,8}}?(?P<lower>{_NUMBER})\s*(?:元|块(?:钱)?)?\s*[-—~～至到]\s*"
    rf"(?P<upper>{_NUMBER})\s*(?:元|块(?:钱)?)?(?:之间|以内|以下)?"
)
_LOWER_ONLY = re.compile(
    rf"{_CONTEXT}[^，。！？]{{0,8}}?(?P<number>{_NUMBER})\s*(?:元|块(?:钱)?)?\s*(?:以上|起|\+)"
)
_CURRENCY = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?:元|块(?:钱)?)(?P<suffix>以内|以下|之内|内|左右|上下|附近|多|来块?)?"
)
_WITHOUT_CURRENCY = re.compile(
    rf"{_CONTEXT}(?:在|为|是|控制在|控制为|就|大概|大约|约|别超过|不要超过|不超过|低于)?\s*"
    rf"(?P<number>{_NUMBER})(?!\s*\+)(?P<suffix>以内|以下|之内|内|左右|上下|附近|多)?"
)


def _number(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kK]|万|千|百)?", value.strip())
    if match:
        multiplier = {None: 1, "k": 1000, "K": 1000, "万": 10000, "千": 1000, "百": 100}[match.group(2)]
        return float(match.group(1)) * multiplier
    total = section = 0.0
    digit = None
    last_unit = None
    for char in value.strip():
        if char in _DIGITS:
            digit = _DIGITS[char]
            continue
        unit = _UNITS[char]
        if unit == 10000:
            section += 0 if digit is None else digit
            total += section * unit
            section, digit, last_unit = 0.0, None, None
        else:
            section += (1 if digit is None else digit) * unit
            digit, last_unit = None, unit
    if digit is not None:
        section += digit * (last_unit / 10) if last_unit and last_unit >= 100 else digit
    return total + section


def extract_price_upper(instruction_text: str | None) -> float | None:
    if not instruction_text:
        return None
    if "个位数价格" in instruction_text:
        return 9.0
    ranges = list(_RANGE.finditer(instruction_text))
    if ranges:
        return _number(ranges[-1].group("upper"))
    if _LOWER_ONLY.search(instruction_text):
        return None
    matches = list(_CURRENCY.finditer(instruction_text)) or list(_WITHOUT_CURRENCY.finditer(instruction_text))
    if not matches:
        return None
    match = matches[-1]
    amount = _number(match.group("number"))
    suffix = match.group("suffix") or ""
    context = instruction_text[max(0, match.start() - 8):match.end() + 4]
    if suffix in {"多", "来", "来块"}:
        magnitude = 10 ** max(1, int(math.log10(amount))) if amount > 0 else 1
        return float((math.floor(amount / magnitude) + 1) * magnitude)
    if suffix in {"左右", "上下", "附近"} or any(x in context for x in ("大概", "大约", "约", "差不多")):
        return round(amount * (1 + APPROXIMATE_BUDGET_TOLERANCE), 2)
    return amount

