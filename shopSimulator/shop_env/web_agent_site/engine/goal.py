"""
Functions for specifying goals and reward calculations.
"""
import re
import spacy
from collections import defaultdict
from rich import print
from thefuzz import fuzz
from web_agent_site.engine.normalize import normalize_color
import math

nlp = spacy.load("zh_core_web_sm")

APPROXIMATE_BUDGET_TOLERANCE = 0.10
NO_PRICE_LIMIT = None

_CHINESE_DIGITS = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3,
    '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_CHINESE_UNITS = {'十': 10, '百': 100, '千': 1000, '万': 10000}
_PRICE_NUMBER = (
    r'(?:\d+(?:\.\d+)?\s*(?:[kK]|万|千|百)?|'
    r'[零〇一二两三四五六七八九十百千万]+)'
)
_PRICE_CONTEXT = r'(?:预算(?:价格)?|价格|售价|价位|价钱|心理价位)'
_PRICE_RANGE_RE = re.compile(
    rf'{_PRICE_CONTEXT}[^，。！？]{{0,8}}?'
    rf'(?P<lower>{_PRICE_NUMBER})\s*(?:元|块(?:钱)?)?\s*[-—~～至到]\s*'
    rf'(?P<upper>{_PRICE_NUMBER})\s*(?:元|块(?:钱)?)?(?:之间|以内|以下)?'
)
_LOWER_BOUND_ONLY_RE = re.compile(
    rf'{_PRICE_CONTEXT}[^，。！？]{{0,8}}?(?P<number>{_PRICE_NUMBER})'
    rf'\s*(?:元|块(?:钱)?)?\s*(?:以上|起|\+)'
)
_CURRENCY_PRICE_RE = re.compile(
    rf'(?P<number>{_PRICE_NUMBER})\s*(?:元|块(?:钱)?)'
    rf'(?P<suffix>以内|以下|之内|内|左右|上下|附近|多|来块?)?'
)
_BUDGET_WITHOUT_CURRENCY_RE = re.compile(
    rf'{_PRICE_CONTEXT}'
    rf'(?:在|为|是|控制在|控制为|就|大概|大约|约|别超过|不要超过|不超过|低于)?\s*'
    rf'(?P<number>{_PRICE_NUMBER})(?!\s*\+)'
    rf'(?P<suffix>以内|以下|之内|内|左右|上下|附近|多)?'
)

def get_price_range_above(price, count=4):
    """
    Get the first N price options above the specified price.
    Round up the current price to the nearest multiple of 10, then generate the next N prices.

    Args:
        price: Base price
        count: Number of prices to return

    Returns:
        list: List of prices
    """
    if price <= 100:
        count = 3
    elif price <= 1000:
        count = 10
    elif price <= 5000:
        count = 50
    elif price <= 10000:
        count = 100

    # Round up to the nearest multiple of 10
    base_price = math.ceil(price / 10) * 10

    # Generate the next count prices with step size of 10
    return [base_price + i * 10 for i in range(count)]


def _parse_chinese_number(value):
    """Parse the Chinese integer forms used by shopping-budget instructions."""
    value = value.strip()
    numeric_match = re.fullmatch(
        r'(\d+(?:\.\d+)?)\s*([kK]|万|千|百)?', value
    )
    if numeric_match:
        multiplier = {
            None: 1,
            'k': 1000,
            'K': 1000,
            '万': 10000,
            '千': 1000,
            '百': 100,
        }[numeric_match.group(2)]
        return float(numeric_match.group(1)) * multiplier

    total = 0
    section = 0
    digit = None
    last_small_unit = None
    for char in value:
        if char in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS[char]
        if unit == 10000:
            section += 0 if digit is None else digit
            total += section * unit
            section = 0
            digit = None
            last_small_unit = None
        else:
            section += (1 if digit is None else digit) * unit
            digit = None
            last_small_unit = unit

    if digit is not None:
        # Common colloquial forms: 一千六 = 1600, 一百二 = 120.
        if last_small_unit and last_small_unit >= 100:
            section += digit * (last_small_unit / 10)
        else:
            section += digit
    return float(total + section)


def extract_price_upper(instruction_text):
    """Return a deterministic upper price inferred from the visible instruction.

    Exact limits such as ``100元以内`` use the stated value. Approximate budgets
    such as ``40元左右`` receive a small, documented tolerance so the intended
    nearby SKU is not rejected solely for a minor price difference.
    """
    if not instruction_text:
        return NO_PRICE_LIMIT

    if '个位数价格' in instruction_text:
        return 9.0

    range_matches = list(_PRICE_RANGE_RE.finditer(instruction_text))
    if range_matches:
        return _parse_chinese_number(range_matches[-1].group('upper'))

    # Expressions such as “预算4k+” specify a lower bound, not an upper limit.
    if _LOWER_BOUND_ONLY_RE.search(instruction_text):
        return NO_PRICE_LIMIT

    matches = list(_CURRENCY_PRICE_RE.finditer(instruction_text))
    if not matches:
        matches = list(_BUDGET_WITHOUT_CURRENCY_RE.finditer(instruction_text))
    if not matches:
        return NO_PRICE_LIMIT

    match = matches[-1]
    amount = _parse_chinese_number(match.group('number'))
    suffix = match.group('suffix') or ''
    nearby_context = instruction_text[max(0, match.start() - 8):match.end() + 4]
    approximate = suffix in {'左右', '上下', '附近'} or any(
        marker in nearby_context for marker in ('大概', '大约', '约', '差不多')
    )
    if suffix in {'多', '来', '来块'}:
        # 三十多 means below the next ten; 300多 below the next hundred.
        magnitude = 10 ** max(1, int(math.log10(amount))) if amount > 0 else 1
        return float((math.floor(amount / magnitude) + 1) * magnitude)
    if approximate:
        return round(amount * (1 + APPROXIMATE_BUDGET_TOLERANCE), 2)
    return amount


def resolve_purchase_price(purchased_product, selected_options, fallback_price):
    """Resolve the checkout price shown for the last selected priced SKU."""
    option_to_price = purchased_product.get('option_to_price') or {}
    for option in reversed(list((selected_options or {}).values())):
        option_price = option_to_price.get(option)
        if option_price is not None:
            return float(option_price)
    return float(fallback_price)

def get_goals(all_products, product_prices, if_persona=False):
    return get_existed_goals(all_products, product_prices, if_persona)

def get_existed_goals(all_products, product_prices, if_persona=False):
    goals = []
    cnt_atts = defaultdict(int)
    cnt_1, cnt_2, cnt_3 = 0, 0, 0
    goal_instructions = []
    for item in all_products:
        if 'instructions' not in item:
            cnt_1 += 1
            continue
        asin = item['asin']
        for product in item['instructions']:
            if product['instruction'] in goal_instructions:
                cnt_2 += 1
                #continue
            else:
                goal_instructions.append(product['instruction'])
            attributes = item['instruction_attributes']
            if len(attributes) == 0:
                cnt_3 += 1
                continue

            # Process user_persona, place __reasoning__ field in the first position
            if not isinstance(item['user_persona'], dict):
                item['user_persona'] = {}
            user_persona = item['user_persona'].copy()
            reason_key = item['reason_key']
            if user_persona and '__reasoning__' in user_persona:
                reasoning_value = user_persona.pop('__reasoning__')
                # Create a new ordered dictionary with __reasoning__ in the first position
                ordered_persona = {'__reasoning__': reasoning_value}
                ordered_persona.update(user_persona)
                user_persona = ordered_persona

            if if_persona:
                instruction_text = product['instruction_sample']
            else:
                instruction_text = product['instruction']
            price_upper = extract_price_upper(instruction_text)
            goals.append({
                'asin': asin,
                'category': item['category'],
                'query': item['query'],
                'name': item['title'],
                'instruction_text': instruction_text,
                'instruction_simple': product['instruction_simple'],
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': product['instruction_options'],
                'user_persona': user_persona,
                'reason_key': reason_key,
            })
            for att in attributes:
                cnt_atts[att] += 1
            # goals += product_goals
    for goal in goals:
        goal['weight'] = 1
    print('skipped')
    print(cnt_1, cnt_2, cnt_3)
    return goals

def get_type_reward(purchased_product, goal):
    """Determines the type reward - captures whether chosen product is in the same category"""
    purchased_query = (purchased_product.get('query') or '').strip().lower()
    goal_query = (goal.get('query') or '').strip().lower()
    query_match = bool(purchased_query and goal_query and purchased_query == goal_query)

    # Check number of unique categories that match, ignoring order
    purchased_product_category = [x.strip() for x in purchased_product['category'].split('›')]
    goal_product_category = [x.strip() for x in goal['category'].split('›')]
    category_match = len(set(purchased_product_category) & set(goal_product_category)) >= 2

    # Determine whether types align based on product name similarity
    purchased_type = purchased_product['title']
    desired_type = goal['name']

    purchased_type_parse = nlp(purchased_type)
    desired_type_parse = nlp(desired_type)

    purchased_type_parse = [t.text.lower() for t in purchased_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]
    desired_type_parse = [t.text.lower() for t in desired_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]

    n_intersect_type = len(
        set(purchased_type_parse) & set(desired_type_parse)
    )
    if len(desired_type_parse) == 0:
        title_score = 0.2
    else:
        title_score = n_intersect_type / len(desired_type_parse)

    r_type = 1.0

    # Adjust r_type score based on query, category title matching/scores
    match = query_match or category_match or title_score > 0.2
    if not match:
        r_type = 0.5

    return dict(
        r_type=r_type,
        query_match=query_match,
        category_match=category_match,
        title_score=title_score,
    )

def get_attribute_reward(purchased_product, goal):
    """Determines whether purchased products shares same attributes as goal"""
    purchased_attrs = purchased_product['Attributes']
    goal_attrs = goal['attributes']

    num_attr_matches = 0
    for g_attr in goal_attrs:
        matched = False
        # Check whether goal attribute found in purchased product attribute list
        for p_attr in purchased_attrs:
            score = fuzz.token_set_ratio(p_attr, g_attr)
            if score > 85:
                num_attr_matches += 1
                matched = True
                break
        # If not in purchased attrs, check Title, Bullet Points (Features), Desc
        if (
            not matched and
            (
                g_attr in purchased_product['Title'].lower() or
                g_attr in ' '.join(purchased_product['BulletPoints']).lower() or
                g_attr in purchased_product['Description'].lower()
            )
        ):
            num_attr_matches += 1
            matched = True
    r_attr = num_attr_matches / len(goal_attrs)
    return r_attr, num_attr_matches

def get_option_reward(purchased_options, goal_options):
    """Calculate reward for purchased product's options w.r.t. goal options"""
    def safe_normalize(option):
        if isinstance(option, str):
            return normalize_color(option)
        return str(option)  # Convert non-string options to string

    purchased_options = [safe_normalize(o) for o in purchased_options]
    goal_options = [safe_normalize(o) for o in goal_options]

    # Perform fuzzy matching of each purchased option against each goal option
    num_option_matches = 0
    for g_option in goal_options:
        for p_option in purchased_options:
            score = fuzz.token_set_ratio(p_option, g_option)
            if score > 85:
                num_option_matches += 1
                break
    # Calculate option reward as fraction of goal options hit
    r_option = num_option_matches / len(goal_options) if len(goal_options) > 0 else 1
    return r_option, num_option_matches

def get_reward(purchased_product, goal, price, options, **kwargs):
    """Get cumulative reward score for purchased product and goal"""
    r_type_dict = get_type_reward(purchased_product, goal)
    price = resolve_purchase_price(purchased_product, options, price)
    purchased_product['price'] = price

    price_upper = goal.get('price_upper')
    r_price = price <= price_upper if price_upper is not None and price_upper > 0 else 1

    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)

    r_option, num_option_matches = get_option_reward(
        list(options.values()),
        goal['goal_options'].items()
        if isinstance(goal['goal_options'], dict)
        else goal['goal_options']
    )

    total_reward = (
        (num_attr_matches + num_option_matches + r_price) \
            / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    )
    total_reward *= r_type_dict['r_type']

    # If verbose flag enabled, store score sub-components into dictionary
    if kwargs.get('verbose', False):
        info =  {
            'query_match': r_type_dict['query_match'],
            'category_match': r_type_dict['category_match'],
            'title_score': r_type_dict['title_score'],
            'num_attr_matches': num_attr_matches,
            'num_option_matches': num_option_matches,
            'r_type': r_type_dict['r_type'],
            'r_att': r_att,
            'evaluated_price': price,
            'price_upper': price_upper,
        }
        if r_option is not None:
            info['r_option'] = r_option
        if r_price is not None:
            info['r_price'] = r_price
        return total_reward, info
    return total_reward
