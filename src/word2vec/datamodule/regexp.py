import regex


def join_regexes(patterns: list[regex.Pattern]):
    return "".join(x for x in patterns)


RE_TYPO_BASE_DELIMITERS_AFTER_SPACE = regex.compile(r"(?<=\p{L}\p{L}) ([.,?!]+)")
RE_TYPO_BASE_DELIMITERS_SQUEEZED = regex.compile(r"(?<=\p{L}\p{L})([.,?!]+)(?=\p{L}\p{L})")
RE_TYPO_BASE_DELIMITERS_CENTERED = regex.compile(r"(?<=\p{L}\p{L}) ([.,?!]+) (?=\p{L}\p{L})")
RE_TYPO_MULTIPLE_SPACE = regex.compile(r" +")
RE_TYPO_LOWERCASE_START_OF_SENTENCE = regex.compile(r"(?<=\p{L}\p{L}[.?!] )(\p{Ll})")
RE_MULTIPLE_EMOTION_DELIMITERS = regex.compile(r"([!?]+)")

RE_FIX_BASE_DELIMITERS = r"\1 "
RE_FIX_UPPERCASE_START_OF_SENTENCE = lambda x: x.group(1).upper()
RE_FIX_SINGLE_SPACE = r" "
RE_FIX_SINGLE_EMOTION_DELIMITER = lambda x: x.group(1)[0]

RE_EN_MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
RE_EN_DAY = r"\d{1,2}(th|rd|nd|st)?"
RE_EN_YEAR = r"\d{4}"
RE_NUMBER = r"\d+"

RE_EN_CCY_SIGN = r"\p{Sc}\d+(,\d+)*(\.\d\d)?"
RE_EN_CCY_NAME = r"\d+ ?([dD]ollars?|[eE]uros?|[pP]ounds?)"
RE_EN_TIME = r"(?<=\b)(\d?\d[.:]\d\d ?([ap]m))|(\d?\d[.:]\d\d)|(\d?\d ?([ap]m))(?=\b)"
RE_EN_DATE_FWD = join_regexes([RE_EN_DAY, r"( of)? ", RE_EN_MONTHS, r"\w*(( |, )(\d{4}))?"])
RE_EN_DATE_BWD = join_regexes([RE_EN_MONTHS, r"\w* ", RE_EN_DAY, RE_EN_YEAR])
RE_EN_TIME_DELTA = r"\d+\.?\d? ?(days?|months?|years?|hours?|minutes?|weeks?|mins?|hrs?)"


def rectify_typos(string: str) -> str:
    """Fix typos in text regarding delimiters and lowercase start of sentence."""
    replacements = [
        (RE_TYPO_BASE_DELIMITERS_AFTER_SPACE, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_BASE_DELIMITERS_SQUEEZED, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_BASE_DELIMITERS_CENTERED, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_MULTIPLE_SPACE, RE_FIX_SINGLE_SPACE),
        (RE_TYPO_LOWERCASE_START_OF_SENTENCE, RE_FIX_UPPERCASE_START_OF_SENTENCE),
        (RE_MULTIPLE_EMOTION_DELIMITERS, RE_FIX_SINGLE_EMOTION_DELIMITER),
    ]
    for typo, fix in replacements:
        string = regex.sub(typo, fix, string)
    return string


def mask_ccy(string: str, token="[CCY]") -> str:
    """Replace currency expressions with a special token to reduce vocab size."""
    masked_ccy = regex.sub(RE_EN_CCY_SIGN, token, string)
    return regex.sub(RE_EN_CCY_NAME, token, masked_ccy)


def mask_date(string: str, token="[DATE]") -> str:
    masked_date = regex.sub(RE_EN_DATE_FWD, token, string)
    return regex.sub(RE_EN_DATE_BWD, token, masked_date)


def mask_time(string: str, token="[TIME]") -> str:
    """Replace time expressions in text with a special token to reduce vocab size."""
    return regex.sub(RE_EN_TIME, token, string)


def mask_timedelta(string: str, token="[PERIOD]") -> str:
    return regex.sub(RE_EN_TIME_DELTA, token, string)


def mask_number(string: str, token="[NUM]") -> str:
    return regex.sub(RE_NUMBER, token, string)


def mask_non_words(string: str) -> str:
    return mask_number(mask_date(mask_timedelta(mask_time(mask_ccy(string)))))
