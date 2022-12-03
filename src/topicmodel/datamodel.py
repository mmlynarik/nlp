from dataclasses import dataclass
from datetime import date


@dataclass
class DateSpan:
    date_from: date
    date_to: date
