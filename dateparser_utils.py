# plan to use "humanize" or "arrow" package instead

from typing import Iterable
import dateparser

from parse import parse, Result
from datetime import datetime
from beartype import beartype


@beartype
def convert_parse_result_to_datetime_obj(parsed: Result):
    datetime_obj = datetime(
        year=parsed["year"],
        month=parsed["month"],
        day=parsed["day"],
        hour=parsed["hour"],
        minute=parsed["minute"],
        second=parsed["second"],
    )
    return datetime_obj


@beartype
def parse_date_with_single_format(input_format: str, input_date_string: str):
    # Parse the input string using the specified format
    parsed = parse(input_format, input_date_string)

    if isinstance(parsed, Result):
        # Reconstruct the ISO-formatted date and time string
        return convert_parse_result_to_datetime_obj(parsed)


@beartype
def parse_date_with_multiple_formats(
    custom_formats: Iterable[str], it: str, settings: dict = {"STRICT_PARSING": True}
):
    for fmt in custom_formats:
        result = parse_date_with_single_format(fmt, it)
        if result:
            return result
    result = dateparser.parse(it, settings=settings) # type: ignore
    return result

def format_datetime(dt: datetime, fmt: str):
    return dt.strftime(fmt)

def render_datetime_as_hexo_format(dt:datetime):
    return format_datetime(dt, "%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    candidates = [
        "2023-09-12T15:17:04.131Z",
        "2022-07-10T00:16:40+08:00",
        "2022-11-28-22-01-29",  # problematic.
        # "2022-11-28T22:01:29",
        "Windows 10 system debloating, windows operating system optimization, winget, windows commandline package manager",
    ]

    custom_date_formats = ["{year:d}-{month:d}-{day:d}-{hour:d}-{minute:d}-{second:d}"]

    for it in candidates:
        print("parsing:", it)
        result = parse_date_with_multiple_formats(custom_date_formats, it)
        if result is None:
            report = "Could not parse: " + it
        else:
            parse_result = render_datetime_as_hexo_format(result)
            report = "Parsed: " + parse_result
        print(report)
        print("-" * 30)
