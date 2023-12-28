# usually a few lines away from metadata. just find it.

# no you should remove those lines that most containing the title.
from beartype import beartype
from typing import NewType, cast

# from enum import Enum, auto

# METADATA_SPLITER = "---"
HEADLINE_MARKER = "# "

StringWithoutHeadlineMarker = NewType('StringWithoutHeadlineMarker', str)

@beartype
def strip_headline_marker(line: str) -> StringWithoutHeadlineMarker:
    return cast(StringWithoutHeadlineMarker, line.strip(HEADLINE_MARKER))


@beartype
def is_empty_string(line: str) -> bool:
    return line.strip() == ""


@beartype
def remove_headline_from_lines(lines: list[str], title: str) -> list[str]:
    @beartype
    def is_headline_line(line: str, stripped_title: StringWithoutHeadlineMarker):
        stripped_line = strip_headline_marker(line)
        if not is_empty_string(stripped_line):
            if stripped_line == stripped_title:
                return True
        return False

    @beartype
    def process_lines(stripped_title: StringWithoutHeadlineMarker):
        new_lines = []
        for line in lines:
            if is_headline_line(line, stripped_title):
                continue
            new_lines.append(line)
        return new_lines

    def strip_title_and_process_lines():
        stripped_title = strip_headline_marker(title)
        if is_empty_string(stripped_title):
            return lines
        return process_lines(stripped_title)

    return strip_title_and_process_lines()


# class ReaderState(Enum):
#     init = auto()
#     within_metadata = auto()
#     out_of_metadata = auto()
#     headline_found = auto()
#     headline_not_found = auto()


# @beartype
# def remove_headline_from_lines(lines: list[str]) -> list[str]:
#     new_lines = []
#     state = ReaderState.init
#     for it in lines:
#         if state == ReaderState.init:
#             if it == METADATA_SPLITER:
#                 state = ReaderState.within_metadata
#         elif state == ReaderState.within_metadata:
#             if it == METADATA_SPLITER:
#                 state = ReaderState.out_of_metadata
#         elif state == ReaderState.out_of_metadata:
#             if it.startswith(HEADLINE_MARKER):
#                 state = ReaderState.headline_found
#                 continue
#             elif it.strip():
#                 state = ReaderState.headline_not_found
#         new_lines.append(it)
#     return new_lines
