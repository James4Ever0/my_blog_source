# usually a few lines away from metadata. just find it.

# no you should remove those lines that most containing the title.
from beartype import beartype
from typing import NewType, cast

from enum import Enum, auto

METADATA_SPLITER = "---"
CODEBLOCK_MARKER = "```"
FORM_MARKER = "|"
HEADLINE_MARKER = "# "
NEWLINE = "\n"

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


class ReaderState(Enum):
    init = auto()
    within_metadata = auto()
    content = auto()
    within_codeblock = auto()
    within_form = auto()

@beartype
def join_lines_with_state(lines: list[str]) -> str:
    new_lines = []
    state = ReaderState.init
    for line in lines:
        it = line.strip()
        if state == ReaderState.init:
            if it == METADATA_SPLITER:
                state = ReaderState.within_metadata
        elif state == ReaderState.within_metadata:
            if it == METADATA_SPLITER:
                state = ReaderState.content
        elif state == ReaderState.content:
            if it.startswith(CODEBLOCK_MARKER):
                state = ReaderState.within_codeblock
            elif it.startswith(FORM_MARKER):
                state = ReaderState.within_form
            if state != ReaderState.content:
                new_lines.append(NEWLINE)
        elif state == ReaderState.within_codeblock:
            if it.startswith(CODEBLOCK_MARKER):
                state = ReaderState.content
        elif state == ReaderState.within_form:
            if not it.startswith(FORM_MARKER):
                state = ReaderState.content
        if state == ReaderState.content:
            new_lines.append(NEWLINE)
        new_lines.append(line)
        new_lines.append(NEWLINE)
    return "".join(new_lines)
