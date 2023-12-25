import re
import yaml
from beartype import beartype

# Multiline regular expression to match the pattern
metadata_pattern = r"""
---     # Match the opening ---
(.*?)     # Match any characters (including newlines) zero or more times
---     # Match the closing ---
"""

# Compile the regular expression
metadata_regex = re.compile(metadata_pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)


@beartype
def parse_content_metadata(markdown_content: str):
    # Match the pattern in the Markdown content
    matches = metadata_regex.findall(markdown_content)
    has_metadata = len(matches) > 0
    if has_metadata:
        first_match = matches[0]
        metadata = yaml.safe_load(first_match)
        content_without_metadata = remove_metadata(markdown_content)
    else:
        metadata = {}
        content_without_metadata = markdown_content

    return has_metadata, metadata, content_without_metadata


@beartype
def replace_metadata(source: str, replace_str: str, count=1):
    result, _ = metadata_regex.subn(replace_str, source, count=count)
    return result


@beartype
def remove_metadata(source: str):
    result = replace_metadata(source, "")
    return result


@beartype
def modify_content_metadata(markdown_content: str, has_metadata: bool, metadata: dict):
    replaced_metadata_str = yaml.safe_dump(metadata).strip()
    replaced_metadata_str = f"""---
{replaced_metadata_str}
---"""
    if has_metadata:
        result = replace_metadata(markdown_content, replaced_metadata_str)
    else:
        result = "\n".join([replaced_metadata_str, markdown_content])
    return result


def test_main():
    # Sample Markdown content
    markdown_content = """
content
---
title: Sample Title
tags: [Tag1, Tag2]
---
content
content

---
title: Sample Title
tags: Tag1, Tag3
---
"""
    has_metadata, _, content_without_metadata = parse_content_metadata(markdown_content)
    updated_content = modify_content_metadata(
        markdown_content, has_metadata, {"new_title": "Sample Title"}
    )
    print(updated_content)
    print("-" * 20)
    print(content_without_metadata)


if __name__ == "__main__":
    test_main()
