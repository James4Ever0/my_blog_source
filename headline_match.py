import re
import yaml
import beartype

# Multiline regular expression to match the pattern
metadata_pattern = r"""
---     # Match the opening ---
(.*?)     # Match any characters (including newlines) zero or more times
---     # Match the closing ---
"""

# Compile the regular expression
metadata_regex = re.compile(metadata_pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)

@beartype.beartype
def parse_content_metadata(markdown_content: str):
    # Match the pattern in the Markdown content
    matches = metadata_regex.findall(markdown_content)
    has_metadata = len(matches) > 0
    if has_metadata:
        first_match = matches[0]
        metadata = yaml.safe_load(first_match)
    else:
        metadata = {}

    return has_metadata, metadata


@beartype.beartype
def modify_content_metadata(markdown_content: str, has_metadata: bool, metadata: dict):
    replaced_metadata_str = yaml.safe_dump(metadata).strip()
    replaced_metadata_str = f"""---
{replaced_metadata_str}
---"""
    if has_metadata:
        result, _ = metadata_regex.subn(
            replaced_metadata_str, markdown_content, count=1
        )
    else:
        result = "\n".join([replaced_metadata_str, markdown_content])
    return result


if __name__ == "__main__":
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
    has_metadata, metadata = parse_content_metadata(markdown_content)
    updated_content = modify_content_metadata(
        markdown_content, has_metadata, {"new_title": "Sample Title"}
    )
    print(updated_content)
