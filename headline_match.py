import re
import yaml

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

# Multiline regular expression to match the pattern
pattern = r"""
---     # Match the opening ---
(.*?)     # Match any characters (including newlines) zero or more times
---     # Match the closing ---
"""

# Compile the regular expression
regex = re.compile(pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)

# Match the pattern in the Markdown content
matches = regex.findall(markdown_content)
# Output the matches
print(matches)  # we only want the first match.

first_match = matches[0]  # we will process this thing.
metadata = yaml.safe_load(first_match)
# first_match = first_match.strip()

result, _ = regex.subn("[replaced]", markdown_content, count=1)
print(result)
print(metadata)  # loaded!

if __name__ == "__main__":
    ...