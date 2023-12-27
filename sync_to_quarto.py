source_dir = "source/_posts"
target_dir = "quarto_blog/myblog/posts"

import os
import shutil
import yaml
import sys
from headline_match import parse_content_metadata
from beartype import beartype
from io_utils import load_file, write_file


@beartype
def fix_metadata_line_wrap_in_content(content: str):
    (
        has_metadata,
        metadata,
        content_without_metadata,
        first_match,
    ) = parse_content_metadata(content)
    if has_metadata:
        if metadata is not None:
            for k in ["created", "modified"]:
                if k in metadata.keys():
                    del metadata[k]
            keys = list(metadata.keys())
            for k in keys:
                v = metadata[k]
                if isinstance(v, str):
                    metadata[k] = v.replace("`", "&grave;")
                elif isinstance(v, list):
                    metadata[k] = [
                        x if not isinstance(x, str) else x.replace("`", "&grave;")
                        for x in v
                    ]
            metadata["categories"] = metadata.get("tags", [])
            repl = yaml.safe_dump(
                metadata,
                width=sys.maxsize,
                default_style='"',
                default_flow_style=True,
                allow_unicode=True,
            )
            new_content = (
                "---\n"
                + repl
                + "\n---\n"
                + ("\n" + content_without_metadata + "\n").replace(
                    "\n---\n", "\n------\n"
                )
            )
            return new_content
    return content


# you need to convert multiline yaml into one

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.mkdir(target_dir)

for fname in os.listdir(source_dir):
    source_path = os.path.join(source_dir, fname)
    content = load_file(source_path)
    new_content = fix_metadata_line_wrap_in_content(content)
    identifier = fname.split(".")[0]
    target_subdir = os.path.join(target_dir, identifier)
    os.mkdir(target_subdir)
    target_path = os.path.join(target_subdir, "index.qmd")
    write_file(target_path, new_content)
