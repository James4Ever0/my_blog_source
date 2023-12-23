import os
from headline_match import parse_content_metadata, modify_content_metadata
import sys
from beartype import beartype

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)

from cache_db_context import (
    SourceIteratorAndTargetGeneratorParam,
    TargetGeneratorParameter,
    iterate_source_dir_and_generate_to_target_dir,
)

from dateparser_utils import parse_date_with_multiple_formats
 
UTF8 = "utf-8"


def load_file(fname: str):
    with open(fname, "r", encoding=UTF8) as f:
        cnt = f.read()
    return cnt


def split_by_line(cnt: str):
    myitems = cnt.split("\n")
    myitems = [myit.strip() for myit in myitems]
    myitems = [myit for myit in myitems if len(myit) > 0]
    return myitems


def load_bad_words(fname: str):
    cnt = load_file(fname)
    mybadwords = split_by_line(cnt)
    return mybadwords

# should change the schema according to the need, only generate what is needed the most.

@beartype
def generate_tags(existing_tags:list[str], summary:str):
    ...


# if filename is not date, use as title
# if filename is date, perform title generation
@beartype
def generate_title(summary:str):
    ...

@beartype
def generate_description(summary:str):
    ...

@beartype
def generate_category(existing_categories:list[str], summary:str):
    ...

required_fields = ["tags", "title", "description", "category"]
mitigation_map = {"created": "date"}

# need to make sure data format is correct.
# need to parse different "created" data format.

@beartype
def generate_content_metadata(content: str, metadata: dict):
    changed = False
    new_metadata = metadata.copy()
    return new_metadata, changed
@beartype
def check_if_contains_bad_words(content: str, bad_words: list[str]):
    for word in bad_words:
        if word in content:
            return True
    return False

# use two hashs for cache varification
# store filename before and after processing
# one before processing, one after processing
# store processed ones to some place for cacheing

# you need to collect existing tags and categories before processing

# collect only from file without bad words.

def check_if_has_markdown_file_extension(filename:str):
    return filename.endswith(".md")

def iterate_and_get_markdown_filepath_from_notedir(notes_dir:str):
    for filename in os.listdir(notes_dir):
        if check_if_has_markdown_file_extension(filename):
            source_path = os.path.join(notes_dir, filename)
            yield source_path

def get_note_paths_without_bad_words_and_existing_tags_and_categories(notes_dir:str, bad_words:list[str]):

    note_paths = []
    existing_tags = set()
    existing_categories = set()

    for fpath in iterate_and_get_markdown_filepath_from_notedir(notes_dir):
        content = load_file(fpath)
        if not check_if_contains_bad_words(content, bad_words):
            note_paths.append(fpath)
            has_metadata, metadata = parse_content_metadata(
                content
            )
            if has_metadata:
                update_tags_and_categories_from_metadata(metadata, existing_tags, existing_categories)
    return note_paths, existing_tags, existing_categories


def update_tags_set_from_metadata(metadata:dict, tags_set:set[str]):
    for tag in metadata.get("tags", []):
        tags_set.add(tag)

def update_categories_set_from_metadata(metadata:dict, categories_set:set[str]):
    category = metadata.get("category", None)
    if category:
        categories_set.add(category)

def update_tags_and_categories_from_metadata(metadata:dict, tags_set:set[str], categories_set:set[str]):
    update_tags_set_from_metadata(metadata, tags_set)
    update_categories_set_from_metadata(metadata, categories_set)

bad_words = load_bad_words(bad_words_path)

for filename in os.listdir(notes_dir):
    if filename.endswith(".md"):
        source_path = os.path.join(notes_dir, filename)
        print("processing file:", filename)
        content = load_file(source_path)
        if check_if_contains_bad_words(content, bad_words):
            print("bad words detected, skipping file")
            continue
        has_metadata, metadata = parse_content_metadata(
            content
        )  # does it have metadata?
        new_metadata, changed = generate_content_metadata(content, metadata)
        if changed:
            new_content = modify_content_metadata(content, has_metadata, new_metadata)
        else:
            # do nothing about it.
            ...

        # instance_list = filter_setextheading_in_children(md_obj)
        # first_instance = instance_list[0]
        # for child in first_instance.children if ['modified' in children]
        # assert instance_count == 1, "file %s has more than one instance (%d)" % (filename, instance_count)
        # chores to do: tagging, categorization
        # filter unwanted content: passwords, bad articles

if __name__ == "__main__":

    cache_dir = "cache"
    final_dir = "source/_posts" # clear and put transformed notes to final dir
    notes_source_dir = "notes"

    db_path = "cache_db.json"
    bad_words_path = "bad_words.txt"

    param = SourceIteratorAndTargetGeneratorParam(
        source_dir_path=notes_dir, target_dir_path=cache_dir, db_path=db_path
    )