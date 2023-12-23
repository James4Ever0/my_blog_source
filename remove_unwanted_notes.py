import os
from headline_match import parse_content_metadata, modify_content_metadata
import sys
from beartype import beartype
import pydantic
import inspect
import uuid
from typing import Callable, TypeVar

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)

from cache_db_context import (
    SourceIteratorAndTargetGeneratorParam,
    TargetGeneratorParameter,
    iterate_source_dir_and_generate_to_target_dir,
)

from custom_doc_writer import LLM, process_code_and_write_result

from dateparser_utils import parse_date_with_multiple_formats
from similarity_utils import SimilarityIndex

UTF8 = "utf-8"

T = TypeVar("T")


def generate_markdown_name():
    file_id = str(uuid.uuid4())
    fname = f"{file_id}.md"
    return fname


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


# if filename is not date, use as title
# if filename is date, perform title generation
@beartype
def generate_title(summary: str):
    ...


@beartype
def generate_description(summary: str):
    ...


class Categories(pydantic.BaseModel):
    categories: list[str]


class Tags(pydantic.BaseModel):
    tags: list[str]


class Category(pydantic.BaseModel):
    category: str


class Title(pydantic.BaseModel):
    title: str


class Description(pydantic.BaseModel):
    description: str


@beartype
def call_llm_once(init_prompt: str, prompt: str) -> str:
    model = LLM(init_prompt)
    ret = model.run(prompt)
    del model
    return ret


@beartype
def call_llm_once_and_parse(init_prompt: str, prompt: str, pydantic_type: type[T]) -> T:
    response = call_llm_once(init_prompt, prompt)
    ret = pydantic_type.parse_raw(response)  # type: ignore
    return ret


@beartype
def generate_init_prompt_with_schema(identity: str, task: str, pydantic_schema):
    schema_str = inspect.getsource(pydantic_schema).strip()
    init_prompt = f"""{identity}
{task}

Respond strictly in following pydantic schema:
```python
{schema_str}
```
"""
    return init_prompt


@beartype
def generate_blogger_init_prompt_with_schema(task: str, schema_class: type):
    identity = "You are a professional blogger."
    init_prompt = generate_init_prompt_with_schema(identity, task, schema_class)
    return init_prompt


def generate_item_recommended_init_prompt(item_name: str, schema_class: type[T]):
    task = f"""You will be given an article summary.
You will produce recommended {item_name}."""
    init_prompt = generate_blogger_init_prompt_with_schema(task, schema_class)
    return init_prompt, schema_class


def generate_category_recommender_init_prompt():
    return generate_item_recommended_init_prompt("categories", Categories)


def generate_tag_recommender_init_prompt():
    return generate_item_recommended_init_prompt("tags", Tags)


@beartype
def generate_item_chooser_init_prompt(
    item_name: str, objective: str, schema_class: type[T]
):
    task = f"""You will be given an article summary, similar {item_name} in database, and your recommended {item_name}.
You would prefer {item_name} in database if they match the summary.
You will produce {objective} that best matches the summary."""
    init_prompt = generate_blogger_init_prompt_with_schema(task, schema_class)
    return init_prompt, schema_class


def generate_category_chooser_init_prompt():
    return generate_item_chooser_init_prompt(
        "categories", "a single category", Category
    )


def generate_tag_chooser_init_prompt():
    return generate_item_chooser_init_prompt("tags", "tags", Tags)


def generate_prompt_context_from_prompt_context_dict(
    prompt_context_dict: dict[str, str]
):
    prompt_context = ""
    for k, v in prompt_context_dict.items():
        prompt_context += f"{k.strip().title()}:\n{v.strip()}\n"
    return prompt_context.strip()


def generate_json_prompt(prompt_context_dict: dict[str, str]):
    prompt_context = generate_prompt_context_from_prompt_context_dict(
        prompt_context_dict
    )
    prompt = f"""{prompt_context}

Response in JSON format:
"""
    return prompt


def generate_summary_prompt_context_dict(summary: str):
    return {"summary": summary}


def generate_json_prompt_with_summary(summary: str):
    prompt_context_dict = generate_summary_prompt_context_dict(summary)
    ret = generate_json_prompt(prompt_context_dict)
    return ret


@beartype
def generate_similar_and_recommended_items_prompt_context_dict(
    items_name: str, similar_items: list[str], recommended_items: list[str]
):
    ret = {
        f"similar {items_name} in database": str(similar_items),
        f"your recommended {items_name}": str(recommended_items),
    }
    return ret


@beartype
def generate_items_chooser_json_prompt(
    items_name: str,
    summary: str,
    similar_items: list[str],
    recommended_items: list[str],
):
    prompt_context_dict = generate_summary_prompt_context_dict(summary)
    prompt_context_dict.update(
        generate_similar_and_recommended_items_prompt_context_dict(
            items_name, similar_items, recommended_items
        )
    )
    ret = generate_json_prompt(prompt_context_dict)
    return ret


@beartype
def generate_categories_chooser_json_prompt(
    summary: str, similar_categories: list[str], recommended_categories: list[str]
):
    items_name = "categories"
    return generate_items_chooser_json_prompt(
        items_name, summary, similar_categories, recommended_categories
    )


@beartype
def generate_tags_chooser_json_prompt(
    summary: str, similar_tags: list[str], recommended_tags: list[str]
):
    items_name = "tags"
    return generate_items_chooser_json_prompt(
        items_name, summary, similar_tags, recommended_tags
    )


@beartype
def generate_recommended_items(
    summary: str, init_prompt_generator: Callable[[], tuple[str, type[T]]]
) -> T:
    init_prompt, data_class = init_prompt_generator()
    prompt = generate_json_prompt_with_summary(summary)
    response = call_llm_once_and_parse(init_prompt, prompt, data_class)
    return response


@beartype
def generate_chosen_item(
    summary: str,
    recommended_items: list[str],
    similar_items: list[str],
    init_prompt_generator: Callable[[], tuple[str, type[T]]],
) -> T:
    init_prompt, data_class = init_prompt_generator()
    prompt = generate_categories_chooser_json_prompt(
        summary, similar_items, recommended_items
    )
    response = call_llm_once_and_parse(init_prompt, prompt, data_class)
    return response


@beartype
def generate_recommended_categories(summary: str):
    response = generate_recommended_items(
        summary, generate_category_recommender_init_prompt
    )
    return response.categories


@beartype
def generate_chosen_category(
    summary: str, recommended_categories: list[str], similar_categories: list[str]
):
    response = generate_chosen_item(
        summary,
        recommended_categories,
        similar_categories,
        generate_category_chooser_init_prompt,
    )
    return response.category


@beartype
def generate_recommended_tags(summary: str):
    response = generate_recommended_items(summary, generate_tag_recommender_init_prompt)
    return response.tags


@beartype
def generate_chosen_tags(
    summary: str, recommended_tags: list[str], similar_tags: list[str]
):
    response = generate_chosen_item(
        summary,
        recommended_tags,
        similar_tags,
        generate_tag_chooser_init_prompt,
    )
    return response.tags


@beartype
def generate_item(
    items_similarity_index: SimilarityIndex,
    summary: str,
    item_recommender: Callable[[str], list[str]],
    item_chooser: Callable[[str, list[str], list[str]], T],
    top_k: int = 3,
) -> T:
    recommended_items = item_recommender(summary)
    similar_items = items_similarity_index.search(recommended_items, top_k=top_k)
    ret = item_chooser(summary, recommended_items, similar_items)
    return ret


@beartype
def generate_category(
    categories_similarity_index: SimilarityIndex, summary: str, top_k: int = 3
):
    ret = generate_item(
        categories_similarity_index,
        summary,
        generate_recommended_categories,
        generate_chosen_category,
        top_k=top_k,
    )
    return ret


@beartype
def generate_tags(tags_similarity_index: SimilarityIndex, summary: str, top_k: int = 3):
    ret = generate_item(
        tags_similarity_index,
        summary,
        generate_recommended_tags,
        generate_chosen_tags,
        top_k=top_k,
    )
    return ret


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


def check_if_has_markdown_file_extension(filename: str):
    return filename.endswith(".md")


def iterate_and_get_markdown_filepath_from_notedir(notes_dir: str):
    for filename in os.listdir(notes_dir):
        if check_if_has_markdown_file_extension(filename):
            source_path = os.path.join(notes_dir, filename)
            yield source_path


def get_note_paths_without_bad_words_and_existing_tags_and_categories(
    notes_dir: str, bad_words: list[str]
):
    note_paths: list[str] = []
    existing_tags: set[str] = set()
    existing_categories: set[str] = set()

    for fpath in iterate_and_get_markdown_filepath_from_notedir(notes_dir):
        content = load_file(fpath)
        if not check_if_contains_bad_words(content, bad_words):
            note_paths.append(fpath)
            has_metadata, metadata = parse_content_metadata(content)
            if has_metadata:
                update_tags_and_categories_from_metadata(
                    metadata, existing_tags, existing_categories
                )
    return note_paths, existing_tags, existing_categories


def update_tags_set_from_metadata(metadata: dict, tags_set: set[str]):
    for tag in metadata.get("tags", []):
        tags_set.add(tag)


def update_categories_set_from_metadata(metadata: dict, categories_set: set[str]):
    category = metadata.get("category", None)
    if category:
        categories_set.add(category)


def update_tags_and_categories_from_metadata(
    metadata: dict, tags_set: set[str], categories_set: set[str]
):
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
    final_dir = "source/_posts"  # clear and put transformed notes to final dir
    notes_source_dir = "notes"

    db_path = "cache_db.json"
    bad_words_path = "bad_words.txt"

    param = SourceIteratorAndTargetGeneratorParam(
        source_dir_path=notes_dir, target_dir_path=cache_dir, db_path=db_path
    )
