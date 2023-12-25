import sys

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)
from functools import lru_cache
import os
from headline_match import parse_content_metadata, modify_content_metadata, remove_metadata
from beartype import beartype
import pydantic
import inspect
import uuid
from typing import Callable, Iterable, TypeVar, Union

import datetime
import re

from cache_db_context import (
    SourceIteratorAndTargetGeneratorParam,
    TargetGeneratorParameter,
    iterate_source_dir_and_generate_to_target_dir,
)

from custom_doc_writer import (
    llm_context,
    process_content_and_write_result,
    assemble_prompt_components,
)

from dateparser_utils import (
    parse_date_with_multiple_formats,
    render_datetime_as_hexo_format,
)
from similarity_utils import SimilarityIndex

UTF8 = "utf-8"

T = TypeVar("T")

required_fields = ("tags", "title", "description", "category", "date")
fields_that_need_summary_to_generate = ("tags", "title", "description", "category")
date_mitigation_fields = ("created",)

custom_date_formats = ("{year:d}-{month:d}-{day:d}-{hour:d}-{minute:d}-{second:d}",)


def generate_markdown_name():
    file_id = str(uuid.uuid4())
    fname = f"{file_id}.md"
    return fname


def load_file(fname: str):
    with open(fname, "r", encoding=UTF8) as f:
        cnt = f.read()
    return cnt

def write_file(fname: str, content:str):
    with open(fname, "w+", encoding=UTF8) as f:
        f.write(content)


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
    with llm_context(init_prompt) as model:
        ret = model.run(prompt)
        return ret


@beartype
def call_llm_once_and_parse(init_prompt: str, prompt: str, pydantic_type: type[T]) -> T:
    response = call_llm_once(init_prompt, prompt)
    ret = pydantic_type.parse_raw(response)  # type: ignore
    return ret


@lru_cache(maxsize=20)
def cached_getsource(obj):
    source = inspect.getsource(obj)
    return source


@beartype
def generate_init_prompt_with_schema(identity: str, task: str, pydantic_schema):
    schema_str = cached_getsource(pydantic_schema)
    init_prompt = f"""{identity.strip()}
{task.strip()}

Respond strictly in following pydantic schema:
```python
{schema_str.strip()}
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


def generate_description_recommended_init_script():
    return generate_item_recommended_init_prompt("description", Description)


def generate_title_recommended_init_script():
    return generate_item_recommended_init_prompt("title", Title)


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


@beartype
def generate_title(summary: str):
    response = generate_recommended_items(
        summary, generate_title_recommended_init_script
    )
    return response.title


@beartype
def generate_description(summary: str):
    response = generate_recommended_items(
        summary, generate_description_recommended_init_script
    )
    return response.description


@beartype
def generate_summary_prompt_base(word_limit: int):
    init_prompt = f"""You are reading text from file in chunks. You would understand what the text is about and return brief summary (under {word_limit} words)."""
    return init_prompt


@beartype
def generate_previous_comment_component(previous_comment: str = ""):
    comp = (
        f"""Previous comment:
{previous_comment}"""
        if previous_comment.strip()
        else ""
    )
    return comp


@beartype
def generate_content_component(content: str, programming_language=""):
    comp = f"""Content:
```{programming_language}
{content}
```"""
    return comp


@beartype
def generate_summary_prompt_generator(programming_language: str):
    @beartype
    def prompt_generator(content: str, location: str, previous_comment: str = ""):
        components = [
            generate_content_component(content, programming_language),
            generate_previous_comment_component(previous_comment),
        ]
        ret = assemble_prompt_components(components)
        return ret

    return prompt_generator


@beartype
def generate_summary(
    content_without_metadata: str,
    filename: str = "<unknown>",
    word_limit: int = 30,
    programming_language="markdown",
):
    prompt_base = generate_summary_prompt_base(word_limit)
    prompt_generator = generate_summary_prompt_generator(programming_language)

    with llm_context(prompt_base) as model:
        ret = process_content_and_write_result(
            model, prompt_generator, filename, content_without_metadata
        )
        return ret["summary"]


@beartype
def get_date_obj_by_file_ctime(filepath: str):
    creation_timestamp = os.path.getctime(filepath)
    date_obj = datetime.datetime.fromtimestamp(creation_timestamp)
    return date_obj


@beartype
def get_filename_without_extension(filepath: str):
    base_filepath = os.path.basename(filepath)
    filename_without_extension = re.sub(r"\.\w+$", "", base_filepath)
    return filename_without_extension


@beartype
def get_date_obj_by_metadata(metadata: dict):
    for field in date_mitigation_fields:
        if field in metadata:
            date_obj = parse_date_with_multiple_formats(custom_date_formats, field)
            if date_obj:
                return date_obj


@beartype
def get_date_obj_by_filepath(filepath: str):
    filename_without_extension = get_filename_without_extension(filepath)
    date_obj = parse_date_with_multiple_formats(
        custom_date_formats, filename_without_extension
    )
    return date_obj


@beartype
def generate_date_obj(filepath: str, metadata: dict):
    maybe_methods = (
        lambda: get_date_obj_by_metadata(metadata),
        lambda: get_date_obj_by_filepath(filepath),
    )
    fallback = lambda: get_date_obj_by_file_ctime(filepath)
    return maybe_with_fallback(maybe_methods, fallback)


@beartype
def generate_date(filepath: str, metadata: dict):
    date_obj = generate_date_obj(filepath, metadata)
    ret = render_datetime_as_hexo_format(date_obj)
    return ret


@beartype
def maybe_with_fallback(
    maybe_methods: Iterable[Callable[[], Union[T, None]]], fallback: Callable[[], T]
) -> T:
    for it in maybe_methods:
        obj = it()
        if obj is not None:
            return obj
    return fallback()


@beartype
def build_field_generation_methods_with_summary(
    content_without_metadata: str,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    tag_top_k: int = 3,
    category_top_k: int = 3,
    summary_word_limit: int = 30,
    programming_language: str = "markdown",
):
    summary = generate_summary(
        content_without_metadata,
        word_limit=summary_word_limit,
        programming_language=programming_language,
    )

    data = {
        "tags": lambda: generate_tags(tags_similarity_index, summary, top_k=tag_top_k),
        "title": lambda: generate_title(summary),
        "description": lambda: generate_description(summary),
        "category": lambda: generate_category(
            categories_similarity_index, summary, top_k=category_top_k
        ),
    }
    return data


@beartype
def find_missing_fields_and_build_field_to_method(
    filepath: str,
    content_without_metadata: str,
    metadata: dict,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    tag_top_k: int = 3,
    category_top_k: int = 3,
    summary_word_limit: int = 30,
    programming_language: str = "markdown",
):
    field_to_method = {
        "date": lambda: generate_date(filepath, metadata),
    }
    missing_fields = [
        field for field in required_fields if field not in new_metadata.keys()
    ]
    if set(missing_fields).intersection(fields_that_need_summary_to_generate):
        field_to_method_with_summary = build_field_generation_methods_with_summary(
            content_without_metadata,
            tags_similarity_index,
            categories_similarity_index,
            tag_top_k=tag_top_k,
            category_top_k=category_top_k,
            summary_word_limit=summary_word_limit,
            programming_language=programming_language,
        )
        field_to_method.update(field_to_method_with_summary)
    return missing_fields, field_to_method


@beartype
def get_additional_metadata(
    missing_fields: Iterable[str], field_to_method: dict[str, Callable[[], str]]
):
    additional_metadata = {}
    for field in missing_fields:
        additional_metadata[field] = field_to_method[field]()
    return additional_metadata


@beartype
def generate_new_metadata(metadata: dict, additional_metadata: dict):
    changed = False
    new_metadata = metadata.copy()
    if additional_metadata != {}:
        changed = True
        new_metadata.update(additional_metadata)
    return new_metadata, changed


@beartype
def generate_content_metadata(
    filepath: str,
    content_without_metadata: str,
    metadata: dict,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    tag_top_k: int = 3,
    category_top_k: int = 3,
    summary_word_limit: int = 30,
    programming_language: str = "markdown",
):
    missing_fields, field_to_method = find_missing_fields_and_build_field_to_method(
        filepath,
        content_without_metadata,
        metadata,
        tags_similarity_index,
        categories_similarity_index,
        tag_top_k=tag_top_k,
        category_top_k=category_top_k,
        summary_word_limit=summary_word_limit,
        programming_language=programming_language,
    )
    additional_metadata = get_additional_metadata(missing_fields, field_to_method)
    new_metadata, changed = generate_new_metadata(metadata, additional_metadata)
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


@beartype
def extract_and_update_existing_tags_and_categories(
    content: str, existing_tags: set[str], existing_categories: set[str]
):
    has_metadata, metadata, _ = parse_content_metadata(content)
    if has_metadata:
        update_tags_and_categories_from_metadata(
            metadata, existing_tags, existing_categories
        )


@beartype
def append_note_path_without_bad_words_and_update_existing_tags_and_categories(
    content: str,
    bad_words: list[str],
    note_paths: list[str],
    filepath: str,
    existing_tags: set[str],
    existing_categories: set[str],
):
    if not check_if_contains_bad_words(content, bad_words):
        note_paths.append(filepath)
        extract_and_update_existing_tags_and_categories(
            content, existing_tags, existing_categories
        )


@beartype
def get_note_paths_without_bad_words_and_existing_tags_and_categories(
    notes_dir: str, bad_words: list[str]
):
    note_paths: list[str] = []
    existing_tags: set[str] = set()
    existing_categories: set[str] = set()

    for fpath in iterate_and_get_markdown_filepath_from_notedir(notes_dir):
        content = load_file(fpath)
        append_note_path_without_bad_words_and_update_existing_tags_and_categories(
            content, bad_words, note_paths, fpath, existing_tags, existing_categories
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

(
    note_paths,
    existing_tags,
    existing_categories,
) = get_note_paths_without_bad_words_and_existing_tags_and_categories(
    notes_dir, bad_words
)

tags_similarity_index = SimilarityIndex(sent_trans, existing_tags)
categories_similarity_index = SimilarityIndex(sent_trans, existing_categories)

@beartype
def ...(source_path:str, target_path:str):
    content = load_file(source_path)
    has_metadata, metadata, content_without_metadata = parse_content_metadata(
        content
    )
    new_metadata, changed = generate_content_metadata(
        source_path, content_without_metadata, metadata, tags_similarity_index, categories_similarity_index
    )
    if changed:
        new_content = modify_content_metadata(content, has_metadata, new_metadata)
    else:
        new_content = content
    write_file(target_path, new_content)


@beartype
def ...(param:TargetGeneratorParameter):
    basename = generate_markdown_name()
    ret = os.path.join(param.target_dir_path, basename)
    return ret

if __name__ == "__main__":
    cache_dir = "cache"
    final_dir = "source/_posts"  # clear and put transformed notes to final dir
    notes_source_dir = "notes"

    db_path = "cache_db.json"
    bad_words_path = "bad_words.txt"

    param = SourceIteratorAndTargetGeneratorParam(
        source_dir_path=notes_dir, target_dir_path=cache_dir, db_path=db_path
    )

    iterate_source_dir_and_generate_to_target_dir(
        param, source_walker=..., target_path_generator=..., target_file_geneator=...
    )
