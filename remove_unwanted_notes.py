import argparse
import shutil
import sys
from sentence_transformers import SentenceTransformer
import traceback

from remove_headline_from_markdown import remove_headline_from_lines,join_lines_with_state, NEWLINE

sys.path.append(
    "/media/root/Toshiba XG3/works/prometheous/document_agi_computer_control"
)
import datetime
import inspect
import os
import re
import uuid
from functools import lru_cache
from typing import Callable, Iterable, Optional, TypeVar, Union

import pydantic
from beartype import beartype
from cache_db_context import (  # type:ignore
    SourceIteratorAndTargetGeneratorParam,
    TargetGeneratorParameter,
    iterate_source_dir_and_generate_to_target_dir,
)
from custom_doc_writer import (  # type:ignore
    assemble_prompt_components,
    llm_context,
    process_content_and_return_result,
    assemble_prompt_components,
)

from dateparser_utils import (
    parse_date_with_multiple_formats,
    render_datetime_as_hexo_format,
    CUSTOM_DATE_FORMATS,
)
from headline_match import (
    modify_content_metadata,
    parse_content_metadata,
    JSONDict,
    purify_dict,
)
from similarity_utils import SimilarityIndex, sentence_transformer_context
from io_utils import load_file, write_file

T = TypeVar("T")
DEFAULT_TOP_K = 7
REQUIRED_FIELDS = ("tags", "title", "description", "category", "date")
FIELDS_THAT_NEED_SUMMARY_TO_GENERATE = ("tags", "title", "description", "category")
DATE_MITIGATION_FIELDS = ("created", "modified")


def generate_markdown_name():
    file_id = str(uuid.uuid4())
    fname = f"{file_id}.md"
    return fname


@beartype
def split_by_line(cnt: str, newline=NEWLINE):
    myitems = cnt.split(newline)
    myitems = [myit.strip() for myit in myitems]
    myitems = [myit for myit in myitems if len(myit) > 0]
    return myitems


@beartype
def join_lines(lines: list[str], newline=NEWLINE):
    return newline.join(lines)


def load_bad_words(fname: str):
    cnt = load_file(fname)
    mybadwords = split_by_line(cnt)
    return mybadwords


# should change the schema according to the need, only generate what is needed the most.


# if filename is not date, use as title
# if filename is date, perform title generation

# TODO: constraint string generation
# validation not working. and the llm does not pay attention to the constraints by the name
# string_without_comma_period = Annotated[str, pydantic.Field(regex=r'^[^,^.]*$')]
# string_without_comma_period_and_space = Annotated[str, pydantic.Field(regex=r'^[^,^ ^.]*$')]


# suspicious chars being used in string, like the comma
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


from retrying import retry


@beartype
def call_llm_once_and_parse(
    init_prompt: str, prompt: str, pydantic_type: type[T], retry_times: int = 3
) -> T:
    def generate_fix_init_prompt():
        identity = "You are a professional JSON response fixer. You can fix data failed to parsed as JSON."
        task = "You will be given the data to be fixed, the error message during parsing the data and return fixed response according to the schema, and a few hints."
        fix_init_prompt = generate_init_prompt_with_schema(
            identity, task, pydantic_type
        )
        return fix_init_prompt

    @beartype
    def generate_fix_prompt(response: str, error: str):
        prompt_context_dict = {
            "Invalid data to be fixed": response,
            "Parsing error message": error,
            "Hint": "Check for quote issues, like using both double quotes inslde and around the string, or invalid format according to the schema.",
        }
        fix_prompt = generate_json_prompt(prompt_context_dict)
        return fix_prompt

    @beartype
    def fix_invalid_response(response: str, error: str):
        fix_init_prompt = generate_fix_init_prompt()
        fix_prompt = generate_fix_prompt(response, error)
        fix_response = call_llm_once(fix_init_prompt, fix_prompt)
        ret = pydantic_type.parse_raw(fix_response)  # type:ignore
        return ret

    @retry(stop_max_attempt_number=retry_times)
    def try_once():
        response = call_llm_once(init_prompt, prompt)
        try:
            ret = pydantic_type.parse_raw(response)  # type: ignore
        except:
            error = traceback.format_exc(limit=1)
            ret = fix_invalid_response(response, error)
        return ret

    return try_once()


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


def generate_item_recommended_init_prompt(
    item_name: str, schema_class: type[T], max_num: Optional[int] = None
):
    components = [
        f"""You will be given an article summary.
You will produce recommended {item_name}.""",
        f"""You can most generate {max_num} {item_name}.""" if max_num else "",
    ]
    task = assemble_prompt_components(components)
    init_prompt = generate_blogger_init_prompt_with_schema(task, schema_class)
    return init_prompt, schema_class


def generate_description_recommended_init_script():
    return generate_item_recommended_init_prompt("description", Description)


def generate_title_recommended_init_script():
    return generate_item_recommended_init_prompt("title", Title)


@beartype
def generate_category_recommender_init_prompt(max_num: int = DEFAULT_TOP_K):
    return generate_item_recommended_init_prompt(
        "categories", Categories, max_num=max_num
    )


@beartype
def generate_tag_recommender_init_prompt(max_num: int = DEFAULT_TOP_K):
    return generate_item_recommended_init_prompt("tags", Tags, max_num=max_num)


@beartype
def generate_item_chooser_init_prompt(
    item_name: str, objective: str, schema_class: type[T]
):
    task = f"""You will be given an article summary, similar {item_name} in database, and your recommended {item_name}.
You would prefer {item_name} in database if they match the summary.
You will choose {objective} that best matches the summary."""
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
    # i miss ollama json format restrictions. can i have that?
    prompt = f"""{prompt_context}

Response in JSON format (curly bracket key-value pairs):
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
    top_k: int = DEFAULT_TOP_K,
) -> T:
    recommended_items = item_recommender(summary)
    similar_items = items_similarity_index.search(recommended_items, top_k=top_k)
    ret = item_chooser(summary, recommended_items, similar_items)
    return ret


@beartype
def generate_category(
    categories_similarity_index: SimilarityIndex,
    summary: str,
    top_k: int = DEFAULT_TOP_K,
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
def generate_tags(
    tags_similarity_index: SimilarityIndex, summary: str, top_k: int = DEFAULT_TOP_K
):
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
    char_limit: int = 1000,
    line_limit: int = 15,
    sample_size: Optional[int] = None,
):
    prompt_base = generate_summary_prompt_base(word_limit)
    prompt_generator = generate_summary_prompt_generator(programming_language)

    with llm_context(prompt_base) as model:
        ret = process_content_and_return_result(
            model,
            prompt_generator,
            filename,
            content_without_metadata,
            char_limit=char_limit,
            line_limit=line_limit,
            sample_size=sample_size,
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
def get_date_obj_by_metadata(metadata: JSONDict):
    for field in DATE_MITIGATION_FIELDS:
        value = metadata.get(field, None)
        date_obj = None
        if isinstance(value, datetime.datetime):
            date_obj = value
        elif isinstance(value, str):
            date_obj = parse_date_with_multiple_formats(CUSTOM_DATE_FORMATS, value)
        if date_obj is not None:
            return date_obj


@beartype
def get_date_obj_by_filepath(filepath: str):
    filename_without_extension = get_filename_without_extension(filepath)
    date_obj = parse_date_with_multiple_formats(
        CUSTOM_DATE_FORMATS, filename_without_extension
    )
    return date_obj


@beartype
def generate_date_obj(filepath: str, metadata: JSONDict):
    maybe_methods = (
        lambda: get_date_obj_by_metadata(metadata),
        lambda: get_date_obj_by_filepath(filepath),
    )
    fallback = lambda: get_date_obj_by_file_ctime(filepath)
    return maybe_with_fallback(maybe_methods, fallback)


@beartype
def generate_date(filepath: str, metadata: JSONDict):
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
def replace_double_quotes_as_single_quotes(content: str):
    return content.replace('"', "'")


@beartype
def generate_content_metadata(
    filepath: str,
    content_without_metadata: str,
    metadata: JSONDict,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    tag_top_k: int = DEFAULT_TOP_K,
    category_top_k: int = DEFAULT_TOP_K,
    summary_word_limit: int = 30,
    programming_language: str = "markdown",
    char_limit: int = 1000,
    line_limit: int = 15,
    sample_size: Optional[int] = None,
):
    @beartype
    def get_additional_metadata(
        missing_fields: Iterable[str], field_to_method: dict[str, Callable[[], str]]
    ):
        additional_metadata = {}
        for field in missing_fields:
            additional_metadata[field] = field_to_method[field]()
        return purify_dict(additional_metadata)

    @beartype
    def generate_new_metadata(additional_metadata: JSONDict):
        changed = False
        new_metadata = metadata.copy()
        if additional_metadata != {}:
            changed = True
            new_metadata.update(additional_metadata)
        return new_metadata, changed

    def build_field_generation_methods_with_summary():
        summary = generate_summary(
            content_without_metadata,
            word_limit=summary_word_limit,
            programming_language=programming_language,
            char_limit=char_limit,
            line_limit=line_limit,
            sample_size=sample_size,
        )

        summary = replace_double_quotes_as_single_quotes(summary)

        data = {
            "tags": lambda: generate_tags(
                tags_similarity_index, summary, top_k=tag_top_k
            ),
            "title": lambda: generate_title(summary),
            "description": lambda: generate_description(summary),
            "category": lambda: generate_category(
                categories_similarity_index, summary, top_k=category_top_k
            ),
        }
        return data

    def find_missing_fields_and_build_field_to_method():
        field_to_method = {
            "date": lambda: generate_date(filepath, metadata),
        }
        missing_fields = [
            field for field in REQUIRED_FIELDS if field not in metadata.keys()
        ]
        if set(missing_fields).intersection(FIELDS_THAT_NEED_SUMMARY_TO_GENERATE):
            field_to_method_with_summary = build_field_generation_methods_with_summary()
            field_to_method.update(field_to_method_with_summary)
        return missing_fields, field_to_method

    def get_new_metadata_and_changed_flag():
        (
            missing_fields,
            field_to_method,
        ) = find_missing_fields_and_build_field_to_method()
        additional_metadata = get_additional_metadata(missing_fields, field_to_method)
        new_metadata, changed = generate_new_metadata(additional_metadata)
        return new_metadata, changed

    return get_new_metadata_and_changed_flag()


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
    has_metadata, metadata, _, _ = parse_content_metadata(content)
    if has_metadata:
        update_tags_and_categories_from_metadata(
            metadata, existing_tags, existing_categories
        )


@beartype
def get_note_paths_without_bad_words_and_existing_tags_and_categories(
    notes_dir: str, bad_words: list[str], cache_dir: str
):
    note_paths: list[str] = []
    existing_tags: set[str] = set()
    existing_categories: set[str] = set()

    @beartype
    def check_bad_words_passed(content: str, check_bad_words: bool):
        if check_bad_words:
            passed = not check_if_contains_bad_words(content, bad_words)
        else:
            passed = True
        return passed

    @beartype
    def append_note_path_and_update_existing_tags_and_categories(
        content: str,
        filepath: str,
        append_and_check_bad_words: bool,
    ):
        if check_bad_words_passed(content, append_and_check_bad_words):
            if append_and_check_bad_words:
                note_paths.append(filepath)
            extract_and_update_existing_tags_and_categories(
                content, existing_tags, existing_categories
            )

    @beartype
    def iterate_dir_and_update_tags_and_categoiries(
        dirpath: str, append_and_check_bad_words: bool = True
    ):
        for fpath in iterate_and_get_markdown_filepath_from_notedir(dirpath):
            content = load_file(fpath)
            append_note_path_and_update_existing_tags_and_categories(
                content, fpath, append_and_check_bad_words
            )

    iterate_dir_and_update_tags_and_categoiries(
        notes_dir,
    )
    iterate_dir_and_update_tags_and_categoiries(
        cache_dir, append_and_check_bad_words=False
    )

    return note_paths, existing_tags, existing_categories


def update_tags_set_from_metadata(metadata: JSONDict, tags_set: set[str]):
    for tag in metadata.get("tags", []):
        tags_set.add(tag)


def update_categories_set_from_metadata(metadata: JSONDict, categories_set: set[str]):
    category = metadata.get("category", None)
    if category:
        categories_set.add(category)


@beartype
def update_tags_and_categories_from_metadata(
    metadata: JSONDict, tags_set: set[str], categories_set: set[str]
):
    update_tags_set_from_metadata(metadata, tags_set)
    update_categories_set_from_metadata(metadata, categories_set)


@beartype
def process_note_content_with_similarity_indices(
    content: str,
    source_path: str,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    sample_size: Optional[int] = None,
):
    (
        has_metadata,
        metadata,
        content_without_metadata,
        first_match,
    ) = parse_content_metadata(content)
    new_metadata, changed = generate_content_metadata(
        source_path,
        content_without_metadata,
        metadata,
        tags_similarity_index,
        categories_similarity_index,
        sample_size=sample_size,
    )
    if changed:
        return modify_content_metadata(content, has_metadata, new_metadata, first_match)
    return content


@beartype
def process_and_write_note_with_similarity_indices(
    source_path: str,
    target_path: str,
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    sample_size: Optional[int] = None,
):
    content = load_file(source_path)
    new_content = process_note_content_with_similarity_indices(
        content,
        source_path,
        tags_similarity_index,
        categories_similarity_index,
        sample_size=sample_size,
    )
    write_file(target_path, new_content)


@beartype
def get_existing_note_info_from_notes_dir_and_bad_words_path(
    notes_dir: str, bad_words_path: str, cache_dir: str
):
    bad_words = load_bad_words(bad_words_path)
    (
        note_paths,
        existing_tags,
        existing_categories,
    ) = get_note_paths_without_bad_words_and_existing_tags_and_categories(
        notes_dir, bad_words, cache_dir
    )
    return (note_paths, existing_tags, existing_categories)


@beartype
def generate_processed_note_path(param: TargetGeneratorParameter):
    basename = generate_markdown_name()
    ret = os.path.join(param.target_dir_path, basename)
    return ret


@beartype
def generate_process_and_write_note_method(
    tags_similarity_index: SimilarityIndex,
    categories_similarity_index: SimilarityIndex,
    sample_size: Optional[int] = None,
):
    @beartype
    def process_and_write_note(
        source_path: str,
        target_path: str,
    ):
        return process_and_write_note_with_similarity_indices(
            source_path,
            target_path,
            tags_similarity_index,
            categories_similarity_index,
            sample_size=sample_size,
        )

    return process_and_write_note


@beartype
def generate_source_walker_from_note_paths(note_paths: list[str]):
    def source_walker(dirpath: str):
        for fpath in note_paths:
            yield dirpath, fpath

    return source_walker


@beartype
def prepare_note_iterator_extra_params(
    note_paths: list[str],
    sent_trans_model: SentenceTransformer,
    existing_tags: set[str],
    existing_categories: set[str],
    sample_size: Optional[int] = None,
):
    @beartype
    def create_similarity_index_with_candidates(candidates: Iterable[str]):
        return SimilarityIndex(sent_trans_model, candidates)

    def get_tags_and_categories_similarity_indices():
        tags_similarity_index = create_similarity_index_with_candidates(existing_tags)
        categories_similarity_index = create_similarity_index_with_candidates(
            existing_categories
        )
        return tags_similarity_index, categories_similarity_index

    def generate_source_walker_and_target_file_generator():
        (
            tags_similarity_index,
            categories_similarity_index,
        ) = get_tags_and_categories_similarity_indices()

        source_walker = generate_source_walker_from_note_paths(note_paths)
        target_file_generator = generate_process_and_write_note_method(
            tags_similarity_index, categories_similarity_index, sample_size=sample_size
        )
        return source_walker, target_file_generator

    return generate_source_walker_and_target_file_generator()


@beartype
def iterate_note_paths_without_bad_words_and_write_to_cache(
    param: SourceIteratorAndTargetGeneratorParam,
    note_paths: list[str],
    existing_tags: set[str],
    existing_categories: set[str],
    sample_size: Optional[int] = None,
) -> list[str]:
    with sentence_transformer_context() as sent_trans_model:
        source_walker, target_file_generator = prepare_note_iterator_extra_params(
            note_paths,
            sent_trans_model,
            existing_tags,
            existing_categories,
            sample_size=sample_size,
        )
        return iterate_source_dir_and_generate_to_target_dir(
            param,
            source_walker=source_walker,
            target_path_generator=generate_processed_note_path,
            target_file_geneator=target_file_generator,
            join_source_dir=False,
        )


@beartype
def walk_notes_source_dir_and_write_to_cache_dir(
    param: SourceIteratorAndTargetGeneratorParam,
    bad_words_path: str,
    sample_size: Optional[int] = None,
):
    (
        note_paths,
        existing_tags,
        existing_categories,
    ) = get_existing_note_info_from_notes_dir_and_bad_words_path(
        param.source_dir_path, bad_words_path, param.target_dir_path
    )
    return iterate_note_paths_without_bad_words_and_write_to_cache(
        param, note_paths, existing_tags, existing_categories, sample_size=sample_size
    )


@beartype
def remove_and_create_dir(dirpath: str):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

@beartype
def fix_date_and_get_title_in_content(filepath: str, content: str) -> tuple[str, str]:
    def fix_date_and_get_title():
        (
            has_metadata,
            metadata,
            _,
            first_match,
        ) = parse_content_metadata(content)

        @beartype
        def get_new_content_and_title() -> tuple[str, str]:
            title = metadata.get("title", "")
            date = generate_date(filepath, metadata)
            metadata["date"] = date
            new_content = modify_content_metadata(
                content, has_metadata, metadata, first_match
            )
            return new_content, title

        def process_parsed_metadata():
            title = ""
            new_content = content
            if has_metadata:
                if metadata is not None:
                    new_content, title = get_new_content_and_title()
            return new_content, title

        return process_parsed_metadata()

    return fix_date_and_get_title()


@beartype
def remove_headline_from_content(content: str, title: str):
    lines = split_by_line(content)
    new_lines = remove_headline_from_lines(lines, title)
    return join_lines_with_state(new_lines)


@beartype
def fix_date_in_cache_and_write_to_final_dir(
    processed_cache_paths: list[str], final_dir: str
):
    remove_and_create_dir(final_dir)
    for path in processed_cache_paths:
        content = load_file(path)
        content, title = fix_date_and_get_title_in_content(path, content)
        content = remove_headline_from_content(content, title)

        new_path = os.path.join(final_dir, os.path.basename(path))
        write_file(new_path, content)

@beartype
def parse_params() -> tuple[SourceIteratorAndTargetGeneratorParam, str, str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes-source-dir", type=str, default="notes")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--final-dir", type=str, default="source/_posts")
    parser.add_argument("--db-path", type=str, default="cache_db.json")
    parser.add_argument("--bad-words-path", type=str, default="bad_words.txt")
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()
    param = SourceIteratorAndTargetGeneratorParam(
        source_dir_path=args.notes_source_dir,
        target_dir_path=args.cache_dir,
        db_path=args.db_path,
    )
    return param, args.bad_words_path, args.final_dir, args.sample_size


def main():
    param, bad_words_path, final_dir, sample_size = parse_params()
    processed_cache_paths = walk_notes_source_dir_and_write_to_cache_dir(
        param, bad_words_path, sample_size=sample_size
    )
    fix_date_in_cache_and_write_to_final_dir(processed_cache_paths, final_dir)


if __name__ == "__main__":
    main()
