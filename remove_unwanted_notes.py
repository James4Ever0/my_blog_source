import tinydb
import os
from headline_match import parse_metadata, modify_metadata

# import mistletoe
# from mistletoe.block_token import SetextHeading

cache_dir = "cache"
notes_dir = "source/_posts"

db_path = "cache_db.json"
bad_words_path = "bad_words.txt"

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

# def filter_instance_in_children(parent, instance_class):
#     return [
#         child for child in parent.children if isinstance(child, instance_class) 
#     ]

# def check_instance_count_in_children(parent, instance_class):
#     my_instance_check_list = filter_instance_in_children(parent, instance_class)
#     my_instance_count = len(my_instance_check_list)
#     return my_instance_count

# def filter_setextheading_in_children(parent):
#     return filter_instance_in_children(parent, SetextHeading)

# def check_setextheading_count_in_children(parent):
#     return check_instance_count_in_children(parent, SetextHeading)

# use two hashs for cache varification
# store filename before and after processing
# one before processing, one after processing
# store processed ones to some place for cacheing

for filename in os.listdir(notes_dir):
    if filename.endswith(".md"):
        path = os.path.join(notes_dir, filename)
        content = load_file(path)
        metadata = parse_metadata(content)
        new_content = modify_metadata(new_metadata)

        # instance_list = filter_setextheading_in_children(md_obj)
        # first_instance = instance_list[0]
        # for child in first_instance.children if ['modified' in children]
        # assert instance_count == 1, "file %s has more than one instance (%d)" % (filename, instance_count)
        # chores to do: tagging, categorization
        # filter unwanted content: passwords, bad articles
