import os

# use sys.path.append to insert dependencies

# ask llm to give some potential tags & category for content chunks
# calculate cosine similarity to existing content
# ask the llm to use existing tag & category or create new ones.
# check if the newly created tag & category exists and update

# to create title:
# summarize the content
# generate title from summary

# get the time:
# usr rclone to preserve timestamp
# get mtime from file metadata

# set mirror path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sentence_transformers  # i recommend you to use cpu only.
import sentence_transformers.util

model = sentence_transformers.SentenceTransformer(
    "distiluse-base-multilingual-cased-v1", device="cpu"
)

# Sample text document
# text = "我爱我的宠物狗，并且经常和它在一起。" # does not work with chinese.
text = "I love my pet dog and spend a lot of time with it."
query_embedding = model.encode(text)

# Predefined list of tags
tags = ["dogs", "cats", "computer", "tech", "life"]
tags_embedding = model.encode(tags)

tag_similarities = sentence_transformers.util.cos_sim(query_embedding, tags_embedding)
print("Similarity:", tag_similarities)  # this is fast, but inaccurate.

# # Find the closest tags based on similarity
# tag_similarities.sort(key=lambda item: item[1], reverse=True)
# closest_tags = tag_similarities[:3]
# # Display the closest tags
# print("Closest tags:", closest_tags)
