import spacy

# Load the English language model with word vectors
# nlp = spacy.load("en_core_web_trf") # no similarity.
nlp = spacy.load("en_core_web_md") # this is much better. small model is shit.
# nlp = spacy.load("en_core_web_sm") # small we bave.

# Predefined list of tags
tags = ["dogs", "cats", "computer", "tech", "life"]

# Sample text document
text = "我爱我的宠物狗，并且经常和它在一起。" # does not work with chinese.
# text = "I love my pet dog and spend a lot of time with it."

# Process the text using spaCy
doc = nlp(text)

# Calculate similarity between the document and each tag
tag_similarities = [(tag, nlp(tag).similarity(doc)) for tag in tags]

# Find the closest tags based on similarity
tag_similarities.sort(key=lambda item: item[1], reverse=True)
closest_tags = tag_similarities[:3]
# Display the closest tags
print("Closest tags:", closest_tags)
