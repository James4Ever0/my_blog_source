import os
from typing import Iterable, Optional, Union, overload, Literal
from beartype import beartype
import weakref
import torch

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

from functools import lru_cache


@lru_cache(maxsize=1)
def get_sentence_transformer_model():
    model = sentence_transformers.SentenceTransformer(
        "distiluse-base-multilingual-cased-v1", device="cpu"
    )
    return model


@beartype
class SimilarityIndex(object):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        candidates: Iterable[str] = [],
    ):
        self.init_properties(model)
        self.insert_multiple_candidates(candidates)

    def init_index(self):
        self.word_index: list[str] = []
        self.embedding_index: Optional[torch.Tensor] = None

    def init_properties(self, model: sentence_transformers.SentenceTransformer):
        self.init_index()
        self._model_weakref_ = weakref.ref(model)

    @property
    def model(self):
        return self._model_weakref_()

    def encode_multiple(self, items: list[str]):
        embed_list = [self.encode_single(it) for it in items]
        ret = torch.cat(embed_list, dim = 0)
        return ret

    def encode_single(self, it: str):
        embed: torch.Tensor = self.model.encode([it], convert_to_tensor=True)  # type: ignore
        return embed

    def encode(self, it: Union[str, list[str]]):
        if isinstance(it, str):
            return self.encode_single(it)
        else:
            return self.encode_multiple(it)

    def update_embedding_index(self, embed: torch.Tensor):
        if self.embedding_index is None:
            self.embedding_index = embed
        else:
            self.embedding_index = torch.cat([self.embedding_index, embed], dim=0)

    def insert_single_candidate(self, candidate: str):
        it = candidate.strip()
        if it:
            if it not in self.word_index:
                self.word_index.append(it)
                embed = self.encode_single(it)
                self.update_embedding_index(embed)

    def insert_multiple_candidates(self, candidates: Iterable[str]):
        for it in candidates:
            self.insert_single_candidate(it)

    def compute_similarity(self, it: Union[str, list[str]]):
        if self.embedding_index is None:
            raise Exception("No embedding index yet. Cannot compute similarity.")
        embed = self.encode(it)
        similarity = sentence_transformers.util.cos_sim(embed, self.embedding_index)
        similarity = torch.sum(similarity, dim=0)
        return similarity

    # first overload should agree with default keyword arguments
    @overload
    def search(
        self,
        query: Union[str, list[str]],
        top_k: int = 10,
        return_similarity: Literal[False] = False,
    ) -> list[str]:
        ...

    @overload
    def search(
        self,
        query: Union[str, list[str]],
        top_k: int = 10,
        return_similarity: Literal[True] = True,
    ) -> dict[str, float]:
        ...

    def search(
        self,
        query,
        top_k=10,
        return_similarity=False,
    ):
        query_length = 1 if isinstance(query, str) else len(query)
        similarity = self.compute_similarity(query)
        similarity_list = similarity.tolist()
        top_k_indices = self.get_top_k_indices(similarity, top_k)
        if return_similarity:
            return {
                self.word_index[ind]: similarity_list[ind] / query_length
                for ind in top_k_indices
            }
        else:
            return [self.word_index[ind] for ind in top_k_indices]

    @property
    def index_size(self):
        return len(self.word_index)

    def get_top_k_indices(self, similarity: torch.Tensor, top_k=10):
        if self.index_size <= top_k:
            top_k_indices = list(range(self.index_size))
        else:
            top_k_indices = torch.topk(similarity, top_k).indices.squeeze().tolist()
        return top_k_indices


if __name__ == "__main__":
    texts = ["I love my pet dog and spend a lot of time with it.", "我爱我的宠物狗，并且经常和它在一起。"]
    tags = [
        "dogs",
        "cats",
        "computer",
        "tech",
        "life",
        "dress",
        "cook",
        "outfit",
        "fixing",
        "mechanics",
        "car",
        "gasoline",
    ]

    sentence_transformer_model = get_sentence_transformer_model()

    sim_index = SimilarityIndex(sentence_transformer_model, candidates=tags)

    for it in texts:
        ret = sim_index.search(query=it)
        # ret = sim_index.search(query=it, return_similarity=True)
        print(it, "->", ret)
