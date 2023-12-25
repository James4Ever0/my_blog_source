from contextlib import contextmanager
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
from contextlib import contextmanager


@contextmanager
def sentence_transformer_context(
    model_name="distiluse-base-multilingual-cased-v1", device="cpu", **kwargs
):
    model = sentence_transformers.SentenceTransformer(
        model_name, device=device, **kwargs
    )
    try:
        yield model
    finally:
        del model


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
        ret = torch.cat(embed_list, dim=0)
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
        query_length, similarity_list, top_k_indices = self.get_similarity_info(
            query, top_k
        )
        ret = self.prepare_search_results(
            query_length, similarity_list, top_k_indices, return_similarity
        )
        return ret

    def prepare_search_results(
        self,
        query_length: int,
        similarity_list: list[float],
        top_k_indices: list[int],
        return_similarity: bool,
    ):
        if return_similarity:
            return {
                self.word_index[ind]: similarity_list[ind] / query_length
                for ind in top_k_indices
            }
        else:
            return [self.word_index[ind] for ind in top_k_indices]

    def can_compute_similarity(self, query: Union[str, list[str]]):
        query_length = 1 if isinstance(query, str) else len(query)
        can_compute = self.index_size > 0 and query_length > 0
        return can_compute, query_length

    def compute_similarity_and_get_top_k_indices(
        self, query: Union[str, list[str]], top_k: int = 10
    ):
        similarity = self.compute_similarity(query)
        similarity_list = similarity.tolist()
        top_k_indices = self.get_top_k_indices(similarity, top_k)
        return similarity_list, top_k_indices

    def get_similarity_info(self, query: Union[str, list[str]], top_k: int = 10):
        similarity_list = []
        top_k_indices = []
        can_compute, query_length = self.can_compute_similarity(query)
        if can_compute:
            (
                similarity_list,
                top_k_indices,
            ) = self.compute_similarity_and_get_top_k_indices(query, top_k)

        return query_length, similarity_list, top_k_indices

    @property
    def index_size(self):
        return len(self.word_index)

    def get_top_k_indices(self, similarity: torch.Tensor, top_k=10):
        if self.index_size <= top_k:
            top_k_indices = list(range(self.index_size))
        else:
            top_k_indices = torch.topk(similarity, top_k).indices.squeeze().tolist()
        return top_k_indices


def test_main():
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

    with sentence_transformer_context() as model:
        sim_index = SimilarityIndex(model, candidates=tags)

        for it in texts:
            ret = sim_index.search(query=it)
            print(it, "->", ret)


if __name__ == "__main__":
    test_main()
