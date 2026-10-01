"""Read-only BGE-M3 retrieval contracts, without loading models or using a cluster."""
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import httpx
import pytest

from chatbot.retriever import Retriever, canonical_url, chunk_key, compose_passage


def hit(index, *, text=None, url=None, score=.5):
    return NS(id=index, score=score, payload={"chunk_id": str(index), "text": text or f"Passage {index}",
              "url": url or f"https://vt.edu/{index}", "title": f"Title {index}"})


@pytest.fixture
def retriever():
    r = Retriever.__new__(Retriever)
    r._collection = "hokiehelp-test"
    r._top_k = 2
    r._candidate_k = 30
    r._min_score = .3
    r._hybrid_enabled = True
    r._program_candidates_enabled = True
    r._client = MagicMock()
    r._client.query_points.return_value = NS(points=[hit(1), hit(2), hit(3)])
    r.calls = []
    def handler(request):
        body = json.loads(request.content)
        r.calls.append((request.url.path, body))
        if request.url.path == "/embed":
            return httpx.Response(200, json={"dense": [[.1] * 1024], "sparse": [{"indices": [1], "values": [.2]}]})
        assert request.url.path == "/rerank"
        return httpx.Response(200, json={"results": [{"index": 0, "score": .1}, {"index": 1, "score": .9}, {"index": 2, "score": .8}]})
    r._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(handler))
    yield r
    r.close()


def test_raw_bge_m3_query_named_hybrid_no_neighbors(retriever):
    result = retriever.search("CS credits")
    assert retriever.calls[0] == ("/embed", {"texts": ["CS credits"]})
    kwargs = retriever._client.query_points.call_args.kwargs
    assert [p.using for p in kwargs["prefetch"]] == ["dense", "sparse"]
    assert all(p.limit == 30 for p in kwargs["prefetch"])
    assert kwargs["limit"] == 30
    assert [c["chunk_id"] for c in result] == ["2", "3"]
    assert result[0]["fusion_score"] == .5
    assert result[0]["rerank_score"] == .9
    assert result[0]["query_fusion_score"] == pytest.approx(1 / 62)
    assert "vector_score" not in result[0]
    retriever._client.scroll.assert_not_called()
    retriever._client.create_payload_index.assert_not_called()


def test_dense_path_uses_same_rerank_floor(retriever):
    retriever._hybrid_enabled = False
    result = retriever.search("Q")
    kwargs = retriever._client.query_points.call_args.kwargs
    assert kwargs["using"] == "dense"
    assert "prefetch" not in kwargs
    assert [c["chunk_id"] for c in result] == ["2", "3"]
    assert result[0]["vector_score"] == .5
    assert "fusion_score" not in result[0]


def test_empty_sparse_dense_fallback(retriever):
    def handler(request):
        return httpx.Response(200, json={"dense": [[.1] * 1024], "sparse": [{"indices": [], "values": []}]})
    retriever._embedding_client.close()
    retriever._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(handler))
    retriever._candidates("Q")
    assert retriever._client.query_points.call_args.kwargs["using"] == "dense"


def test_content_url_dedup_before_rerank_without_mutation(retriever):
    points = [hit(1, text="Same", url="https://www.vt.edu/page#x"),
              hit(2, text="Same", url="http://vt.edu/page/?utm_source=x"), hit(3)]
    retriever._client.query_points.return_value = NS(points=points)
    result = retriever._candidates("Q")
    assert len(result) == 2
    assert "fusion_score" not in points[0].payload
    assert points[0].score == .5


def test_multiquery_fusion_one_global_cap(retriever):
    a = {"chunk_id": "a", "text": "a", "url": "https://vt.edu/a"}
    b = {"chunk_id": "b", "text": "b", "url": "https://vt.edu/b"}
    c = {"chunk_id": "c", "text": "c", "url": "https://vt.edu/c"}
    retriever._candidates = MagicMock(side_effect=[[a, b], [c, b], [c]])
    retriever._rank = MagicMock(side_effect=lambda query, pool: [{**c, "rerank_score": .9} for c in pool])
    result = retriever.search_many(["Q1", "Q2"], original_query="raw question")
    assert len(result) == 2
    assert {c["chunk_id"] for c in result} == {"b", "c"}
    assert len(result[0]["query_ranks"]) == 2
    assert "query_ranks" not in b
    assert retriever._rank.call_count == 1
    assert retriever._rank.call_args.args[0] == "Q1"
    assert retriever._candidates.call_args_list[-1].args == ("raw question",)


def test_original_identical_to_plan_not_searched_twice(retriever):
    retriever.search_many(["Q"], original_query="q")
    assert retriever._client.query_points.call_count == 1


@pytest.mark.parametrize("queries", [[], ["a"] * 4, [""], [None]])
def test_query_limits(retriever, queries):
    with pytest.raises(ValueError):
        retriever.search_many(queries)


def test_dependency_failure_not_empty_success(retriever):
    retriever._client.query_points.side_effect = RuntimeError("database failed")
    with pytest.raises(RuntimeError):
        retriever.search("Q")


def test_rerank_bad_index_is_failure(retriever):
    retriever._embedding_client.close()
    retriever._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [{"index": 99, "score": .9}]})))
    with pytest.raises(ValueError, match="index"):
        retriever._rank("Q", [{"text": "body"}])


def test_incomplete_reranking_is_not_successful_no_evidence(retriever):
    retriever._embedding_client.close()
    retriever._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": []})))
    with pytest.raises(ValueError, match="Incomplete"):
        retriever._rank("Q", [{"text": "body"}])


def test_invalid_embedding_fails_before_qdrant(retriever):
    retriever._embedding_client.close()
    retriever._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"dense": [[1, 2]], "sparse": [{}]})))
    with pytest.raises(ValueError, match="embedding"):
        retriever._candidates("Q")
    retriever._client.query_points.assert_not_called()


def test_constructor_checks_named_index_without_writes():
    client = MagicMock()
    client.get_collection.return_value = NS(config=NS(params=NS(vectors={"dense": NS(size=1024)}, sparse_vectors={"sparse": {}})))
    with patch("chatbot.retriever.QdrantClient", return_value=client):
        retriever = Retriever("BAAI/bge-m3", "localhost", 6333, "hokiehelp-test")
        client.create_payload_index.assert_not_called()
        retriever.close()
        client.get_collection.return_value.config.params.vectors = NS(size=1024)
        with pytest.raises(ValueError, match="named"):
            Retriever("BAAI/bge-m3", "localhost", 6333, "wrong")


def program_hit(index, name="Computer Science Major", *, url="https://catalog.vt.edu/cs/"):
    point = hit(index, url=url)
    point.payload.update(page_type="program", program_name=name, program_level="undergraduate",
                         catalog_edition="2026-2027")
    return point


def test_catalog_lane_reuses_embedding_and_shares_candidate_budget(retriever):
    general = [hit(i) for i in range(30)]
    scoped = [program_hit(i) for i in range(100, 110)] + [program_hit(200, "Dairy Science Major")]
    retriever._client.query_points.side_effect = [NS(points=general), NS(points=scoped)]
    result = retriever._candidates("Virginia Tech undergraduate Computer Science major degree credits")
    assert len(result) == 30
    assert sum(c["candidate_lane"] == "program" for c in result) == 6
    assert retriever.calls == [("/embed", {"texts": ["Virginia Tech undergraduate Computer Science major degree credits"]})]
    calls = retriever._client.query_points.call_args_list
    assert all(p.filter is None for p in calls[0].kwargs["prefetch"])
    for p in calls[1].kwargs["prefetch"]:
        assert p.limit == 30
        assert [(c.key, c.match.value) for c in p.filter.must] == [("page_type", "program"), ("program_level", "undergraduate")]
    retriever._client.scroll.assert_not_called()
    retriever._client.create_payload_index.assert_not_called()
    assert "candidate_lane" not in scoped[0].payload


def test_catalog_lane_dense_mode_and_disabled_switch(retriever):
    retriever._hybrid_enabled = False
    retriever._candidates("undergraduate Computer Science major credits")
    assert retriever._client.query_points.call_args.kwargs["using"] == "dense"
    assert retriever._client.query_points.call_args.kwargs["query_filter"] is not None
    retriever._program_candidates_enabled = False
    retriever._client.query_points.reset_mock()
    retriever._candidates("undergraduate Computer Science major credits")
    assert retriever._client.query_points.call_count == 1
    assert retriever._client.query_points.call_args.kwargs["query_filter"] is None


def test_nonmatching_catalog_candidates_do_not_displace_general(retriever):
    general = [hit(i) for i in range(30)]
    retriever._client.query_points.side_effect = [NS(points=general), NS(points=[program_hit(100, "Dairy Science Major")])]
    result = retriever._candidates("undergraduate Computer Science major credits")
    assert [c["chunk_id"] for c in result] == [str(i) for i in range(30)]


def test_comparison_lane_round_robins_program_pages(retriever):
    scoped = [program_hit(i) for i in range(100, 110)] + [program_hit(200, "Mathematics Major", url="https://catalog.vt.edu/math/")]
    retriever._client.query_points.side_effect = [NS(points=[hit(i) for i in range(30)]), NS(points=scoped)]
    result = retriever._candidates("Compare undergraduate Computer Science and mathematics major credits")
    assert [c["chunk_id"] for c in result[:2]] == ["100", "200"]
    assert len(result) == 30


def test_priority_selection_uses_identity_not_compressed_rerank_score(retriever):
    wrong = {**hit(1).payload, "rerank_score": .99}
    right = {**program_hit(2).payload, "rerank_score": .575}
    extra = {**program_hit(3).payload, "rerank_score": .55}
    ranked = [wrong, right, extra]
    query = "undergraduate Computer Science major credits"
    assert [c["chunk_id"] for c in retriever._select_ranked(query, ranked)] == ["2", "1"]
    assert [c["chunk_id"] for c in retriever._select_ranked(query + " for 2023-24", ranked)] == ["1"]
    retriever._program_candidates_enabled = False
    assert retriever._select_ranked(query, ranked) == [wrong, right]


def test_catalog_identity_never_bypasses_rerank_floor(retriever):
    retriever._candidates = MagicMock(return_value=[hit(1).payload, program_hit(2).payload])
    retriever._embedding_client.close()
    retriever._embedding_client = httpx.Client(base_url="http://embedding/", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [{"index": 0, "score": .9}, {"index": 1, "score": .2}]})))
    result = retriever.search("undergraduate Computer Science major credits")
    assert [c["chunk_id"] for c in result] == ["1"]


def test_explicit_wrong_program_sources_filtered_after_rerank(retriever):
    right = {**program_hit(1).payload, "rerank_score": .6}
    stat = {**hit(2, url="https://registrar.vt.edu/Updates/cos/23-24/cos_stat_smth_23_24.pdf").payload,
            "title": "Cos Stat Smth 23 24", "rerank_score": .99}
    graduate = {**hit(3, url="https://students.cs.vt.edu/Graduate/Degrees/MENG.html").payload,
                "title": "Master of Engineering | students.cs | Virginia Tech", "rerank_score": .98}
    sparse = {**hit(4, url="https://vt.edu/files/unlabeled.pdf").payload,
              "title": "Unlabeled Requirements PDF", "rerank_score": .97}
    query = "Virginia Tech undergraduate Computer Science major credits"
    assert [c["chunk_id"] for c in retriever._select_ranked(query, [stat, graduate, sparse, right])] == ["1", "4"]
    retriever._program_candidates_enabled = False
    assert [c["chunk_id"] for c in retriever._select_ranked(query, [stat, sparse, right])] == ["4", "1"]


def test_source_filter_preserves_comparison_entities(retriever):
    cs = {**program_hit(1).payload, "rerank_score": .7}
    math = {**program_hit(2, "Mathematics Major", url="https://catalog.vt.edu/math/").payload, "rerank_score": .8}
    stat = {**program_hit(3, "Statistics Major", url="https://catalog.vt.edu/stat/").payload, "rerank_score": .99}
    query = "Compare undergraduate Computer Science and Mathematics major credit requirements"
    assert [c["chunk_id"] for c in retriever._select_ranked(query, [stat, math, cs])] == ["2", "1"]


def test_helpers():
    assert canonical_url("https://www.VT.edu/index.html?utm_source=test#x") == "vt.edu/"
    assert compose_passage({"title": "## Computer%20Science", "headings_path": ["Computer Science", "**Requirements**"], "text": "Body"}) == "Computer Science\nRequirements\n\nBody"
    assert chunk_key({"text": "one", "url": "https://vt.edu"}) != chunk_key({"text": "two", "url": "https://vt.edu"})
