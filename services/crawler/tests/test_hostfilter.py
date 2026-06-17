from crawler.discovery.hostfilter import HostFilter


def test_blocks_by_pattern_and_explicit_host():
    hf = HostFilter(patterns=[r"^login\.", r"^mail\."], hosts=["banweb.banner.vt.edu"])
    assert hf.is_blocked("login.vt.edu") is True
    assert hf.is_blocked("mail.cs.vt.edu") is True
    assert hf.is_blocked("banweb.banner.vt.edu") is True
    assert hf.is_blocked("dining.vt.edu") is False


def test_from_yaml(tmp_path):
    p = tmp_path / "block.yaml"
    p.write_text("patterns:\n  - '^git'\nhosts:\n  - dead.vt.edu\n")
    hf = HostFilter.from_yaml(str(p))
    assert hf.is_blocked("gitlab.cs.vt.edu") is True
    assert hf.is_blocked("dead.vt.edu") is True
    assert hf.is_blocked("www.vt.edu") is False


def test_path_patterns_block_archive_traps_but_not_front_page():
    hf = HostFilter(patterns=[], hosts=[], path_patterns=[r"/playlists?/", r"/track/"])
    # front page survives
    assert hf.is_blocked_url("https://www.wuvt.vt.edu/") is False
    # playlist/track archive firehose is dropped
    assert hf.is_blocked_url("https://www.wuvt.vt.edu/playlists/track/449476") is True
    assert hf.is_path_blocked("https://www.wuvt.vt.edu/playlist/123") is True


def test_from_yaml_merges_path_patterns_and_extra():
    p = tmp_path_factory()
    p.write_text("path_patterns:\n  - '/calendar/'\nhosts: []\npatterns: []\n")
    hf = HostFilter.from_yaml(str(p), extra_path_patterns=[r"\?page=\d"])
    assert hf.is_blocked_url("https://events.vt.edu/calendar/2026") is True
    assert hf.is_blocked_url("https://x.vt.edu/list?page=3") is True
    assert hf.is_blocked_url("https://x.vt.edu/about") is False


def tmp_path_factory():
    import tempfile, pathlib
    return pathlib.Path(tempfile.mkdtemp()) / "block.yaml"
