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
