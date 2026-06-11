from crawler.registry import load_seed_hosts, host_root_urls, load_modes, host_handler

_REG = """
full:
  - registrar.vt.edu
catalog:
  - host: vtechworks.lib.vt.edu
    type: dspace
special:
  - host: catalog.vt.edu
    handler: courseleaf_catalog
  - host: dining.vt.edu
    handler: dining_tool
"""


def test_seed_hosts_excludes_runtime_tool_handlers(tmp_path):
    p = tmp_path / "reg.yaml"; p.write_text(_REG)
    hosts = load_seed_hosts(str(p))
    # full + catalog + browser-crawled special are seeded; dining_tool is NOT (runtime tool)
    assert "registrar.vt.edu" in hosts
    assert "vtechworks.lib.vt.edu" in hosts
    assert "catalog.vt.edu" in hosts
    assert "dining.vt.edu" not in hosts
    assert "https://registrar.vt.edu/" in host_root_urls(hosts)


def test_modes_and_handlers(tmp_path):
    p = tmp_path / "reg.yaml"; p.write_text(_REG)
    modes = load_modes(str(p))
    assert modes["registrar.vt.edu"] == "full"
    assert modes["vtechworks.lib.vt.edu"] == "catalog"
    assert modes["dining.vt.edu"] == "special"
    assert host_handler(str(p), "dining.vt.edu") == "dining_tool"
    assert host_handler(str(p), "catalog.vt.edu") == "courseleaf_catalog"
