from crawler.registry import (
    host_handler,
    host_root_urls,
    load_catalog_entries,
    load_modes,
    load_seed_hosts,
)

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


def test_load_catalog_entries_normalises_to_dicts(tmp_path):
    p = tmp_path / "reg.yaml"
    p.write_text("""
full:
  - registrar.vt.edu
catalog:
  - host: vtechworks.lib.vt.edu
    type: dspace
    oai: /server/oai/request
    rest: /server/api
  - host: opendsa.cs.vt.edu
  - bare-string-host.vt.edu
special:
  - host: dining.vt.edu
    handler: dining_tool
""")
    entries = load_catalog_entries(str(p))
    by_host = {e["host"]: e for e in entries}
    assert by_host["vtechworks.lib.vt.edu"]["oai"] == "/server/oai/request"
    assert by_host["opendsa.cs.vt.edu"] == {"host": "opendsa.cs.vt.edu"}
    assert by_host["bare-string-host.vt.edu"] == {"host": "bare-string-host.vt.edu"}
    assert "registrar.vt.edu" not in by_host
    assert "dining.vt.edu" not in by_host
