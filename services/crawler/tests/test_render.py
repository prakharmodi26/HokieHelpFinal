from crawler.render import fast_run_config, js_run_config, bucket_for_scope
from crawler.scope import Scope


def test_run_configs_set_robots_and_generator():
    fast = fast_run_config(threshold=0.5)
    js = js_run_config(threshold=0.5)
    assert fast.check_robots_txt is True
    assert js.check_robots_txt is True
    assert js.scan_full_page is True
    assert js.page_timeout >= fast.page_timeout


def test_bucket_for_scope_routing():
    assert bucket_for_scope(Scope.VT_PAGE, "crawled-pages", "crawled-pages-external") == "crawled-pages"
    assert bucket_for_scope(Scope.DOCUMENT, "crawled-pages", "crawled-pages-external") == "crawled-pages"
    assert bucket_for_scope(Scope.EXTERNAL_PAGE, "crawled-pages", "crawled-pages-external") == "crawled-pages-external"
