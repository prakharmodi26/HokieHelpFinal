# Lane 3 — App-like / Infra — tiering
| host | mode | confidence | reason | notes (js/api) |
|------|------|-----------|--------|----------------|
| arc01.cntrlsrvs.w2k.vt.edu | drop | high | Control-services / w2k infra host, not public content | — |
| arc03.cntrlsrvs.w2k.vt.edu | drop | high | Control-services / w2k infra host, not public content | — |
| arc03.cc.vt.edu | drop | high | Numbered ArcGIS/compute infra node, no readable site | — |
| arc01.gis.vt.edu | drop | high | Numbered ArcGIS server node, machine endpoint | — |
| arcgis-dev.gis.vt.edu | drop | high | Dev ArcGIS instance, staging mirror | — |
| assets.cms.vt.edu | drop | high | CMS asset/CDN host, no readable text | — |
| author.cms.vt.edu | drop | high | CMS authoring endpoint, not public content | — |
| author.ensemble.vt.edu | drop | high | Ensemble media authoring/admin endpoint | — |
| autodiscover.vtc.vt.edu | drop | high | Exchange/Outlook autodiscover service, no content | — |
| autodiscover.w2k-dev.vt.edu | drop | high | Autodiscover dev infra service, no content | — |
| birdseye.cntrlsrvs.w2k.vt.edu | drop | high | Control-services / w2k infra host | — |
| cannondale.cns.ipv6.vt.edu | drop | high | Named ipv6 network/CNS infra host, no public content | — |
| cmsa-pprd-01.db.vt.edu | drop | high | Database server (pre-prod), not a website | — |
| cmsa-pprd-02.db.vt.edu | drop | high | Database server (pre-prod), not a website | — |
| cmsa-pprd-03.db.vt.edu | drop | high | Database server (pre-prod), not a website | — |
| cmsa-prod-01.db.vt.edu | drop | high | Database server (prod), not a website | — |
| cmsa-prod-02.db.vt.edu | drop | high | Database server (prod), not a website | — |
| cmsa-prod-03.db.vt.edu | drop | high | Database server (prod), not a website | — |
| cmsa-prod.db.vt.edu | drop | high | Database server (prod), not a website | — |
| cms-pprd.hosting.vt.edu | drop | high | CMS pre-prod hosting backend, staging mirror | — |
| cmsw-prod.hosting.vt.edu | drop | high | CMS prod hosting backend node, not a public site | — |
| docs-sandbox.crash.cgit.vt.edu | drop | high | Sandbox/test docs mirror, dev environment | — |
| hstest.vpas.vt.edu | drop | high | Test host (hstest), staging environment | — |
| portal.vtdev.vt.edu | drop | high | vtdev portal, dev environment | — |
| secure-dev-02.nis.vt.edu | drop | high | Dev infra node (NIS), staging | — |
| secure-dev.nis.vt.edu | drop | high | Dev infra node (NIS), staging | — |
| secure-test.gis.vt.edu | drop | high | Test GIS host, staging mirror | — |
| webdav-prod-02.gis.vt.edu | drop | high | WebDAV file-transfer endpoint, no readable site | — |
| webdav.gis.vt.edu | drop | high | WebDAV file-transfer endpoint, no readable site | — |
| www.sslvpn.export.vt.edu | drop | high | SSL VPN login gateway, no public content | — |
| registry.gitlab.vtti.vt.edu | drop | high | GitLab container registry, machine endpoint | — |
| kibana.insight.aws.vtti.ad.vt.edu | drop | high | Kibana monitoring dashboard, internal, no public content | — |
| apps.cals.vt.edu | drop | high | Parked page: "Nothing Here / nothing of interest" | — |
| ensemble.vt.edu | drop | high | Ensemble media platform, returns HTTP 401 auth | — |
| ensemble.cms.vt.edu | drop | high | Ensemble media admin, returns HTTP 403 | — |
| author.ensemble.vt.edu (dup-check) | drop | high | (already listed) Ensemble authoring endpoint | — |
| confluence.research.vt.edu | drop | high | Redirects to Confluence wiki, returns HTTP 403 (login-gated) | — |
| www.confluence.research.vt.edu | drop | high | Same Confluence wiki, login-gated, no public content | — |
| couchdb.apps.wuvt.vt.edu | drop | high | CouchDB database for WUVT radio, internal, TLS-mismatch | api (internal DB) |
| impala.apps.wuvt.vt.edu | drop | high | Internal WUVT app service, TLS cert mismatch, not public | — |
| linx.apps.wuvt.vt.edu | drop | high | WUVT app, redirects to OAuth2 login | — |
| distleech.apps.wuvt.vt.edu | drop | high | Internal WUVT app service, TLS cert mismatch, not public | — |
| registrar.vt.edu | full | high | University Registrar homepage, full readable content | — |
| www.registrar.vt.edu | full | high | Registrar site (www alias), full readable content | — |
| soiltest.vt.edu | full | high | Soil Testing Lab, full readable informational content | — |
| www.soiltest.vt.edu | full | high | Soil Testing Lab (www alias), full readable content | — |
| cgit.vt.edu | full | high | Center for Geospatial IT, full readable content | — |
| www.cgit.vt.edu | full | high | CGIT site (www alias), full readable content | — |
| cast.centers.vt.edu | full | high | Center for Advanced Separation Tech, full readable content | — |
| www.cast.centers.vt.edu | full | high | CAST site (www alias), full readable content | — |
| ecoforecast.centers.vt.edu | full | high | Center for Ecosystem Forecasting, full readable content | — |
| jobs.apply.vt.edu | full | med | VT careers site, readable text + job-search interface | listings may be JS |
| engapplets.vt.edu | special | med | Legacy frameset + Java applets; plain fetch returns frame shell | js (frameset/applets) |
| www.engapplets.vt.edu | special | med | Same legacy applet site (www alias), frame shell on fetch | js (frameset/applets) |
| engapplets.aoe.vt.edu | special | med | Legacy frameset + Java applets (AOE dept); fetch returns frame shell | js (frameset/applets) |
| geovine.cgit.vt.edu | special | med | Geovine viticulture geospatial app: interactive data layers + login forms | js (map/SPA app) |
| garden.gis.vt.edu | full | med | Redirects to GARDEN geospatial archive landing, readable content | — |
| secure.gis.vt.edu | drop | high | "prod site under construction" placeholder | — |
| secure-test.gis.vt.edu (dup-check) | drop | high | (already listed) test GIS host | — |
| birdseye.gis.vt.edu | drop | high | Default Microsoft IIS landing page, no content | — |
| birdseye.cntrlsrvs.w2k.vt.edu (dup-check) | drop | high | (already listed) control-services infra | — |
| dashboard.arc.vt.edu | drop | high | Grafana dashboard (load error), monitoring, no public content | — |
| webapps.cee.vt.edu | full | low | CEE web-apps portal hub; readable nav/links to resources | — |
| www.webapps.cee.vt.edu | full | low | CEE portal hub (www alias); readable nav, links to apps | — |
| webapps.es.vt.edu | drop | high | Apache "It works!" default page, no content | — |
| www.apps.pamplin.vt.edu | full | low | Pamplin app catalog portal; readable app list + descriptions | — |
| register.cpe.vt.edu | drop | high | Continuing-ed registration app, transactional, error state | — |
| register.classroomav.vt.edu | drop | high | Classroom A/V access registration app, login-gated transactional | — |
| vtsa.registrar.vt.edu | drop | high | TLS cert mismatch, internal/misconfigured, not public content | — |
| www.vtsa.registrar.vt.edu | drop | high | TLS cert mismatch (www alias), not public content | — |
| mailman.music.vt.edu | drop | high | Placeholder "hello" page, no mailing-list content | — |
| cgprapps.cee.vt.edu | drop | high | CGPR web-app management shell, login-gated, no public content | — |
| www.assets.cms.vt.edu | drop | high | CMS asset/CDN host (www alias), no readable text | — |
| www.ensemble.cms.vt.edu | drop | high | Ensemble media admin (www alias), no public content | — |

## Summary

Counts per mode (all 71 input hosts; "dup-check" rows are annotations, not extra hosts):

- **drop:** 53
- **full:** 14 (registrar, www.registrar, soiltest, www.soiltest, cgit, www.cgit, cast, www.cast, ecoforecast, jobs.apply, garden.gis, webapps.cee, www.webapps.cee, www.apps.pamplin)
- **special:** 4 (engapplets.vt.edu, www.engapplets.vt.edu, engapplets.aoe.vt.edu, geovine.cgit.vt.edu)
- **catalog:** 0
- **Total:** 71

### special hosts (drive JS-rendering work)
All four are **sub-case (a) — JS/heavy-client**, none are API-better:
- `engapplets.vt.edu` — legacy frameset + Java applets; plain fetch returns frame shell
- `www.engapplets.vt.edu` — same legacy applet site (www alias)
- `engapplets.aoe.vt.edu` — legacy frameset + Java applets (AOE dept)
- `geovine.cgit.vt.edu` — interactive viticulture geospatial map app (data layers, login forms)

No sub-case (b) API hosts. The one DB/API host found (couchdb.apps.wuvt) is internal-only and dropped.
