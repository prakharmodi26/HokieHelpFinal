# Lane 1 — Library / Repositories / Archives — tiering
| host | mode | confidence | reason | notes (api/js) |
|------|------|-----------|--------|----------------|
| vtechworks.lib.vt.edu | catalog | high | DSpace 7+ institutional repository: theses, dissertations, papers, ETDs | DSpace REST /server/api; OAI-PMH /server/oai/request?verb=Identify; some JS-loaded recent items |
| etd.lib.vt.edu | full | high | 301 redirects to guides.lib.vt.edu/ETDguide — now just an ETD how-to guide page | redirect to LibGuide; actual ETD content lives in vtechworks |
| imagebase.lib.vt.edu | drop | high | 301 redirects to digitalsc.lib.vt.edu; legacy imagebase merged there | duplicate/superseded host |
| omeka.lib.vt.edu | drop | high | 301 redirects to digitalsc.lib.vt.edu; duplicate of special collections host | duplicate/superseded host |
| digitalsc.lib.vt.edu | catalog | med | Omeka special-collections/digital archive; image+text objects with metadata | Omeka; metadata-only cards; exhibits sub-paths should be dropped |
| scholar.lib.vt.edu | full | med | Scholarly Communication dept info hub (services, OA funding, copyright) | content site, points users to vtechworks |
| opentext.lib.vt.edu | drop | high | 301 redirects off-domain to pressbooks.pub/thermo (external host) | not a vt.edu page; out of scope |
| lib.vt.edu | full | high | Main University Libraries site: hours, services, news, guides | server HTML + JS enhancements |
| www.lib.vt.edu | full | high | www alias of main University Libraries site | same as apex |
| exhibits.lib.vt.edu | drop | high | Digital exhibits host; 302s to lib.vt.edu exhibits index — exhibits dropped by owner policy | exhibit gateway |
| www.opentext.lib.vt.edu | drop | high | www alias of opentext; redirects off-domain to pressbooks.pub | external host, out of scope |
| spec.lib.vt.edu | full | high | Special Collections landing/service page (hours, finding-aid links) | gateway; objects live in digitalsc/ARVAS |
| data.lib.vt.edu | catalog | low | Research-data host (name); returns 403 to fetch, likely a Dataverse/data repo or data-services page | 403 blocked; if Dataverse use native API; verify manually |
| april16documentarchive.vt.edu | drop | high | Original April 16 archive GONE; now a parked "Discovery Compass" e-discovery login (Servient, 2004-08) | TLS cert mismatch on apex; dead/repurposed host |
| www.april16documentarchive.vt.edu | drop | high | Same parked Discovery Compass login wall; original archive defunct | login-walled, not VT content |
| news.lib.vt.edu | full | high | 301 redirects to lib.vt.edu/news — library news articles | covered by lib.vt.edu crawl |
| cdrs.lib.vt.edu | full | high | 301 redirects to lib.vt.edu CDRS service page | content/service page |
| dhomeka.lib.vt.edu | drop | high | Archived Omeka digital-humanities exhibits (Lost Communities, VA Speedways); no longer updated | exhibits — owner drops; Omeka |
| flu1918.lib.vt.edu | full | high | WordPress research-project site (Epidemiology of Information); readable scholarly text | WordPress; publications/findings text |
| vpiworldwarone.lib.vt.edu | catalog | med | Omeka searchable database of 1000+ VT WWI veterans (names, bios) + an exhibit | catalog the veteran records; Omeka; exhibit sub-paths droppable |
| www.vpiworldwarone.lib.vt.edu | catalog | med | www alias of vpiworldwarone veteran database | same as apex |
| roanokevalleygreatwar.lib.vt.edu | catalog | med | Archived Omeka searchable database of 3000+ Roanoke WWI veterans | catalog records; Omeka; archived but accessible |
| odyssey.lib.vt.edu | full | med | Educational-resource repository: lesson plans, handouts, infographics, videos | readable instructional content |
| www.odyssey.lib.vt.edu | full | med | www alias of odyssey educational resource repo | same as apex |
| pharos.lib.vt.edu | drop | high | Pharos Print Center — secure print-management login gateway, not library content | login-walled; CAS auth |
| rosetta.lib.vt.edu | full | high | 301 redirects to guides.lib.vt.edu LibGuide for Rosetta Stone language software | a guide page, not a preservation repo |
| www.rosetta.lib.vt.edu | full | high | www alias; same Rosetta Stone LibGuide redirect | guide page |
| dice.lib.vt.edu | full | high | 301 redirects to lib.vt.edu data-services page | service/content page |
| gromit.dlib.vt.edu | drop | high | Serves only the default nginx welcome page — parked/unconfigured server | dead host |
| info-skills.lib.vt.edu | full | high | 301 redirects to guides.lib.vt.edu getting-started info-literacy guide | LibGuide instructional content |
| www.info-skills.lib.vt.edu | full | high | www alias; same getting-started LibGuide | LibGuide |
| dsp.lib.vt.edu | full | low | Library digital-services/studio host (name); returns 403 to fetch | 403 blocked; verify manually |
| datavizstudio.lib.vt.edu | full | high | 301 redirects to lib.vt.edu data-viz studio service page | content covered by lib.vt.edu |
| designstudio.lib.vt.edu | full | high | 301 redirects to lib.vt.edu 3D design studio service page | content covered by lib.vt.edu |
| fusionstudio.lib.vt.edu | full | high | 301 redirects to lib.vt.edu fusion studio service page | content covered by lib.vt.edu |
| dtstudio.lib.vt.edu | full | high | 301 redirects to lib.vt.edu data consulting lab service page | content covered by lib.vt.edu |
| virtualstudio.lib.vt.edu | full | high | 301 redirects to lib.vt.edu visualization/VES studio service page | content covered by lib.vt.edu |
| thinkandcode.lib.vt.edu | full | med | Readable info page for a 2019 ML/philosophy summer workshop | static event content; standalone |
| informaticslab.lib.vt.edu | full | high | 301 redirects to lib.vt.edu data-services page | content covered by lib.vt.edu |
| theses.lib.vt.edu | catalog | high | 301 redirects into VTechWorks ETD collection (handle/10919/5534) | metadata-only; harvest via VTechWorks DSpace/OAI |
| kiosk.lib.vt.edu | drop | high | LibCal booking/kiosk hub (gaming, study rooms, scanning) — JS app, no readable corpus text | heavy JS booking tool |
| tour.lib.vt.edu | full | med | University Libraries tour landing/service page | thin content page |
| www.tour.lib.vt.edu | full | med | www alias of library tour page | same as apex |
| lts.lib.vt.edu | drop | high | 301 redirects off-domain to libff.questionpro.com survey | external survey, not VT content |
| rms.lib.vt.edu | full | high | 301 redirects to lib.vt.edu/urig (undergrad research info group page) | content covered by lib.vt.edu |
| static.lib.vt.edu | drop | high | Static asset/CDN host (images/css/js); returns 403, no readable pages | asset host, no corpus value |
| libx.cs.vt.edu | drop | high | Connection refused — LibX browser-extension project defunct/dead host | ECONNREFUSED |
| wordpress.cs.vt.edu | drop | med | CS WordPress multisite admin/landing hub (infra notices, login) — no coherent corpus | multisite gateway; thin/admin content |
| archive.aoe.vt.edu | drop | high | Returns HTTP 404 at root — dead/empty archive host | no content served |
| aoir.archives.cddc.vt.edu | drop | med | AoIR/CDDC archive host returns 404; socket-close on www — defunct | dead/unreachable; was an old internet-research archive |
| www.aoir.archives.cddc.vt.edu | drop | med | Same AoIR/CDDC archive; connection unstable / 404 | dead/unreachable |
| secure-archive.gis.vt.edu | drop | med | "secure-archive" GIS data host returns 403 — login-walled data repository | auth-walled; no public readable text |
| schevreporting.aie.vt.edu | full | high | Static SCHEV competency-assessment reporting portal; readable report pages | no login; CMS content |
| www.schevreporting.aie.vt.edu | full | high | www alias of SCHEV reporting portal | same as apex |
| liberalarts.vt.edu | full | high | College of Liberal Arts & Human Sciences (CLAHS) institutional content site | news, programs, research highlights |
| www.liberalarts.vt.edu | full | high | www alias of CLAHS college site | same as apex |
| memalpha.lib.vt.edu | drop | low | Returned empty body twice; likely an archived/empty wiki-style exhibit host | unresolved — verify manually; treat as exhibit/dead |
| www.exhibits.lib.vt.edu | drop | high | www alias of exhibits host (302s to lib.vt.edu exhibits index); digital exhibits dropped by owner policy | exhibit gateway |

## Summary

Counts per mode (56 input hosts):
- **full** — 30: lib (+www), spec, scholar, news, cdrs, flu1918, odyssey (+www), dice, info-skills (+www), dsp, datavizstudio, designstudio, fusionstudio, dtstudio, virtualstudio, thinkandcode, informaticslab, rms, tour (+www), schevreporting (+www), liberalarts (+www), etd, rosetta (+www)
- **catalog** — 6: vtechworks, theses, vpiworldwarone (+www), roanokevalleygreatwar, data (low-conf)
- **drop** — 20: imagebase, omeka, opentext (+www), april16documentarchive (+www), dhomeka, pharos, gromit, kiosk, lts, static, libx.cs, wordpress.cs, archive.aoe, aoir.archives.cddc (+www), secure-archive.gis, memalpha, exhibits, www.exhibits
- **special** — 0 (VTechWorks' DSpace REST + OAI-PMH endpoint is noted inline on its catalog row rather than as a separate special tier)
- (1 extra informational row, `digitalsc.lib.vt.edu` = catalog, is included as the live redirect target of imagebase/omeka but is not one of the 56 input hosts.)

Hosts I could NOT fully resolve (verify manually):
- `data.lib.vt.edu` — 403; tiered **catalog** (low) on the assumption it's a Dataverse/data repo. If Dataverse, harvest via its native API; if just a data-services page, retier **full**.
- `dsp.lib.vt.edu` — 403; tiered **full** (low) by name (digital-services/studio). Confirm.
- `secure-archive.gis.vt.edu` — 403; tiered **drop** assuming auth-walled GIS data.
- `static.lib.vt.edu` — 403; tiered **drop** as an asset/CDN host.
- `memalpha.lib.vt.edu` — empty body; tiered **drop** as an archived/empty exhibit.

Note on `exhibits.lib.vt.edu`: the apex 302-redirects to the lib.vt.edu exhibits index; per the owner-drops-all-exhibits rule both apex and www are **drop**.
