# TODO — tugamorph

## Open issues

None open.

## Gaps

- [ ] CI missing standard gh-automations `build-tests` workflow (tests never run in CI; ~250 unittest tests exist locally).
- [ ] CI missing `coverage` workflow.
- [ ] CI missing `license-check` workflow.
- [ ] gh-automations reusable workflows in `release_workflow.yml` and `publish_stable.yml` are pinned at `@master`; org convention is `@dev`.
- [ ] No `LICENSE` file. README states MIT; `setup.py` `license=''` is empty.
- [ ] `setup.py` `description=''` is empty (real description lives only in README).
- [ ] Committed scratch artifact: `tugamorph.egg-info/` is tracked in the repo.
- [ ] No `pyproject.toml`; packaging is legacy `setup.py` only.

## Code TODOs

None found.
