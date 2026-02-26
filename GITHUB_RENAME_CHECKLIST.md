# GitHub Repository Rename Checklist: document-mcp → story-mcp

## Current Status

**Repository Name**: Still `document-mcp` at GitHub (https://github.com/clchinkc/document-mcp)
**Package Name**: Already renamed to `story-mcp` in code (v0.0.5)
**Status**: Code rename COMPLETE ✅ | GitHub repo rename PENDING ⏳

---

## Summary Overview

| Aspect | Status | Details |
|--------|--------|---------|
| Code package rename | ✅ DONE | `story_mcp/` directory created, `document_mcp/` deprecated |
| Backward compatibility | ✅ DONE | 6-month deprecation layer implemented |
| PyPI package | ✅ READY | `story-mcp` published on PyPI |
| GitHub repo name | ⏳ PENDING | Still `document-mcp`, needs manual rename |
| Documentation | ✅ DONE | README and CLAUDE.md updated |
| Workflows | ✅ READY | Uses relative paths, will auto-update |

---

## Already Completed in Code (v0.0.5)

### 1. Package Rename ✅
- **Main package**: `story_mcp/` created (22 subdirectories)
- **Backward compat**: `document_mcp/` exists as deprecation layer
- **CLI entry point**: Dual commands available
  - New: `story-mcp` (recommended)
  - Old: `document-mcp` (deprecated, redirects to story-mcp)

### 2. PyPI Updates ✅
- **Package name**: `story-mcp` (in `pyproject.toml`)
- **Homepage**: Updated to `https://github.com/clchinkc/story-mcp`
- **Bug Tracker**: Updated to `https://github.com/clchinkc/story-mcp/issues`
- **CLI entry points**: Both `story-mcp` and `document-mcp` commands

### 3. Documentation Updates ✅
- **README.md**:
  - Title changed to "Story MCP"
  - Install command: `pip install story-mcp`
  - Clone command: `git clone https://github.com/clchinkc/story-mcp.git`
  - GitHub badges: Updated to `clchinkc/story-mcp`
- **CLAUDE.md**: Updated with Phase 4 completion, renamed package info
- **pyproject.toml**:
  - `name = "story-mcp"`
  - URLs point to `/story-mcp/`
  - Package list includes both `story_mcp` and backward compat layer

### 4. Deprecation Layer ✅
- **`document_mcp/__init__.py`**: Deprecation warning on import
- **`document_mcp/legacy.py`**: Backward compatibility shim
- **Sunset date**: v1.0.0 (August 2026 - 6 months from v0.0.5)

---

## Manual GitHub Steps Required

GitHub repository rename requires manual steps in GitHub's web interface. The rename is a critical infrastructure change that affects:

1. **Repository Settings** → Rename the repository
2. **Automatic Redirects** → GitHub provides 1-year redirect
3. **Branch Protection Rules** → Update references to old name (auto-updated)
4. **Webhooks** → Update if using external services
5. **Local Clones** → Users must update remote URLs

---

## Step-by-Step Checklist

### Phase 1: Pre-Rename Verification (5 minutes)

- [ ] **Step 1.1**: Confirm current repo structure
  ```bash
  git remote -v
  # Should show: origin	https://github.com/clchinkc/document-mcp.git
  ```

- [ ] **Step 1.2**: Verify no open PRs with conflicting names
  - Go to https://github.com/clchinkc/document-mcp/pulls
  - Note: Open PRs will still work after rename

- [ ] **Step 1.3**: Document current GitHub Actions status
  - Check: `.github/workflows/python-test.yml`
  - Check: `.github/workflows/release.yml`
  - Check: `.github/workflows/deploy-cloud-run.yml`

- [ ] **Step 1.4**: Export any branch protection settings
  - Go to Settings → Branches
  - Note current rules for `main` branch

### Phase 2: GitHub Web Interface Rename (3 minutes)

- [ ] **Step 2.1**: Go to repository settings
  - URL: https://github.com/clchinkc/document-mcp/settings
  - OR: Click "Settings" tab in repo

- [ ] **Step 2.2**: Rename repository
  - Section: "Danger Zone" (bottom of General settings)
  - Current name field: `document-mcp`
  - New name field: `story-mcp`
  - Click "Rename"

- [ ] **Step 2.3**: Verify rename success
  - New URL: https://github.com/clchinkc/story-mcp
  - Old URL: https://github.com/clchinkc/document-mcp (should redirect)
  - Check that redirect works (wait 5 seconds)

### Phase 3: Update Local Repository (5 minutes)

- [ ] **Step 3.1**: Update git remote
  ```bash
  cd /Users/clchinkc/Documents/GitHub/document-mcp
  git remote set-url origin https://github.com/clchinkc/story-mcp.git
  git remote -v
  # Verify: origin	https://github.com/clchinkc/story-mcp.git
  ```

- [ ] **Step 3.2**: Verify local branch configuration
  ```bash
  git config branch.main.remote
  # Should still be: origin
  git config branch.main.merge
  # Should still be: refs/heads/main
  ```

- [ ] **Step 3.3**: Test connection to new remote
  ```bash
  git fetch origin
  # Should succeed without errors
  ```

### Phase 4: Update GitHub Actions & Settings (5 minutes)

- [ ] **Step 4.1**: Review workflow files (verify they still work)
  - File: `.github/workflows/python-test.yml`
    - ✅ GOOD: Uses relative paths, no hardcoded repo names
    - ✅ GOOD: Badge URL references `clchinkc/story-mcp`

  - File: `.github/workflows/release.yml`
    - ✅ GOOD: Uses event-driven triggers, no hardcoded names

  - File: `.github/workflows/deploy-cloud-run.yml`
    - ✅ GOOD: Standard GCP deployment, no repo name dependency

- [ ] **Step 4.2**: Verify codecov configuration
  - Badge URL in README: `https://codecov.io/gh/clchinkc/story-mcp`
  - ✅ Already updated
  - Action workflow sends coverage to: Codecov (no manual update needed)

- [ ] **Step 4.3**: Check branch protection rules
  - Go to: Settings → Branches → Branch protection rules
  - Verify `main` branch protection still active
  - Rule references should auto-update

- [ ] **Step 4.4**: Update GitHub Pages (if enabled)
  - Go to: Settings → Pages
  - Verify source branch is correct
  - If using custom domain, verify DNS still points to GitHub

### Phase 5: Update External Services (10 minutes)

- [ ] **Step 5.1**: PyPI Project
  - Status: ✅ ALREADY CORRECT
  - Project URL: https://pypi.org/project/story-mcp/
  - Homepage URL: Points to `/story-mcp/`
  - No action needed

- [ ] **Step 5.2**: Codecov
  - Status: ✅ ALREADY CORRECT
  - Coverage page: https://codecov.io/gh/clchinkc/story-mcp
  - CODECOV_TOKEN secret in GitHub: Still valid
  - No action needed

- [ ] **Step 5.3**: GitHub Secrets & Variables
  - Go to: Settings → Secrets and variables → Actions
  - Verify all secrets are present:
    - [ ] `PYPI_API_TOKEN` (for release.yml)
    - [ ] `GEMINI_API_KEY` (for e2e tests)
    - [ ] `OPENAI_API_KEY` (for e2e tests)
    - [ ] `CODECOV_TOKEN` (for coverage)
  - All secrets automatically carried over ✅

- [ ] **Step 5.4**: Any webhooks (if configured)
  - Go to: Settings → Webhooks
  - Verify no webhooks point to old URL
  - If any exist, note the payload URLs and test after rename

### Phase 6: Documentation Update (5 minutes)

- [ ] **Step 6.1**: Verify README is correct
  - Main clone command: `git clone https://github.com/clchinkc/story-mcp.git`
  - ✅ Already updated in code

- [ ] **Step 6.2**: Check CLAUDE.md references
  - Repository clone references
  - ✅ Already updated in code

- [ ] **Step 6.3**: Update docs/ directory
  - File: `docs/manual_testing.md`
    - Current: `git clone https://github.com/clchinkc/document-mcp.git`
    - [ ] Update to: `git clone https://github.com/clchinkc/story-mcp.git`

- [ ] **Step 6.4**: Check for other clone instructions
  - Search for any other references to old repo URL
  - Update if found

### Phase 7: Validation & Testing (10 minutes)

- [ ] **Step 7.1**: Test old URL redirect
  ```bash
  curl -I https://github.com/clchinkc/document-mcp
  # Should return 301 with Location: https://github.com/clchinkc/story-mcp
  ```

- [ ] **Step 7.2**: Test new URL
  ```bash
  curl -I https://github.com/clchinkc/story-mcp
  # Should return 200 OK
  ```

- [ ] **Step 7.3**: Test git operations
  ```bash
  cd /tmp && git clone https://github.com/clchinkc/story-mcp.git test-story-mcp
  # Should succeed
  cd test-story-mcp && git log --oneline | head -3
  # Verify history is intact
  ```

- [ ] **Step 7.4**: Verify GitHub Actions trigger
  - Make a minor commit and push to verify workflows
  - Check: https://github.com/clchinkc/story-mcp/actions
  - Verify workflows run successfully

- [ ] **Step 7.5**: Verify PyPI install still works
  ```bash
  pip install --upgrade story-mcp
  # Should install successfully from new repo
  ```

### Phase 8: Post-Rename Communication (5 minutes)

- [ ] **Step 8.1**: Update any external documentation
  - Any guides referencing old URL
  - Any blog posts or announcements

- [ ] **Step 8.2**: Notify users (optional)
  - GitHub Release: Mention rename in next release notes
  - Old URL will redirect for 1 year

- [ ] **Step 8.3**: Archive old branch if needed
  - Any feature branches pointing to old URL
  - ✅ Not needed (local branches auto-redirect)

---

## Files to Update After Rename

### 1. Update docs/manual_testing.md

**Location**: `/Users/clchinkc/Documents/GitHub/document-mcp/docs/manual_testing.md`

Search for and update:
```bash
# OLD:
git clone https://github.com/clchinkc/document-mcp.git

# NEW:
git clone https://github.com/clchinkc/story-mcp.git
```

### 2. Verify GitHub workflow paths

- `.github/workflows/python-test.yml` - ✅ Uses relative paths
- `.github/workflows/release.yml` - ✅ Uses event triggers
- `.github/workflows/deploy-cloud-run.yml` - ✅ Uses GCP service

All workflows already compatible with rename (no hardcoded repo names).

---

## Risk Assessment

| Risk | Severity | Mitigation | Status |
|------|----------|-----------|--------|
| Breaking user clones | HIGH | GitHub provides 1-year redirect | ✅ Handled |
| CI/CD failures | MEDIUM | Workflows use relative paths | ✅ Safe |
| PyPI package URL | LOW | Already points to `/story-mcp/` | ✅ Ready |
| Secret token access | NONE | Secrets auto-transferred | ✅ Secure |
| Codecov integration | LOW | Coverage token still valid | ✅ Verified |
| Open PRs/Issues | LOW | Auto-redirect in URLs | ✅ Compatible |

---

## Time Estimate

| Phase | Task | Time |
|-------|------|------|
| 1 | Pre-rename verification | 5 min |
| 2 | GitHub web interface rename | 3 min |
| 3 | Local repo update | 5 min |
| 4 | GitHub Actions & settings | 5 min |
| 5 | External services | 10 min |
| 6 | Documentation update | 5 min |
| 7 | Validation & testing | 10 min |
| 8 | Post-rename communication | 5 min |
| **TOTAL** | **Complete migration** | **48 min** |

---

## What's Already Done (Code Level)

✅ Package rename: `story_mcp/` created
✅ Deprecation layer: `document_mcp/` with warnings
✅ PyPI metadata: Points to new repo
✅ CLI commands: Both `story-mcp` and `document-mcp` available
✅ README: Updated with new URLs
✅ CLAUDE.md: Updated with new package name
✅ Backward compatibility: Full 6-month deprecation period
✅ CI/CD: Already uses relative paths

---

## What Needs Manual Action (GitHub Level)

⏳ Rename repo on GitHub (Phase 2)
⏳ Update local remote URL (Phase 3)
⏳ Update docs/manual_testing.md (Phase 6)
⏳ Verify workflows run after rename (Phase 7)
⏳ Test old URL redirect (Phase 7)

---

## Post-Rename Validation Checklist

After completing all steps above, verify:

```bash
# Test 1: New URL works
curl https://github.com/clchinkc/story-mcp

# Test 2: Old URL redirects
curl https://github.com/clchinkc/document-mcp

# Test 3: Clone works
git clone https://github.com/clchinkc/story-mcp.git /tmp/test-story-mcp
cd /tmp/test-story-mcp
git log --oneline | head -5

# Test 4: PyPI package is correct
pip show story-mcp | grep "Home-page"

# Test 5: GitHub Actions work
# Check: https://github.com/clchinkc/story-mcp/actions

# Test 6: Codecov integration
# Check: https://codecov.io/gh/clchinkc/story-mcp
```

---

## Troubleshooting

### If workflows fail after rename:
- Check GitHub Actions logs: Settings → Actions → General
- Most likely cause: None (workflows use relative paths)
- Fix: Re-run failed jobs

### If PyPI integration breaks:
- Verify PYPI_API_TOKEN is present in Secrets
- Check release.yml workflow has correct permissions
- PyPI should auto-update based on git tags

### If codecov fails:
- Verify CODECOV_TOKEN in Secrets
- Coverage action uses repo slug, which should auto-update
- Force resync: Delete and re-enable Codecov integration

### If git push fails:
- Error: "The repository moved" or "Repository not found"
- Fix: Run `git remote set-url origin https://github.com/clchinkc/story-mcp.git`

---

## References

- GitHub docs on renaming: https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository
- PyPI package: https://pypi.org/project/story-mcp/
- Codecov docs: https://docs.codecov.io/docs

---

**Generated**: February 26, 2026
**Status**: Ready for execution
**Estimated Duration**: 48 minutes
**Risk Level**: LOW (GitHub provides 1-year redirect, code already prepared)
