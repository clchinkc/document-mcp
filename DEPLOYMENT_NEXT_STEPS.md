# OAuth 2.1 Implementation - Next Steps

## ✅ What Has Been Completed

### Infrastructure (COMPLETE)
- ✅ Redis instance created: `mcp-oauth-store` (Host: 10.97.160.243)
- ✅ VPC connector created: `mcp-vpc-connector`
- ✅ All infrastructure in us-central1 region

### Code Implementation (COMPLETE)
- ✅ OAuth 2.1 Authorization Server endpoints implemented in [server.py](server.py)
- ✅ PKCE validation (S256 method) with security best practices
- ✅ Redis session/token storage integration
- ✅ Updated /mcp endpoint to validate our tokens
- ✅ Dependencies updated (redis, requests)
- ✅ GitHub Actions workflow configured with VPC connector
- ✅ Code committed and pushed to GitHub

### Documentation (COMPLETE)
- ✅ [OAUTH_IMPLEMENTATION_STATUS.md](OAUTH_IMPLEMENTATION_STATUS.md) - Complete status and instructions
- ✅ [QUICK_OAUTH_FIX.md](QUICK_OAUTH_FIX.md) - Architecture explanation
- ✅ This summary document

---

## 🔴 CRITICAL: Manual Steps Required Before Deployment

The latest deployment failed because GitHub Secrets are missing. You need to complete these two steps:

### STEP 1: Add GitHub Secrets (5 minutes)

Navigate to: https://github.com/clchinkc/document-mcp/settings/secrets/actions

Click "New repository secret" and add each of these:

| Secret Name | Secret Value | Where to Find |
|-------------|--------------|---------------|
| `GOOGLE_OAUTH_CLIENT_SECRET` | `GOCSPX-...` | In `client_secret_*.json` file (repository root) → `web.client_secret` |
| `REDIS_HOST` | `10.97.160.243` | This document (see above) |
| `REDIS_PORT` | `6379` | Standard Redis port |

**To get the client secret from JSON file:**
```bash
# Run this in the repository:
cat client_secret_*.json | jq -r '.web.client_secret'
```

### STEP 2: Update Google OAuth Console (3 minutes)

Navigate to: https://console.cloud.google.com/apis/credentials?project=document-mcp-54749

1. Click on OAuth 2.0 Client ID: `451560119112-qut82fn0liagmjmg89fmgc00v9a6rg1c.apps.googleusercontent.com`
2. Under "Authorized redirect URIs", click "ADD URI"
3. Add: `https://document-mcp-451560119112.us-central1.run.app/oauth/callback`
4. Click "SAVE"

**Final redirect URIs should be (3 total):**
- `https://claude.ai/api/mcp/auth_callback` (existing)
- `https://claude.com/api/mcp/auth_callback` (existing)
- `https://document-mcp-451560119112.us-central1.run.app/oauth/callback` (NEW)

---

## After Manual Steps Are Complete

### Trigger Deployment

Once both manual steps are done, trigger a new deployment:

**Option 1: Automatic (Recommended)**
```bash
# Make a trivial change to trigger deployment
git commit --allow-empty -m "Trigger deployment after adding GitHub Secrets"
git push origin main
```

**Option 2: Manual Trigger**
- Go to: https://github.com/clchinkc/document-mcp/actions/workflows/deploy-cloud-run.yml
- Click "Run workflow" → "Run workflow"

### Monitor Deployment

```bash
# Watch the deployment status
gh run watch --repo clchinkc/document-mcp

# Or view in browser:
# https://github.com/clchinkc/document-mcp/actions
```

### Expected Deployment Output

Successful deployment should show:
```
✅ Deployment verified successfully
✅ Health check passed
✅ OAuth metadata endpoint available
⚠️  Full OAuth flow requires Claude Desktop Custom Connector
```

---

## Testing After Deployment

### Test 1: Health Check
```bash
curl https://document-mcp-451560119112.us-central1.run.app/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "mcp_server": "document-mcp",
  "tools_count": 26
}
```

### Test 2: OAuth Metadata
```bash
curl https://document-mcp-451560119112.us-central1.run.app/.well-known/oauth-authorization-server
```

**Expected Output:**
```json
{
  "issuer": "https://document-mcp-451560119112.us-central1.run.app",
  "authorization_endpoint": "https://document-mcp-451560119112.us-central1.run.app/oauth/authorize",
  "token_endpoint": "https://document-mcp-451560119112.us-central1.run.app/oauth/token",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code"],
  "code_challenge_methods_supported": ["S256"],
  "token_endpoint_auth_methods_supported": ["none"]
}
```

### Test 3: Configure Claude Desktop

After successful deployment:

1. **Open Claude Desktop Settings** → Developer → Custom Connectors
2. **Add New Connector**:
   - **Name**: Try Document MCP
   - **MCP URL**: `https://document-mcp-451560119112.us-central1.run.app/mcp`
   - **OAuth Client ID**: (Get from `client_secret_*.json` → `web.client_id`)
   - **OAuth Client Secret**: (Get from `client_secret_*.json` → `web.client_secret`)
3. **Save and Test**:
   - Claude Desktop will redirect you to Google OAuth
   - Authenticate with your Google account
   - You should be redirected back to Claude Desktop
   - Connection should show as "Connected"

### Test 4: Use MCP Tools

Once connected, try a simple command in Claude Desktop:
```
List all documents in my storage
```

Claude should use the `list_documents` tool and show your isolated document storage.

---

## Troubleshooting

### Issue: Deployment Still Fails

**Check:**
1. Verify all 3 GitHub Secrets are added correctly
2. Check secret names match exactly (case-sensitive)
3. Ensure REDIS_HOST is `10.97.160.243` (no quotes)
4. Ensure REDIS_PORT is `6379` (no quotes)

**View Logs:**
```bash
# Get the run ID from failed deployment
gh run list --limit 1 --repo clchinkc/document-mcp

# View logs
gh run view <RUN_ID> --log --repo clchinkc/document-mcp
```

### Issue: OAuth Flow Fails in Claude Desktop

**Symptoms:** Redirected to Google but returns error

**Check:**
1. Verify redirect URI is added to Google OAuth Console
2. Check Cloud Run logs for errors:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=document-mcp" \
     --project=document-mcp-54749 \
     --limit=50 \
     --format=json
   ```

### Issue: Redis Connection Timeout

**Symptoms:** Server starts but can't connect to Redis

**Check:**
1. Verify VPC connector is attached to Cloud Run service:
   ```bash
   gcloud run services describe document-mcp \
     --region=us-central1 \
     --project=document-mcp-54749 \
     --format="get(spec.template.spec.containers[0].resources)"
   ```

2. Ensure REDIS_HOST and REDIS_PORT secrets are correct

---

## Timeline Estimate

- ✅ Infrastructure: 30 minutes - **COMPLETE**
- ✅ Code implementation: 2-3 hours - **COMPLETE**
- ✅ Initial deployment config: 30 minutes - **COMPLETE**
- ⏳ **Manual configuration: 5-10 minutes** (YOU ARE HERE)
- ⏳ Deployment retry: 2-3 minutes
- ⏳ Testing: 15-30 minutes
- ⏳ Documentation updates: 15-30 minutes (optional)

**Total**: ~4.5 hours completed, ~30-60 minutes remaining

---

## Summary

**What You Need to Do Right Now:**

1. ✅ Add 3 GitHub Secrets (5 minutes)
2. ✅ Update Google OAuth Console redirect URI (3 minutes)
3. ✅ Trigger new deployment (1 minute)
4. ✅ Wait for deployment to complete (2-3 minutes)
5. ✅ Test with Claude Desktop (5-10 minutes)

**Total Time**: ~15-20 minutes to get everything working

---

## Quick Reference

**Redis Host**: `10.97.160.243`
**Redis Port**: `6379`
**Server URL**: `https://document-mcp-451560119112.us-central1.run.app`
**OAuth Client ID**: (In `client_secret_*.json`)
**OAuth Client Secret**: (In `client_secret_*.json`)
**Redirect URI to Add**: `https://document-mcp-451560119112.us-central1.run.app/oauth/callback`

---

**Questions?** Check:
- [OAUTH_IMPLEMENTATION_STATUS.md](OAUTH_IMPLEMENTATION_STATUS.md) - Detailed implementation status
- [QUICK_OAUTH_FIX.md](QUICK_OAUTH_FIX.md) - OAuth architecture explanation
- [docs/OAUTH_2.1_ECOSYSTEM.md](docs/OAUTH_2.1_ECOSYSTEM.md) - Complete OAuth 2.1 technical spec
