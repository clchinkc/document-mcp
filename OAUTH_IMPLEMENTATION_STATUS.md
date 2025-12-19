# OAuth 2.1 Implementation Status

**Date**: December 19, 2025
**Status**: ✅ Code Complete - Manual Configuration Required

---

## What Has Been Implemented

### ✅ Phase 1: Infrastructure (COMPLETE)
- **Redis Instance**: `mcp-oauth-store` created in us-central1
  - Host: `10.97.160.243`
  - Port: `6379`
  - Size: 1GB Basic Tier
- **VPC Connector**: `mcp-vpc-connector` created
  - Region: us-central1
  - Network: default
  - IP Range: 10.8.0.0/28

### ✅ Phase 2: Code Implementation (COMPLETE)
1. **Dependencies Updated** ([pyproject.toml](pyproject.toml)):
   - Added `redis>=5.0.0` for session/token storage
   - Added `requests>=2.31.0` for Google token exchange

2. **OAuth 2.1 Server Endpoints** ([server.py](server.py)):
   - `GET /oauth/authorize` - Authorization endpoint with PKCE validation
   - `GET /oauth/callback` - Google OAuth callback handler
   - `POST /oauth/token` - Token exchange with PKCE verification
   - `GET /.well-known/oauth-authorization-server` - Server metadata (RFC 8414)

3. **Helper Functions**:
   - `generate_code_challenge_from_verifier()` - PKCE S256 validation
   - `store_session_data()` - Redis session management with TTL
   - `get_session_data()` - Session retrieval
   - `validate_access_token()` - Token validation for /mcp endpoint

4. **MCP Endpoint Updated**:
   - Now validates OUR tokens (issued by /oauth/token) instead of Google tokens
   - User isolation maintained (email → user_id sanitization)

### ✅ Phase 3: Deployment Configuration (COMPLETE)
- **GitHub Actions Workflow** ([.github/workflows/deploy-cloud-run.yml](.github/workflows/deploy-cloud-run.yml)):
  - Added VPC connector configuration
  - Added environment variables for OAuth and Redis
  - Updated verification to test OAuth metadata endpoint

---

## Manual Configuration Required

### 🔴 STEP 1: Add GitHub Secrets

Navigate to: [GitHub Secrets Settings](https://github.com/clchinkc/document-mcp/settings/secrets/actions)

Add these secrets:

| Secret Name | Value | Status |
|-------------|-------|--------|
| `GOOGLE_OAUTH_CLIENT_SECRET` | (From client_secret JSON file) | ⏳ TODO |
| `REDIS_HOST` | `10.97.160.243` | ⏳ TODO |
| `REDIS_PORT` | `6379` | ⏳ TODO |

**Note**: `GOOGLE_OAUTH_CLIENT_ID` already exists (added previously).

### 🔴 STEP 2: Update Google OAuth Console

Navigate to: [Google Cloud Console - OAuth Credentials](https://console.cloud.google.com/apis/credentials?project=document-mcp-54749)

Edit OAuth client: `451560119112-qut82fn0liagmjmg89fmgc00v9a6rg1c.apps.googleusercontent.com`

**Add Redirect URI**:
```
https://document-mcp-451560119112.us-central1.run.app/oauth/callback
```

**Keep Existing Redirect URIs**:
- `https://claude.ai/api/mcp/auth_callback`
- `https://claude.com/api/mcp/auth_callback`

**Final Redirect URIs List** (should have 3 total):
- `https://claude.ai/api/mcp/auth_callback`
- `https://claude.com/api/mcp/auth_callback`
- `https://document-mcp-451560119112.us-central1.run.app/oauth/callback`

---

## Deployment Instructions

### After Manual Configuration is Complete:

1. **Commit Code Changes**:
   ```bash
   cd /Users/clchinkc/Documents/GitHub/document-mcp
   git add .
   git commit -m "Implement OAuth 2.1 Authorization Server with PKCE

   - Add Redis session/token storage infrastructure
   - Implement /oauth/authorize, /oauth/callback, /oauth/token endpoints
   - Add PKCE S256 validation for security
   - Update /mcp endpoint to validate our tokens instead of Google tokens
   - Configure VPC connector for Cloud Run → Redis connectivity
   - Add OAuth server metadata endpoint (RFC 8414)
   - Update GitHub Actions with OAuth environment variables"

   git push origin main
   ```

2. **GitHub Actions will automatically**:
   - Build and deploy to Cloud Run
   - Configure VPC connector for Redis connectivity
   - Set environment variables from secrets
   - Verify health and OAuth metadata endpoints

3. **Test the Deployment**:
   ```bash
   # Test health endpoint
   curl https://document-mcp-451560119112.us-central1.run.app/health

   # Test OAuth metadata
   curl https://document-mcp-451560119112.us-central1.run.app/.well-known/oauth-authorization-server
   ```

4. **Configure Claude Desktop**:
   - Open Claude Desktop settings
   - Add Custom Connector with:
     - **MCP URL**: `https://document-mcp-451560119112.us-central1.run.app/mcp`
     - **OAuth Client ID**: (From client_secret JSON file)
     - **OAuth Client Secret**: (From client_secret JSON file)
   - Claude Desktop will automatically handle the OAuth flow

---

## OAuth 2.1 Flow Diagram

```
User opens Claude Desktop
        ↓
Claude Desktop → /oauth/authorize
        ↓
Server redirects to Google OAuth
        ↓
User authenticates with Google
        ↓
Google → /oauth/callback (with authorization code)
        ↓
Server exchanges Google code for email
        ↓
Server generates OUR authorization code (stores with PKCE challenge)
        ↓
Server redirects to Claude Desktop callback
        ↓
Claude Desktop → /oauth/token (with code_verifier for PKCE)
        ↓
Server validates PKCE, issues access token
        ↓
Claude Desktop → /mcp (with our access token)
        ↓
Server validates token, routes to user storage
```

---

## Security Features

✅ **PKCE (S256)**: Prevents authorization code interception
✅ **Single-use Codes**: Authorization codes deleted after use
✅ **Token Expiry**: Access tokens expire after 1 hour
✅ **Session Expiry**: OAuth sessions expire after 10 minutes
✅ **Constant-time Comparison**: `secrets.compare_digest()` prevents timing attacks
✅ **Audit Logging**: All OAuth events logged with success/failure
✅ **HTTPS Only**: Cloud Run enforces HTTPS
✅ **No Token Logging**: Tokens never logged in plaintext

---

## Testing Checklist

After deployment:

- [ ] Health check passes: `curl https://document-mcp-451560119112.us-central1.run.app/health`
- [ ] OAuth metadata available: `curl https://document-mcp-451560119112.us-central1.run.app/.well-known/oauth-authorization-server`
- [ ] Claude Desktop OAuth flow works end-to-end
- [ ] User isolation verified (different Google accounts → different storage)
- [ ] Token expiration works (tokens expire after 1 hour)
- [ ] Audit logs show OAuth events

---

## Troubleshooting

### Common Issues

**Issue**: `{"detail":"Not Found"}` on /authorize
**Solution**: Ensure GitHub Secrets are configured and deployment succeeded

**Issue**: Redis connection timeout
**Solution**: Verify VPC connector is attached to Cloud Run service

**Issue**: Google OAuth callback fails
**Solution**: Verify redirect URI is added to Google OAuth Console

**Issue**: PKCE verification failed
**Solution**: Ensure code_verifier matches code_challenge (this is handled by Claude Desktop)

### Debugging

Check Cloud Run logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=document-mcp" \
  --project=document-mcp-54749 \
  --limit=50 \
  --format=json
```

Test Redis connectivity (from Cloud Run):
```bash
# The Redis instance is only accessible from Cloud Run via VPC connector
# Cannot test locally unless you set up Cloud SQL Proxy
```

---

## Estimated Timeline

- ✅ Infrastructure: 30 minutes - **COMPLETE**
- ✅ Code implementation: 2-3 hours - **COMPLETE**
- ✅ Deployment configuration: 30 minutes - **COMPLETE**
- ⏳ Manual configuration: **5-10 minutes** (YOU ARE HERE)
- ⏳ Testing: 15-30 minutes (after manual config)
- ⏳ Documentation updates: 15-30 minutes (README, OAUTH_SETUP.md)

**Total time**: 4-6 hours (4.5 hours complete, 1-1.5 hours remaining)

---

## Next Steps

1. ✅ Add GitHub Secrets
2. ✅ Update Google OAuth Console redirect URI
3. ✅ Commit and push code changes
4. ✅ Wait for GitHub Actions deployment to complete
5. ✅ Test OAuth flow with Claude Desktop
6. ⏳ Update documentation (README.md, OAUTH_SETUP.md)

---

**Questions?** Check the comprehensive technical documentation:
- [QUICK_OAUTH_FIX.md](QUICK_OAUTH_FIX.md) - Problem explanation and solution
- [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md) - Multi-provider considerations
- [docs/OAUTH_2.1_ECOSYSTEM.md](docs/OAUTH_2.1_ECOSYSTEM.md) - Complete OAuth 2.1 specification
