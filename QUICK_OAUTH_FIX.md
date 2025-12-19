# Quick OAuth Fix for Claude Desktop

**Issue**: Claude Desktop expects server to be OAuth authorization server, not just resource server.

**Error**: `{"detail":"Not Found"}` on `/authorize` endpoint

---

## Understanding the Problem

**Current Architecture** (WRONG):
```
Claude Desktop → Sends Google OAuth token → Server validates → Extract user
```

**Required Architecture** (CORRECT):
```
Claude Desktop → Requests /authorize → Server redirects to Google
                                     ↓
User signs in with Google ← Server exchanges code → Server issues OWN token
                                     ↓
Claude Desktop → Sends OUR token → Server validates OUR token → Extract user
```

**Key Insight**: Claude Desktop needs YOUR server to BE the OAuth provider, wrapping Google OAuth.

---

## Solution 1: Implement OAuth 2.1 Server (3-4 hours)

### Required Endpoints

```python
# Add to server.py

@app.get("/oauth/authorize")
async def oauth_authorize(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str,
    scope: str
):
    """OAuth authorization endpoint - redirects to Google"""

    # 1. Validate parameters
    if response_type != "code":
        raise HTTPException(400, "Only authorization_code flow supported")

    if code_challenge_method != "S256":
        raise HTTPException(400, "Only S256 PKCE method supported")

    # 2. Store PKCE challenge and state in session
    session_id = secrets.token_urlsafe(32)
    await redis_client.setex(
        f"oauth:session:{session_id}",
        600,  # 10 minute expiry
        json.dumps({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "state": state,
            "scope": scope
        })
    )

    # 3. Redirect to Google OAuth
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/auth"
        f"?client_id={os.environ['GOOGLE_OAUTH_CLIENT_ID']}"
        f"&redirect_uri={os.environ['SERVER_URL']}/oauth/callback"
        f"&response_type=code"
        f"&scope=openid email profile"
        f"&state={session_id}"
    )

    return RedirectResponse(google_auth_url)


@app.get("/oauth/callback")
async def oauth_callback(code: str, state: str):
    """Google OAuth callback - exchanges code for token"""

    # 1. Get session data
    session_data = await redis_client.get(f"oauth:session:{state}")
    if not session_data:
        raise HTTPException(400, "Invalid or expired session")

    session = json.loads(session_data)

    # 2. Exchange Google code for token
    google_token_response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": os.environ['GOOGLE_OAUTH_CLIENT_ID'],
            "client_secret": os.environ['GOOGLE_OAUTH_CLIENT_SECRET'],
            "redirect_uri": f"{os.environ['SERVER_URL']}/oauth/callback",
            "grant_type": "authorization_code"
        }
    )

    google_token = google_token_response.json()

    # 3. Verify Google token and extract email
    idinfo = id_token.verify_oauth2_token(
        google_token['id_token'],
        google_requests.Request(),
        os.environ['GOOGLE_OAUTH_CLIENT_ID']
    )
    user_email = idinfo.get('email')

    # 4. Generate OUR authorization code
    our_auth_code = secrets.token_urlsafe(32)

    # 5. Store authorization code with PKCE challenge
    await redis_client.setex(
        f"oauth:code:{our_auth_code}",
        600,  # 10 minute expiry
        json.dumps({
            "user_email": user_email,
            "client_id": session['client_id'],
            "redirect_uri": session['redirect_uri'],
            "code_challenge": session['code_challenge'],
            "scope": session['scope']
        })
    )

    # 6. Redirect back to Claude with our code
    callback_url = (
        f"{session['redirect_uri']}"
        f"?code={our_auth_code}"
        f"&state={session['state']}"
    )

    return RedirectResponse(callback_url)


@app.post("/oauth/token")
async def oauth_token(
    grant_type: str = Form(...),
    code: str = Form(...),
    code_verifier: str = Form(...),
    redirect_uri: str = Form(...),
    client_id: str = Form(...)
):
    """Token exchange endpoint - validates PKCE and issues access token"""

    # 1. Validate grant type
    if grant_type != "authorization_code":
        raise HTTPException(400, "Only authorization_code grant supported")

    # 2. Get authorization code data
    code_data = await redis_client.get(f"oauth:code:{code}")
    if not code_data:
        raise HTTPException(400, "Invalid or expired authorization code")

    auth_data = json.loads(code_data)

    # 3. Validate client and redirect URI
    if auth_data['client_id'] != client_id:
        raise HTTPException(400, "Client mismatch")

    if auth_data['redirect_uri'] != redirect_uri:
        raise HTTPException(400, "Redirect URI mismatch")

    # 4. Verify PKCE
    import hashlib
    import base64

    computed_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip('=')

    if not secrets.compare_digest(computed_challenge, auth_data['code_challenge']):
        raise HTTPException(400, "PKCE verification failed")

    # 5. Delete authorization code (single use)
    await redis_client.delete(f"oauth:code:{code}")

    # 6. Generate access token
    access_token = secrets.token_urlsafe(32)

    # 7. Store token with user email
    await redis_client.setex(
        f"oauth:token:{access_token}",
        3600,  # 1 hour expiry
        auth_data['user_email']
    )

    # 8. Return token response
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": auth_data['scope']
    }


# Update MCP endpoint to validate OUR tokens
@app.post("/mcp")
async def mcp_endpoint(
    request: Request,
    authorization: str | None = Header(None)
):
    """MCP endpoint - validates our OAuth tokens"""

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")

    token = authorization[7:]

    # Validate OUR token
    user_email = await redis_client.get(f"oauth:token:{token}")
    if not user_email:
        raise HTTPException(401, "Invalid or expired token")

    user_id = user_email.decode().replace("@", "_at_").replace(".", "_dot_")

    # Rest of MCP logic...
```

### Required Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    "document-mcp>=0.0.4",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "google-auth>=2.34.0",
    "redis>=5.0.0",  # NEW - for session/token storage
]
```

### Required Environment Variables

```bash
# Add to GitHub Secrets and .env
GOOGLE_OAUTH_CLIENT_ID=<your-client-id>
GOOGLE_OAUTH_CLIENT_SECRET=<your-client-secret>
SERVER_URL=https://document-mcp-451560119112.us-central1.run.app
```

**Note**: Get OAuth credentials from the `client_secret_*.json` file in the repository root (gitignored).

### Deployment

```yaml
# Update .github/workflows/deploy-cloud-run.yml
--set-env-vars "DOCUMENTS_STORAGE_PATH=/data/documents_storage,LOG_LEVEL=info,GOOGLE_OAUTH_CLIENT_ID=${{ secrets.GOOGLE_OAUTH_CLIENT_ID }},GOOGLE_OAUTH_CLIENT_SECRET=${{ secrets.GOOGLE_OAUTH_CLIENT_SECRET }},SERVER_URL=https://document-mcp-451560119112.us-central1.run.app"
```

---

## Solution 2: Use MCP Remote Proxy (Quick Fix, Not Recommended)

**Pros**: Works immediately, no server changes
**Cons**: Tokens expire every hour, manual refresh needed

### Setup

1. **Get OAuth token from Google**:
   ```
   https://developers.google.com/oauthplayground/
   - Select: Google OAuth2 API v2 → userinfo.email
   - Authorize and exchange for token
   - Copy access token (starts with ya29.)
   ```

2. **Configure Claude Desktop**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "npx",
         "args": [
           "mcp-remote@latest",
           "--http",
           "https://document-mcp-451560119112.us-central1.run.app/mcp",
           "--header",
           "Authorization: Bearer ya29.YOUR_TOKEN_HERE"
         ]
       }
     }
   }
   ```

3. **Every hour**: Get new token and update config

---

## Recommendation

**For Production**: Implement Solution 1 (OAuth 2.1 server)
- Takes 3-4 hours to implement
- Users get seamless experience
- Tokens auto-refresh
- No manual intervention

**For Testing**: Use Solution 2 (MCP Remote Proxy)
- Works immediately
- Test the MCP tools
- Decide if OAuth 2.1 server is worth the effort

---

## Why This Happened

Claude Desktop's Custom Connectors documentation wasn't entirely clear. It says:
> "Optionally, click 'Advanced settings' to specify an OAuth Client ID and OAuth Client Secret"

This makes it sound like you provide GOOGLE's OAuth credentials. But actually:
1. Claude Desktop acts as OAuth client
2. YOUR server must be OAuth authorization server
3. You WRAP Google OAuth in your own OAuth flow

This is the same OAuth 2.1 + PKCE flow that ChatGPT uses!

---

## Next Steps

**Option A - Implement OAuth Server** (Recommended):
1. Add Redis to dependencies
2. Add 3 OAuth endpoints to server.py
3. Add environment variables
4. Deploy
5. Test with Claude Desktop

**Option B - Use Proxy** (Quick Test):
1. Get token from OAuth Playground
2. Configure claude_desktop_config.json
3. Test MCP tools
4. Decide on permanent solution

**Estimated Time**:
- Option A: 3-4 hours development + testing
- Option B: 5 minutes setup (but requires hourly token refresh)

---

**Want me to implement Option A for you?** I can write the complete OAuth 2.1 server code.
