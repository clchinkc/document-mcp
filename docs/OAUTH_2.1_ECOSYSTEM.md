# OAuth 2.1 Ecosystem for MCP Servers

**Date**: December 19, 2025
**Status**: Research & Future Planning
**Spec Version**: MCP Authorization Spec 2025-11-25
**Purpose**: Technical reference for future multi-provider OAuth implementation

---

## Executive Summary

This document provides comprehensive technical details about OAuth 2.1 for Model Context Protocol (MCP) servers. It serves as a reference for future implementation when multi-provider support becomes necessary.

**Current Status**: Server uses simple OAuth (Google) for Claude Desktop only
**Future Consideration**: OAuth 2.1 server for ChatGPT, Gemini, and other providers

---

## Table of Contents

1. [OAuth 2.1 Overview](#oauth-21-overview)
2. [MCP Authorization Specification](#mcp-authorization-specification)
3. [Provider-Specific Implementations](#provider-specific-implementations)
4. [Technical Requirements](#technical-requirements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Security Considerations](#security-considerations)
7. [References & Resources](#references--resources)

---

## OAuth 2.1 Overview

### What Changed from OAuth 2.0

OAuth 2.1 consolidates best practices from OAuth 2.0 and its extensions:

| Feature | OAuth 2.0 | OAuth 2.1 |
|---------|-----------|-----------|
| **PKCE** | Optional | **Mandatory** for all clients |
| **Client Secret** | Required for confidential clients | Optional (PKCE replaces it) |
| **Redirect URI** | Exact match | **Exact match + no wildcards** |
| **Implicit Flow** | Supported | **Removed** (use Authorization Code + PKCE) |
| **Resource Owner Password** | Supported | **Removed** (security risk) |
| **Bearer Tokens** | Any transport | **HTTPS required** |
| **Refresh Tokens** | Optional | **Recommended** with rotation |

**Key Principle**: "Secure by default" - removes insecure options, mandates best practices.

**Resources**:
- [OAuth 2.1 Draft Specification](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-10)
- [OAuth 2.1 Consolidated Best Practices](https://aaronparecki.com/2019/12/12/21/its-time-for-oauth-2-dot-1)

---

## MCP Authorization Specification

### November 2025 Spec Update

The MCP protocol added comprehensive authorization support on **2025-11-25**. Key features:

#### 1. Client Registration Methods

**Two approaches for OAuth client registration:**

##### A. Dynamic Client Registration (DCR)
```
Client                          Authorization Server
  |                                     |
  | POST /register                      |
  | {                                   |
  |   "client_name": "ChatGPT",        |
  |   "redirect_uris": [...]           |
  | }                                   |
  |------------------------------------>|
  |                                     |
  | 201 Created                         |
  | {                                   |
  |   "client_id": "temp_123",         |
  |   "client_secret": "...",          |  (optional)
  |   "expires_at": 1234567890         |
  | }                                   |
  |<------------------------------------|
```

**Used by**: ChatGPT, dynamic MCP clients
**Spec**: [RFC 7591 - OAuth 2.0 Dynamic Client Registration](https://datatracker.ietf.org/doc/html/rfc7591)

##### B. Client ID Metadata Documents (CIMD)
```
Client                          Authorization Server
  |                                     |
  | GET /.well-known/client-id-metadata |
  |------------------------------------>|
  |                                     |
  | 200 OK                              |
  | {                                   |
  |   "client_id": "fixed_client",     |
  |   "authorization_endpoint": "...", |
  |   "token_endpoint": "...",         |
  |   "scopes_supported": [...]        |
  | }                                   |
  |<------------------------------------|
```

**Used by**: Future MCP clients (preferred method)
**Status**: Emerging standard, replacing DCR complexity
**Resource**: [CIMD Specification](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update)

#### 2. Protected Resource Metadata

Authorization servers advertise their capabilities:

```json
GET /.well-known/oauth-authorization-server
{
  "issuer": "https://document-mcp.run.app",
  "authorization_endpoint": "https://document-mcp.run.app/oauth/authorize",
  "token_endpoint": "https://document-mcp.run.app/oauth/token",
  "registration_endpoint": "https://document-mcp.run.app/oauth/register",
  "scopes_supported": ["read", "write"],
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "token_endpoint_auth_methods_supported": ["none", "client_secret_post"]
}
```

**Spec**: [RFC 8414 - OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414)

#### 3. PKCE (Proof Key for Code Exchange)

**Mandatory** in OAuth 2.1 and MCP Authorization Spec.

**Flow**:
```
1. Client generates:
   code_verifier = random 43-128 character string
   code_challenge = BASE64URL(SHA256(code_verifier))

2. Authorization request:
   GET /authorize?
     client_id=123&
     redirect_uri=https://...&
     code_challenge=xyz&
     code_challenge_method=S256&
     state=abc

3. Token exchange:
   POST /token
   {
     "grant_type": "authorization_code",
     "code": "auth_code_123",
     "code_verifier": "original_verifier",  // Server validates: SHA256(this) == code_challenge
     "redirect_uri": "https://..."
   }
```

**Security Benefit**: Prevents authorization code interception attacks
**Spec**: [RFC 7636 - PKCE](https://datatracker.ietf.org/doc/html/rfc7636)

---

## Provider-Specific Implementations

### 1. Claude Desktop (Current Implementation)

**Authentication Pattern**: Simple Google OAuth with Client ID + Secret

**Flow**:
```
User                Claude Desktop           Our Server              Google
 |                        |                      |                     |
 | Add connector         |                      |                     |
 | (Client ID + Secret)  |                      |                     |
 |---------------------->|                      |                     |
 |                        |                      |                     |
 | First use             |                      |                     |
 |---------------------->|                      |                     |
 |                        | OAuth flow          |                     |
 |                        |---------------------|-------------------->|
 |                        |                      |                     |
 | Google sign-in page   |                      |                     |
 |<---------------------------------------------------------|
 | Authenticate          |                      |                     |
 |--------------------------------------------------------->|
 |                        |                      |                     |
 |                        | access_token         |                     |
 |                        |<--------------------|---------------------|
 |                        |                      |                     |
 |                        | MCP request          |                     |
 |                        | Authorization: Bearer <token>              |
 |                        |--------------------->|                     |
 |                        |                      | Verify token        |
 |                        |                      |-------------------->|
 |                        |                      | Email, validity     |
 |                        |                      |<--------------------|
 |                        |                      | Extract user_id     |
 |                        |                      | Route to storage    |
 |                        | MCP response         |                     |
 |                        |<---------------------|                     |
```

**Implementation** (server.py):
```python
def verify_google_oauth_token(authorization: str) -> str:
    """Simple token verification - server acts as Resource Server only"""
    token = authorization[7:]  # Remove "Bearer "
    idinfo = id_token.verify_oauth2_token(
        token,
        google_requests.Request(),
        os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    )
    return idinfo.get('email')
```

**Characteristics**:
- ✅ Simple: Server doesn't issue tokens, only validates
- ✅ Secure: Google handles OAuth flows
- ✅ UX: Token auto-refresh by Claude Desktop
- ❌ Claude Desktop only

**OAuth Callback URL**: `https://claude.ai/api/mcp/auth_callback`

### 2. ChatGPT (OpenAI Apps SDK)

**Authentication Pattern**: OAuth 2.1 with Dynamic Client Registration + PKCE

**Flow**:
```
User          ChatGPT              Our Server (must be Auth Server)
 |               |                           |
 | Add MCP      |                           |
 | server       |                           |
 |------------->|                           |
 |               | POST /register           |
 |               | (DCR)                    |
 |               |------------------------->|
 |               |                           | Create temp client_id
 |               | client_id                |
 |               |<-------------------------|
 |               |                           |
 |               | Generate PKCE:           |
 |               | - code_verifier          |
 |               | - code_challenge         |
 |               |                           |
 |               | GET /authorize?          |
 |               |   code_challenge=...     |
 |               |------------------------->|
 |               |                           | Store code_challenge
 |               | redirect to Google       |
 |               |<-------------------------|
 | Google       |                           |
 | sign-in      |                           |
 |<-------------|                           |
 | Authenticate |                           |
 |------------->|                           |
 |               | authorization_code       |
 |               |<-------------------------|
 |               |                           |
 |               | POST /token              |
 |               | {                        |
 |               |   code: "...",           |
 |               |   code_verifier: "..."   |
 |               | }                        |
 |               |------------------------->|
 |               |                           | Verify: SHA256(verifier) == challenge
 |               |                           | Issue access_token
 |               | access_token             |
 |               |<-------------------------|
 |               |                           |
 |               | MCP request              |
 |               | Authorization: Bearer... |
 |               |------------------------->|
```

**Required Endpoints**:
```python
@app.post("/oauth/register")  # Dynamic Client Registration
@app.get("/oauth/authorize")  # Authorization endpoint
@app.post("/oauth/token")     # Token exchange (with PKCE validation)
@app.get("/.well-known/oauth-authorization-server")  # Metadata
```

**Key Differences from Claude**:
- ChatGPT acts as OAuth client (not end user)
- Server must BE authorization server (not just resource server)
- No client secret (uses PKCE instead)
- Temporary client registrations per session

**Complexity**: HIGH - requires full OAuth 2.1 server implementation

**Resources**:
- [ChatGPT MCP Authentication Guide](https://stytch.com/blog/guide-to-authentication-for-the-openai-apps-sdk/)
- [OpenAI Apps SDK Documentation](https://developers.openai.com/apps-sdk/build/auth/)
- [MCP OAuth 2.1 Implementation](https://www.scalekit.com/blog/implement-oauth-for-mcp-servers)

### 3. Google Gemini CLI

**Authentication Pattern**: OAuth Personal (CLI-based, not remote MCP)

**Flow**:
```
Developer          Gemini CLI         Google OAuth
    |                  |                    |
    | gemini auth      |                    |
    |----------------->|                    |
    |                  | Browser OAuth flow |
    |                  |------------------->|
    | Sign in          |                    |
    |<-----------------|                    |
    | Grant permission |                    |
    |--------------------------------->|
    |                  | access_token       |
    |                  |<-------------------|
    |                  | Store in           |
    |                  | settings.json      |
    |                  |                    |
    | gemini chat      |                    |
    |----------------->|                    |
    |                  | Use stored token   |
    |                  | (local execution)  |
```

**Configuration** (settings.json):
```json
{
  "selectedAuthType": "oauth-personal",
  "apiKeys": {
    "google": "stored-oauth-token"
  }
}
```

**Characteristics**:
- CLI-based, not remote MCP server pattern
- OAuth token stored locally
- 60 requests/minute, 1000 requests/day (free tier)
- Not compatible with remote MCP servers

**Resources**:
- [Gemini CLI Authentication](https://ai.google.dev/gemini-api/docs/oauth)
- [Gemini CLI Setup Guide](https://www.ajeetraina.com/how-to-setup-gemini-cli-docker-mcp-toolkit-for-ai-assisted-development/)

### 4. Perplexity

**Status**: Remote MCP "coming soon" (as of December 2025)

**Current**: Local MCPs only (macOS via Mac App Store)

**Future OAuth Pattern**: Unknown (waiting for official release)

**Resources**:
- [Perplexity MCP Documentation](https://docs.perplexity.ai/guides/mcp-server)
- [Local and Remote MCPs](https://www.perplexity.ai/help-center/en/articles/11502712-local-and-remote-mcps-for-perplexity)

---

## Technical Requirements

### For OAuth 2.1 Authorization Server Implementation

#### 1. Core Components

```python
# Required libraries
from authlib.integrations.fastapi_oauth2 import AuthorizationServer
from authlib.oauth2 import OAuth2Request
from authlib.oauth2.rfc6749 import grants
from authlib.oauth2.rfc7636 import CodeChallenge  # PKCE

# Database models
class OAuth2Client:
    client_id: str
    client_secret: Optional[str]  # Optional in OAuth 2.1
    redirect_uris: List[str]
    grant_types: List[str]
    scope: str
    expires_at: Optional[datetime]

class OAuth2AuthorizationCode:
    code: str
    client_id: str
    redirect_uri: str
    scope: str
    user_id: str
    code_challenge: str  # PKCE
    code_challenge_method: str  # S256
    expires_at: datetime

class OAuth2Token:
    access_token: str
    refresh_token: Optional[str]
    token_type: str = "Bearer"
    scope: str
    user_id: str
    client_id: str
    expires_at: datetime
```

#### 2. Required Endpoints

```python
# OAuth 2.1 Authorization Server Endpoints
@app.post("/oauth/register")
async def register_client(request: OAuth2Request):
    """Dynamic Client Registration (RFC 7591)"""
    # Validate registration request
    # Create temporary client_id (or permanent for CIMD)
    # Store client metadata
    # Return client_id (and optional client_secret)
    pass

@app.get("/oauth/authorize")
async def authorize(request: OAuth2Request):
    """Authorization endpoint"""
    # Validate request (client_id, redirect_uri, scope, code_challenge)
    # Redirect to Google OAuth (or show consent screen)
    # Generate authorization code
    # Store code with code_challenge
    # Redirect back with code
    pass

@app.post("/oauth/token")
async def token_exchange(request: OAuth2Request):
    """Token endpoint with PKCE validation"""
    # Validate authorization code
    # Verify PKCE: SHA256(code_verifier) == stored code_challenge
    # Issue access_token and refresh_token
    # Return tokens
    pass

@app.post("/oauth/token")
async def refresh_token(request: OAuth2Request):
    """Refresh token endpoint"""
    # Validate refresh_token
    # Issue new access_token
    # Optionally rotate refresh_token
    pass

@app.get("/.well-known/oauth-authorization-server")
async def server_metadata():
    """OAuth Server Metadata (RFC 8414)"""
    return {
        "issuer": "https://document-mcp.run.app",
        "authorization_endpoint": "...",
        "token_endpoint": "...",
        "registration_endpoint": "...",
        # ... more metadata
    }
```

#### 3. Database Schema

**PostgreSQL Example**:
```sql
CREATE TABLE oauth2_clients (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255) UNIQUE NOT NULL,
    client_secret VARCHAR(255),  -- NULL for public clients
    client_name VARCHAR(255),
    redirect_uris TEXT[] NOT NULL,
    grant_types TEXT[] NOT NULL,
    scope TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP  -- For temporary clients (DCR)
);

CREATE TABLE oauth2_authorization_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(255) UNIQUE NOT NULL,
    client_id VARCHAR(255) REFERENCES oauth2_clients(client_id),
    redirect_uri TEXT NOT NULL,
    scope TEXT,
    user_id VARCHAR(255) NOT NULL,
    code_challenge VARCHAR(255) NOT NULL,  -- PKCE
    code_challenge_method VARCHAR(10) NOT NULL,  -- S256
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,  -- 10 minute max
    used BOOLEAN DEFAULT FALSE
);

CREATE TABLE oauth2_tokens (
    id SERIAL PRIMARY KEY,
    access_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    token_type VARCHAR(50) DEFAULT 'Bearer',
    scope TEXT,
    user_id VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) REFERENCES oauth2_clients(client_id),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,  -- 1 hour for access, 7 days for refresh
    revoked BOOLEAN DEFAULT FALSE
);

-- Indexes for performance
CREATE INDEX idx_oauth2_codes_code ON oauth2_authorization_codes(code);
CREATE INDEX idx_oauth2_codes_expires ON oauth2_authorization_codes(expires_at);
CREATE INDEX idx_oauth2_tokens_access ON oauth2_tokens(access_token);
CREATE INDEX idx_oauth2_tokens_refresh ON oauth2_tokens(refresh_token);
```

**Redis for Session/State**:
```python
# Temporary state storage (OAuth flows)
redis_client.setex(
    f"oauth:state:{state}",
    600,  # 10 minute expiry
    json.dumps({
        "client_id": "...",
        "redirect_uri": "...",
        "code_challenge": "..."
    })
)
```

#### 4. PKCE Implementation

```python
import hashlib
import base64
import secrets

def generate_code_verifier():
    """Generate PKCE code verifier (43-128 characters)"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

def generate_code_challenge(verifier: str):
    """Generate PKCE code challenge (S256 method)"""
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')

def verify_code_challenge(verifier: str, challenge: str) -> bool:
    """Verify PKCE code verifier matches challenge"""
    computed_challenge = generate_code_challenge(verifier)
    return secrets.compare_digest(computed_challenge, challenge)
```

#### 5. Security Implementation

```python
# Token generation (cryptographically secure)
def generate_token():
    return secrets.token_urlsafe(32)

# Constant-time comparison (prevents timing attacks)
import secrets
secrets.compare_digest(stored_token, provided_token)

# Authorization code expiration (max 10 minutes per spec)
code_expires_at = datetime.utcnow() + timedelta(minutes=10)

# Access token expiration (1 hour recommended)
token_expires_at = datetime.utcnow() + timedelta(hours=1)

# Refresh token rotation (security best practice)
def rotate_refresh_token(old_refresh_token: str):
    # Revoke old refresh token
    # Issue new refresh token
    # Return new pair
    pass
```

---

## Implementation Roadmap

### Phase 1: Research & Planning (1-2 weeks)

**Objectives**:
- Deep dive into MCP Authorization Spec
- Evaluate OAuth 2.1 libraries (authlib, oauthlib)
- Design database schema
- Security threat modeling

**Deliverables**:
- Technical architecture document
- Library selection justification
- Database design
- Security requirements checklist

### Phase 2: Core OAuth 2.1 Server (2-3 weeks)

**Tasks**:
1. Database setup (PostgreSQL + Redis)
2. Implement authorization server with authlib
3. Add DCR endpoint (/oauth/register)
4. Add authorization endpoint (/oauth/authorize)
5. Add token endpoint with PKCE (/oauth/token)
6. Add server metadata (/.well-known/...)

**Testing**:
- Unit tests for PKCE validation
- Integration tests for OAuth flows
- Load testing for token generation

### Phase 3: ChatGPT Integration (1-2 weeks)

**Tasks**:
1. Test with ChatGPT OAuth client
2. Handle DCR edge cases
3. Validate PKCE implementation
4. Error handling and logging

**Testing**:
- End-to-end testing with ChatGPT
- OAuth flow error scenarios
- Token refresh testing

### Phase 4: Security Hardening (1 week)

**Tasks**:
1. Security audit (consider external audit)
2. Rate limiting implementation
3. HTTPS enforcement
4. Secret rotation procedures
5. Monitoring and alerting

**Testing**:
- Penetration testing
- OAuth security best practices validation
- Incident response procedures

### Phase 5: CIMD Support (1 week)

**Tasks**:
1. Implement Client ID Metadata Documents
2. Transition from DCR to CIMD (future-proofing)
3. Documentation updates

### Total Estimated Time: 6-9 weeks full-time

---

## Security Considerations

### 1. PKCE Validation

**Critical**: Always verify PKCE in token exchange
```python
# ❌ WRONG - Security vulnerability
def token_exchange(code, redirect_uri):
    # Missing PKCE verification!
    return issue_token(code)

# ✅ CORRECT
def token_exchange(code, redirect_uri, code_verifier):
    stored_challenge = get_code_challenge(code)
    computed_challenge = generate_code_challenge(code_verifier)
    if not secrets.compare_digest(stored_challenge, computed_challenge):
        raise InvalidGrant("PKCE verification failed")
    return issue_token(code)
```

### 2. Authorization Code Security

- **Single use only**: Mark code as used immediately
- **Short expiration**: Max 10 minutes per OAuth 2.1 spec
- **Bind to client**: Verify client_id matches registration
- **Exact redirect_uri match**: No wildcards or partial matches

```python
async def validate_authorization_code(code: str, client_id: str, redirect_uri: str):
    db_code = await get_code(code)

    # Check if already used
    if db_code.used:
        # Revoke ALL tokens for this code (security measure)
        await revoke_all_tokens_for_code(code)
        raise InvalidGrant("Authorization code already used")

    # Check expiration
    if datetime.utcnow() > db_code.expires_at:
        raise InvalidGrant("Authorization code expired")

    # Verify client
    if db_code.client_id != client_id:
        raise InvalidClient("Client mismatch")

    # Exact redirect URI match
    if db_code.redirect_uri != redirect_uri:
        raise InvalidRequest("Redirect URI mismatch")

    # Mark as used
    await mark_code_used(code)
    return db_code
```

### 3. Token Security

**Access Tokens**:
- Bearer tokens (must use HTTPS)
- 1 hour expiration (recommended)
- Cryptographically random (32+ bytes)
- Store hash, not plaintext

**Refresh Tokens**:
- Longer expiration (7-30 days)
- Rotation on each use (security best practice)
- Revoke on suspicious activity
- Store hash, not plaintext

```python
import hashlib

def hash_token(token: str) -> str:
    """Store hashed tokens in database"""
    return hashlib.sha256(token.encode()).hexdigest()

async def issue_tokens(user_id: str, client_id: str, scope: str):
    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(32)

    # Store hashed tokens
    await db.tokens.create({
        "access_token_hash": hash_token(access_token),
        "refresh_token_hash": hash_token(refresh_token),
        "user_id": user_id,
        "client_id": client_id,
        "scope": scope,
        "access_expires_at": datetime.utcnow() + timedelta(hours=1),
        "refresh_expires_at": datetime.utcnow() + timedelta(days=7)
    })

    # Return plaintext tokens (only this one time)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": 3600
    }
```

### 4. Common Vulnerabilities to Prevent

| Vulnerability | Prevention |
|---------------|------------|
| **Authorization Code Interception** | PKCE mandatory |
| **CSRF Attacks** | State parameter validation |
| **Redirect URI Manipulation** | Exact match, no wildcards |
| **Token Replay** | Short expiration + one-time codes |
| **Timing Attacks** | `secrets.compare_digest()` |
| **Open Redirect** | Strict redirect URI whitelist |
| **Token Leakage in Logs** | Never log tokens/secrets |

### 5. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/oauth/register")
@limiter.limit("10/hour")  # Prevent client registration spam
async def register_client(request: Request):
    pass

@app.post("/oauth/token")
@limiter.limit("100/minute")  # Prevent token brute force
async def token_exchange(request: Request):
    pass
```

---

## References & Resources

### Official Specifications

1. **OAuth 2.1 Draft**
   - [Draft Specification](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-10)
   - Consolidated best practices from OAuth 2.0

2. **RFC 7591 - Dynamic Client Registration**
   - [Specification](https://datatracker.ietf.org/doc/html/rfc7591)
   - Used by ChatGPT and dynamic MCP clients

3. **RFC 7636 - PKCE**
   - [Specification](https://datatracker.ietf.org/doc/html/rfc7636)
   - Mandatory in OAuth 2.1

4. **RFC 8414 - Authorization Server Metadata**
   - [Specification](https://datatracker.ietf.org/doc/html/rfc8414)
   - /.well-known/oauth-authorization-server

### MCP-Specific Resources

5. **MCP Authorization Spec (2025-11-25)**
   - [Blog Post](https://den.dev/blog/mcp-november-authorization-spec/)
   - [Client ID Metadata Documents](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update)

6. **MCP OAuth Implementation Guides**
   - [Scalekit Guide](https://www.scalekit.com/blog/implement-oauth-for-mcp-servers)
   - [Stytch MCP Authentication](https://stytch.com/blog/MCP-authentication-and-authorization-guide/)
   - [Auth0 MCP Integration](https://auth0.com/blog/add-remote-mcp-server-chatgpt/)

### Provider Documentation

7. **Claude Desktop**
   - [Custom Connectors](https://support.claude.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers)
   - [Getting Started](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp)

8. **ChatGPT/OpenAI**
   - [Apps SDK Authentication](https://developers.openai.com/apps-sdk/build/auth/)
   - [OAuth Implementation Guide](https://stytch.com/blog/guide-to-authentication-for-the-openai-apps-sdk/)

9. **Google Gemini**
   - [Gemini OAuth](https://ai.google.dev/gemini-api/docs/oauth)
   - [Gemini CLI Setup](https://www.ajeetraina.com/how-to-setup-gemini-cli-docker-mcp-toolkit-for-ai-assisted-development/)

10. **Perplexity**
    - [MCP Server Docs](https://docs.perplexity.ai/guides/mcp-server)
    - [Local and Remote MCPs](https://www.perplexity.ai/help-center/en/articles/11502712-local-and-remote-mcps-for-perplexity)

### Implementation Libraries

11. **Authlib** (Recommended)
    - [Documentation](https://docs.authlib.org/)
    - [FastAPI OAuth 2.1 Server](https://docs.authlib.org/en/latest/flask/2/authorization-server.html)
    - Industry-standard, actively maintained

12. **PyJWT** (Token Handling)
    - [Documentation](https://pyjwt.readthedocs.io/)
    - For JWT-based tokens (optional)

13. **Cryptography** (Security)
    - [Documentation](https://cryptography.io/)
    - For PKCE, token generation, secret management

### Security Resources

14. **OAuth 2.0 Security Best Practices**
    - [BCP Draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)
    - Comprehensive security guidance

15. **OWASP OAuth Security**
    - [OWASP OAuth Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/OAuth2_Cheat_Sheet.html)
    - Common vulnerabilities and mitigations

---

## Conclusion

This document provides a comprehensive technical foundation for implementing OAuth 2.1 support for MCP servers. The complexity is significant, but the architecture is well-specified and production-ready libraries exist.

**Key Takeaways**:
1. OAuth 2.1 consolidates best practices (PKCE mandatory, insecure flows removed)
2. MCP Authorization Spec (Nov 2025) provides clear guidance
3. ChatGPT requires full OAuth 2.1 server with DCR + PKCE
4. Claude Desktop uses simpler OAuth pattern (current implementation)
5. Implementation is 6-9 weeks of focused development
6. Security audit strongly recommended

**Current Decision**: Maintain Claude Desktop-only implementation for simplicity and security.

**Future Consideration**: Implement OAuth 2.1 server when:
- Significant user demand for ChatGPT/other providers
- Resources available for proper implementation and security audit
- MCP OAuth ecosystem has matured (6-12 months)

---

**Last Updated**: December 19, 2025
**Next Review**: June 2026 (or when significant user demand emerges)
