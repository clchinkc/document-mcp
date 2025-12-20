"""
Hosted Document MCP Server with FastAPI.

This server wraps the document-mcp package in a FastAPI application,
providing both MCP protocol endpoints and optional REST API for web UIs.
Designed for integration with Firstory and Claude Desktop.

Architecture:
- FastAPI provides HTTP routing, CORS, and optional REST API
- document_mcp.mcp_server provides MCP tools and resources
- The /mcp endpoint handles JSON-RPC requests by calling MCP server methods
"""

import logging
import os
import sys
import json
import secrets
import hashlib
import base64
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlencode

from fastapi import FastAPI, Header, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import redis.asyncio as redis
import requests

# Import the document MCP server
try:
    from document_mcp.doc_tool_server import mcp_server
except ImportError:
    print("Error: document-mcp package not installed. Run: pip install document-mcp", file=sys.stderr)
    sys.exit(1)

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Redis client placeholder - will be initialized in lifespan
redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """Get or create Redis client with proper connection handling."""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
    return redis_client


# Google OAuth verification
def verify_google_oauth_token(authorization: str | None) -> str:
    """
    Verify Google OAuth token and return user email.

    Args:
        authorization: Authorization header value (format: "Bearer <token>")

    Returns:
        User email from verified token

    Raises:
        HTTPException(401): If token is invalid, expired, or missing
        HTTPException(500): If OAuth configuration is missing
    """
    # Check if OAuth client ID is configured
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    if not client_id:
        logger.error("GOOGLE_OAUTH_CLIENT_ID not configured")
        raise HTTPException(
            status_code=500,
            detail="Server OAuth configuration error"
        )

    # Validate Authorization header format
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Provide: Authorization: Bearer <token>"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>"
        )

    # Extract token
    token = authorization[7:]  # Remove "Bearer " prefix

    # Verify token with Google
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            client_id
        )

        # Validate issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            log_audit_event(
                user_id="unknown",
                operation="oauth_authentication",
                tool_name="verify_google_oauth_token",
                arguments={"issuer": idinfo.get('iss')},
                success=False,
                error_message=f"Invalid issuer: {idinfo.get('iss')}"
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid token issuer"
            )

        # Extract email
        email = idinfo.get('email')
        if not email:
            log_audit_event(
                user_id="unknown",
                operation="oauth_authentication",
                tool_name="verify_google_oauth_token",
                arguments={},
                success=False,
                error_message="No email in token claims"
            )
            raise HTTPException(
                status_code=401,
                detail="Token missing email claim"
            )

        # Log successful authentication
        logger.info(f"OAuth authentication successful for: {email}")

        return email

    except ValueError as e:
        # Token verification failed (invalid signature, expired, etc.)
        log_audit_event(
            user_id="unknown",
            operation="oauth_authentication",
            tool_name="verify_google_oauth_token",
            arguments={"token_prefix": token[:20] if token else "None"},
            success=False,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired OAuth token: {str(e)}"
        )


# User isolation utilities
def get_user_id_from_oauth(authorization: str | None) -> str:
    """
    Extract user ID from OAuth token.

    Args:
        authorization: Authorization header value (format: "Bearer <token>")

    Returns:
        Sanitized user ID derived from email address

    Raises:
        HTTPException(401): If OAuth authentication fails
        HTTPException(500): If OAuth configuration is missing

    Example:
        alice@gmail.com → alice_at_gmail_dot_com
    """
    # Verify OAuth token and get email
    email = verify_google_oauth_token(authorization)

    # Sanitize email for filesystem usage
    # Replace @ with _at_ and . with _dot_
    user_id = email.replace("@", "_at_").replace(".", "_dot_")

    logger.info(f"User ID extracted from OAuth: {user_id}")

    return user_id


def get_user_storage_path(user_id: str) -> str:
    """
    Get the storage directory path for a specific user.

    Returns absolute path to user's storage directory.
    Creates directory if it doesn't exist.
    """
    from pathlib import Path

    base_storage = os.environ.get("DOCUMENTS_STORAGE_PATH", "./documents_storage")

    if user_id == "default":
        # Default user uses root storage for backward compatibility
        user_path = Path(base_storage)
    else:
        # Other users get isolated subdirectories
        user_path = Path(base_storage) / user_id

    # Create directory if it doesn't exist
    user_path.mkdir(parents=True, exist_ok=True)

    return str(user_path.absolute())


# Audit logging functionality
def log_audit_event(
    user_id: str,
    operation: str,
    tool_name: str,
    arguments: dict,
    success: bool,
    error_message: str | None = None
):
    """
    Log audit events for compliance and security monitoring.

    Logs to both stderr (for MCP compliance) and audit log file.
    """
    import json
    import datetime

    audit_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user_id": user_id,
        "operation": operation,
        "tool_name": tool_name,
        "arguments": {k: v for k, v in arguments.items() if k != "content"},  # Exclude large content
        "success": success,
        "error": error_message
    }

    # Log to audit file
    audit_log_path = os.environ.get("AUDIT_LOG_PATH", "./audit.log")
    try:
        with open(audit_log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

    # Also log to application logger
    if success:
        logger.info(f"Audit: {user_id} - {operation} - {tool_name}", extra=audit_entry)
    else:
        logger.warning(f"Audit: {user_id} - {operation} - {tool_name} - FAILED: {error_message}", extra=audit_entry)


# OAuth 2.1 helper functions
def generate_code_challenge_from_verifier(code_verifier: str) -> str:
    """Generate PKCE code challenge from verifier (S256 method)."""
    digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')


async def store_session_data(session_id: str, data: dict, ttl: int = 600):
    """Store session data in Redis with TTL (default 10 minutes)."""
    await get_redis_client().setex(
        f"oauth:session:{session_id}",
        ttl,
        json.dumps(data)
    )


async def get_session_data(session_id: str) -> dict | None:
    """Get session data from Redis."""
    data = await get_redis_client().get(f"oauth:session:{session_id}")
    return json.loads(data) if data else None


async def validate_access_token(authorization: str | None) -> str:
    """
    Validate our OAuth access token and return user email.

    Args:
        authorization: Authorization header value (format: "Bearer <token>")

    Returns:
        User email from validated token

    Raises:
        HTTPException(401): If token is invalid, expired, or missing
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = authorization[7:]  # Remove "Bearer " prefix

    # Get user email from Redis
    user_email = await get_redis_client().get(f"oauth:token:{token}")
    if not user_email:
        raise HTTPException(401, "Invalid or expired access token")

    return user_email


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown."""
    logger.info("Starting Document MCP Server...")
    logger.info(f"Server version: {app.version}")
    logger.info(f"MCP server name: {mcp_server.name}")
    logger.info(f"Document storage: {os.environ.get('DOCUMENTS_STORAGE_PATH', './documents_storage')}")
    yield
    logger.info("Shutting down Document MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Document MCP Server",
    description="Hosted MCP server for document management. "
                "Provides MCP protocol endpoints for Claude Desktop and optional REST API for web UIs.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
# Allow requests from Claude Desktop and localhost for development
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else [
    # Anthropic Claude (primary client)
    "https://claude.ai",
    "https://claude.com",  # Future-proofing
    "https://console.anthropic.com",
    # Development
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Mcp-Session-Id", "Mcp-Protocol-Version", "Authorization"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    tools = await mcp_server.list_tools()

    # Verify Redis connection is working
    redis_ok = False
    try:
        await get_redis_client().ping()
        redis_ok = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    # Verify all critical routes are registered
    routes = [route.path for route in app.routes]
    critical_routes = ["/", "/health", "/oauth/register", "/oauth/authorize", "/oauth/callback", "/oauth/token"]
    routes_ok = all(r in routes for r in critical_routes)

    return {
        "status": "healthy" if (redis_ok and routes_ok) else "degraded",
        "version": app.version,
        "mcp_server": mcp_server.name,
        "tools_count": len(tools),
        "redis": "ok" if redis_ok else "error",
        "routes": "ok" if routes_ok else "missing"
    }


# OAuth 2.1 Authorization Server Endpoints

@app.post("/oauth/register")
async def oauth_register(request: Request):
    """
    Dynamic Client Registration (RFC 7591).

    Claude Desktop uses this to register itself as an OAuth client
    before starting the authorization flow.
    """
    try:
        body = await request.json()

        # Extract client metadata
        redirect_uris = body.get("redirect_uris", [])
        client_name = body.get("client_name", "Unknown Client")
        grant_types = body.get("grant_types", ["authorization_code"])
        response_types = body.get("response_types", ["code"])
        token_endpoint_auth_method = body.get("token_endpoint_auth_method", "none")

        # Validate required fields
        if not redirect_uris:
            raise HTTPException(400, "redirect_uris is required")

        # Generate client_id (public client, no secret needed for PKCE flow)
        client_id = secrets.token_urlsafe(16)

        # Store client registration in Redis (long TTL - 30 days)
        client_data = {
            "client_id": client_id,
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "grant_types": grant_types,
            "response_types": response_types,
            "token_endpoint_auth_method": token_endpoint_auth_method
        }

        await get_redis_client().setex(
            f"oauth:client:{client_id}",
            30 * 24 * 60 * 60,  # 30 days
            json.dumps(client_data)
        )

        # Audit log
        log_audit_event(
            user_id="system",
            operation="oauth_register",
            tool_name="register_endpoint",
            arguments={"client_name": client_name},
            success=True
        )

        logger.info(f"Registered new OAuth client: {client_name} ({client_id})")

        # Return client registration response (RFC 7591)
        return JSONResponse({
            "client_id": client_id,
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "grant_types": grant_types,
            "response_types": response_types,
            "token_endpoint_auth_method": token_endpoint_auth_method
        }, status_code=201)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Client registration error: {e}", exc_info=True)
        raise HTTPException(500, f"Registration failed: {str(e)}")


@app.get("/oauth/authorize")
async def oauth_authorize(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str,
    scope: str = "claudeai"
):
    """
    OAuth 2.1 authorization endpoint.
    Validates PKCE challenge and redirects to Google OAuth.
    """
    try:
        # 1. Validate parameters
        if response_type != "code":
            raise HTTPException(400, "Only authorization_code flow supported")

        if code_challenge_method != "S256":
            raise HTTPException(400, "Only S256 PKCE method supported")

        if not code_challenge or len(code_challenge) < 43:
            raise HTTPException(400, "Invalid code_challenge")

        # 2. Generate session ID
        session_id = secrets.token_urlsafe(32)

        # 3. Store session data (PKCE challenge, client info)
        await store_session_data(session_id, {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "state": state,
            "scope": scope
        })

        # 4. Build Google OAuth URL
        google_auth_params = {
            "client_id": os.environ["GOOGLE_OAUTH_CLIENT_ID"],
            "redirect_uri": f"{os.environ['SERVER_URL']}/oauth/callback",
            "response_type": "code",
            "scope": "openid email profile",
            "state": session_id
        }
        google_auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(google_auth_params)}"

        # 5. Audit log
        log_audit_event(
            user_id="unknown",
            operation="oauth_authorize",
            tool_name="authorize_endpoint",
            arguments={"client_id": client_id},
            success=True
        )

        # 6. Redirect to Google
        return RedirectResponse(google_auth_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth authorize error: {e}", exc_info=True)
        raise HTTPException(500, f"Authorization failed: {str(e)}")


@app.get("/oauth/callback")
async def oauth_callback(code: str, state: str):
    """
    Google OAuth callback endpoint.
    Exchanges Google code for email, generates our authorization code.
    """
    try:
        # 1. Get session data
        session_data = await get_session_data(state)
        if not session_data:
            raise HTTPException(400, "Invalid or expired session")

        # 2. Exchange Google code for token
        google_token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.environ["GOOGLE_OAUTH_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_OAUTH_CLIENT_SECRET"],
                "redirect_uri": f"{os.environ['SERVER_URL']}/oauth/callback",
                "grant_type": "authorization_code"
            },
            timeout=10
        )

        if google_token_response.status_code != 200:
            logger.error(f"Google token exchange failed: {google_token_response.status_code} - {google_token_response.text}")
            raise HTTPException(500, f"Failed to exchange Google code: {google_token_response.text}")

        google_token = google_token_response.json()

        # 3. Verify Google token and extract email
        idinfo = id_token.verify_oauth2_token(
            google_token['id_token'],
            google_requests.Request(),
            os.environ['GOOGLE_OAUTH_CLIENT_ID']
        )

        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise HTTPException(401, "Invalid token issuer")

        user_email = idinfo.get('email')
        if not user_email:
            raise HTTPException(401, "Token missing email claim")

        # 4. Generate our authorization code
        our_auth_code = secrets.token_urlsafe(32)

        # 5. Store authorization code with PKCE challenge (10 min TTL)
        await get_redis_client().setex(
            f"oauth:code:{our_auth_code}",
            600,
            json.dumps({
                "user_email": user_email,
                "client_id": session_data['client_id'],
                "redirect_uri": session_data['redirect_uri'],
                "code_challenge": session_data['code_challenge'],
                "scope": session_data['scope']
            })
        )

        # 6. Clean up session
        await get_redis_client().delete(f"oauth:session:{state}")

        # 7. Audit log
        log_audit_event(
            user_id=user_email,
            operation="oauth_callback",
            tool_name="callback_endpoint",
            arguments={"email": user_email},
            success=True
        )

        # 8. Redirect back to Claude with our code
        callback_url = f"{session_data['redirect_uri']}?code={our_auth_code}&state={session_data['state']}"
        return RedirectResponse(callback_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        raise HTTPException(500, f"Callback failed: {str(e)}")


@app.post("/oauth/token")
async def oauth_token(
    grant_type: str = Form(...),
    code: str = Form(...),
    code_verifier: str = Form(...),
    redirect_uri: str = Form(...),
    client_id: str = Form(...)
):
    """
    Token exchange endpoint.
    Validates PKCE and issues access token.
    """
    try:
        # 1. Validate grant type
        if grant_type != "authorization_code":
            raise HTTPException(400, "Only authorization_code grant supported")

        # 2. Get authorization code data
        code_data = await get_redis_client().get(f"oauth:code:{code}")
        if not code_data:
            raise HTTPException(400, "Invalid or expired authorization code")

        auth_data = json.loads(code_data)

        # 3. Validate client and redirect URI
        if auth_data['client_id'] != client_id:
            raise HTTPException(400, "Client mismatch")

        if auth_data['redirect_uri'] != redirect_uri:
            raise HTTPException(400, "Redirect URI mismatch")

        # 4. Verify PKCE (CRITICAL SECURITY)
        computed_challenge = generate_code_challenge_from_verifier(code_verifier)

        if not secrets.compare_digest(computed_challenge, auth_data['code_challenge']):
            log_audit_event(
                user_id=auth_data['user_email'],
                operation="oauth_token",
                tool_name="token_endpoint",
                arguments={"client_id": client_id},
                success=False,
                error_message="PKCE verification failed"
            )
            raise HTTPException(400, "PKCE verification failed")

        # 5. Delete authorization code (single use)
        await get_redis_client().delete(f"oauth:code:{code}")

        # 6. Generate access token
        access_token = secrets.token_urlsafe(32)

        # 7. Store token with user email (1 hour TTL)
        await get_redis_client().setex(
            f"oauth:token:{access_token}",
            3600,
            auth_data['user_email']
        )

        # 8. Audit log
        log_audit_event(
            user_id=auth_data['user_email'],
            operation="oauth_token",
            tool_name="token_endpoint",
            arguments={"client_id": client_id},
            success=True
        )

        # 9. Return token response
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": auth_data['scope']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth token error: {e}", exc_info=True)
        raise HTTPException(500, f"Token exchange failed: {str(e)}")


@app.get("/.well-known/oauth-authorization-server")
async def oauth_server_metadata():
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)."""
    server_url = os.environ.get("SERVER_URL", "https://document-mcp-451560119112.asia-east1.run.app")
    return {
        "issuer": server_url,
        "authorization_endpoint": f"{server_url}/oauth/authorize",
        "token_endpoint": f"{server_url}/oauth/token",
        "registration_endpoint": f"{server_url}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"]
    }


@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource_metadata():
    """OAuth 2.0 Protected Resource Metadata (RFC 9728)."""
    server_url = os.environ.get("SERVER_URL", "https://document-mcp-451560119112.asia-east1.run.app")
    return {
        "resource": server_url,
        "authorization_servers": [server_url],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["claudeai"]
    }


# MCP SSE endpoint - GET / for Claude Desktop SSE transport
@app.get("/")
async def mcp_sse_endpoint(
    request: Request,
    authorization: str | None = Header(None, description="OAuth Bearer token (REQUIRED)")
):
    """
    MCP SSE endpoint for Claude Desktop.

    Claude Desktop may try GET / with Accept: text/event-stream for SSE transport.
    This endpoint returns 401 to trigger OAuth flow, same as POST /.
    """
    # Validate access token - will raise 401 if not valid
    try:
        user_email = await validate_access_token(authorization)
        # If authenticated, return info about the SSE endpoint
        return JSONResponse({
            "message": "MCP SSE endpoint - use POST / for JSON-RPC requests",
            "user": user_email,
            "hint": "Send JSON-RPC 2.0 requests via POST /"
        })
    except HTTPException as e:
        if e.status_code == 401:
            server_url = os.environ.get("SERVER_URL", "https://document-mcp-451560119112.asia-east1.run.app")
            return JSONResponse(
                {"error": "Authentication required", "hint": "Use OAuth 2.1 flow"},
                status_code=401,
                headers={"WWW-Authenticate": f'Bearer resource="{server_url}"'}
            )
        raise


# MCP JSON-RPC endpoint - at root for Claude Desktop compatibility
@app.post("/")
async def mcp_endpoint(
    request: Request,
    authorization: str | None = Header(None, description="OAuth Bearer token (REQUIRED)")
):
    """
    MCP protocol endpoint supporting JSON-RPC 2.0.

    Handles tool calls and other MCP operations with mandatory OAuth 2.1 authentication
    and automatic user isolation. Compatible with Claude Desktop and other MCP clients.

    Headers:
        Authorization: Bearer <access-token> (REQUIRED)
                      Access token issued by this server's /oauth/token endpoint.

    Authentication:
        OAuth 2.1 is MANDATORY. All requests must include a valid access token.

        Environment Variables Required:
            GOOGLE_OAUTH_CLIENT_ID: Your Google OAuth client ID
            GOOGLE_OAUTH_CLIENT_SECRET: Your Google OAuth client secret
            SERVER_URL: Your server URL
            REDIS_HOST: Redis host for token storage
            REDIS_PORT: Redis port

        User Isolation:
            Users are automatically isolated based on their email from the access token.
            Example: alice@gmail.com → storage directory: alice_at_gmail_dot_com/
    """
    try:
        # Validate access token and extract user email (mandatory authentication)
        user_email = await validate_access_token(authorization)

        # Sanitize email for filesystem usage
        user_id = user_email.replace("@", "_at_").replace(".", "_dot_")

        body = await request.json()

        # Extract JSON-RPC fields
        jsonrpc_version = body.get("jsonrpc")
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        # Validate JSON-RPC version
        if jsonrpc_version != "2.0":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request - must be JSON-RPC 2.0"
                }
            }, status_code=400)

        # Handle different MCP methods
        if method == "tools/list":
            tools = await mcp_server.list_tools()
            # Convert Tool objects to dicts for JSON serialization
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools
            ]
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": tools_list
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_arguments = params.get("arguments", {})

            # Set per-user storage directory
            user_storage_path = get_user_storage_path(user_id)

            # Temporarily override DOCUMENT_ROOT_DIR for this request
            original_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
            os.environ["DOCUMENT_ROOT_DIR"] = user_storage_path

            try:
                # Call the MCP tool with user's isolated storage
                # Returns tuple: (list of TextContent, dict result)
                text_content_list, dict_result = await mcp_server.call_tool(tool_name, tool_arguments)

                # Audit log successful operation
                log_audit_event(
                    user_id=user_id,
                    operation="tools/call",
                    tool_name=tool_name,
                    arguments=tool_arguments,
                    success=True
                )

                # Convert TextContent objects to JSON-serializable dicts
                content_items = [
                    {
                        "type": "text",
                        "text": item.text
                    }
                    for item in text_content_list
                ]

                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": content_items
                    }
                })

            except Exception as e:
                # Audit log failed operation
                log_audit_event(
                    user_id=user_id,
                    operation="tools/call",
                    tool_name=tool_name,
                    arguments=tool_arguments,
                    success=False,
                    error_message=str(e)
                )
                raise  # Re-raise to be handled by outer exception handler

            finally:
                # Restore original DOCUMENT_ROOT_DIR
                if original_doc_root is not None:
                    os.environ["DOCUMENT_ROOT_DIR"] = original_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)

        elif method == "resources/list":
            # Resources not yet implemented, return empty list
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "resources": []
                }
            })

        elif method == "initialize":
            # MCP protocol initialization handshake
            logger.info(f"MCP initialize request from user: {user_id}")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"subscribe": False, "listChanged": False}
                    },
                    "serverInfo": {
                        "name": mcp_server.name,
                        "version": app.version
                    }
                }
            })

        elif method == "notifications/initialized":
            # Client confirms initialization - no response needed for notifications
            logger.info(f"MCP client initialized for user: {user_id}")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            })

        elif method == "ping":
            # Keep-alive ping
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            })

        else:
            # Return -32601 but with 200 status (JSON-RPC errors should be 200)
            logger.warning(f"Unknown MCP method: {method}")
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            })

    except HTTPException as e:
        # Re-raise HTTP exceptions with proper headers for OAuth flow
        if e.status_code == 401:
            server_url = os.environ.get("SERVER_URL", "https://document-mcp-451560119112.asia-east1.run.app")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id if 'request_id' in locals() else None,
                    "error": {
                        "code": -32001,
                        "message": "Unauthorized",
                        "data": str(e.detail)
                    }
                },
                status_code=401,
                headers={
                    "WWW-Authenticate": f'Bearer resource="{server_url}"'
                }
            )
        raise
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}", exc_info=True)
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }, status_code=500)


# Catch-all handler to debug 404 issues
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    """Catch any requests that don't match other routes for debugging."""
    logger.warning(f"Catch-all hit: method={request.method}, path=/{path}, headers={dict(request.headers)}")
    body = await request.body()
    logger.warning(f"Catch-all body: {body[:500] if body else 'empty'}")
    return JSONResponse({
        "error": "Route not found",
        "method": request.method,
        "path": f"/{path}",
        "hint": "This request did not match any defined routes"
    }, status_code=404)


# Development server runner
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload during development
        log_level=os.environ.get("LOG_LEVEL", "info").lower()
    )
