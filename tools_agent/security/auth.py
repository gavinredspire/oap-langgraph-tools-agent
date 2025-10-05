import os
import asyncio
import logging
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from supabase import create_client, Client
from typing import Optional, Any

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

logger = logging.getLogger(__name__)

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


# The `authenticate` decorator tells LangGraph to call this function as middleware
# for every request. This will determine whether the request is allowed or not
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Check if the user's JWT token is valid using Supabase."""

    # Ensure we have authorization header
    if not authorization:
        logger.warning("Auth: missing Authorization header")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    # Parse the authorization header
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        logger.warning("Auth: invalid Authorization header format")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    # Ensure Supabase client is initialized
    if not supabase:
        logger.error("Auth: Supabase client not initialized - check SUPABASE_URL/KEY env vars")
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Supabase client not initialized"
        )

    try:
        # Verify the JWT token with Supabase using asyncio.to_thread to avoid blocking
        # This will decode and verify the JWT token in a separate thread
        async def verify_token() -> dict[str, Any]:
            response = await asyncio.to_thread(supabase.auth.get_user, token)
            return response

        response = await verify_token()
        user = response.user

        if not user:
            logger.warning("Auth: token verified but user missing")
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Invalid token or user not found"
            )

        # Return user info if valid, include token in metadata for downstream hooks
        logger.info(
            "Auth: authenticated user; storing token in metadata (len=%s)",
            len(token) if isinstance(token, str) else "n/a",
        )
        return {
            "identity": user.id,
            "metadata": {"supabase_token": token},
        }
    except Exception as e:
        # Handle any errors from Supabase
        logger.exception("Auth: error during authentication: %s", e)
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Authentication error: {str(e)}"
        )


@auth.on.threads.create
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """Add owner when creating threads.

    This handler runs when creating new threads and does two things:
    1. Sets metadata on the thread being created to track ownership
    2. Returns a filter that ensures only the creator can access it
    """

    if isinstance(ctx.user, StudioUser):
        logger.info("Auth Hook: StudioUser detected; skipping token injection")
        return

    # Inject Supabase token into run configuration if available
    # so downstream graph code can call external services on behalf of the user
    supabase_token = None
    try:
        # Prefer token stored on the user metadata (set in authenticate step)
        if hasattr(ctx, "user") and getattr(ctx.user, "metadata", None):
            supabase_token = ctx.user.metadata.get("supabase_token")
    except Exception:
        supabase_token = None

    if supabase_token:
        configurable = value.setdefault("configurable", {})
        configurable["x-supabase-access-token"] = supabase_token
        logger.info(
            "Auth Hook: injected Supabase token into run config (len=%s)",
            len(supabase_token) if isinstance(supabase_token, str) else "n/a",
        )
    else:
        logger.warning("Auth Hook: no Supabase token found in user metadata; not injecting")

    # Add owner metadata to the thread being created
    # This metadata is stored with the thread and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    logger.info("Auth Hook: set thread owner to %s", ctx.user.identity)


@auth.on.threads.create_run
async def on_thread_create_run(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create_run.value,
):
    """Inject Supabase token into the run configuration for create_run events."""

    if isinstance(ctx.user, StudioUser):
        logger.info("Auth Hook (create_run): StudioUser detected; skipping token injection")
        return

    token: Optional[str] = None
    try:
        if hasattr(ctx.user, "metadata") and ctx.user.metadata:
            token = ctx.user.metadata.get("supabase_token")
    except Exception:
        token = None

    if token:
        cfg = value.setdefault("config", {})
        configurable = cfg.setdefault("configurable", {})
        configurable["x-supabase-access-token"] = token
        logger.info(
            "Auth Hook (create_run): injected Supabase token into config.configurable (len=%s)",
            len(token),
        )
    else:
        logger.warning("Auth Hook (create_run): no Supabase token found; not injecting")


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """Only let users read their own threads.

    This handler runs on read operations. We don't need to set
    metadata since the thread already exists - we just need to
    return a filter to ensure users can only see their own threads.
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    if isinstance(ctx.user, StudioUser):
        return

    # Add owner metadata to the assistant being created
    # This metadata is stored with the assistant and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """Only let users read their own assistants.

    This handler runs on read operations. We don't need to set
    metadata since the assistant already exists - we just need to
    return a filter to ensure users can only see their own assistants.
    """

    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    if isinstance(ctx.user, StudioUser):
        return

    # The "namespace" field for each store item is a tuple you can think of as the directory of an item.
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "Not authorized"
