import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web


def test_graph_passes_supabase_token_to_rag_tool(monkeypatch):
    """Ensure the graph wires the Supabase access token into RAG tool creation."""

    async def run_test():
        # Arrange stubs to avoid hitting external services when building the graph
        create_react_stub = MagicMock(return_value="fake-agent")
        monkeypatch.setattr("tools_agent.agent.create_react_agent", create_react_stub)

        init_chat_model_stub = MagicMock(return_value="fake-model")
        monkeypatch.setattr("tools_agent.agent.init_chat_model", init_chat_model_stub)

        rag_tool_stub = AsyncMock(return_value="rag-tool-instance")
        monkeypatch.setattr("tools_agent.agent.create_rag_tool", rag_tool_stub)

        config = {
            "configurable": {
                "model_name": "openai:gpt-4o-mini",
                "rag": {
                    "rag_url": "https://langconnect-production-db56.up.railway.app",
                    "collections": ["collection-123"],
                },
                "x-supabase-access-token": "supabase-access-token",
            }
        }

        from tools_agent.agent import graph

        # Act
        agent = await graph(config)

        # Assert: the RAG tool should receive the Supabase token and collection info
        rag_tool_stub.assert_awaited_once_with(
            "https://langconnect-production-db56.up.railway.app",
            "collection-123",
            "supabase-access-token",
        )
        assert agent == "fake-agent"

    asyncio.run(run_test())


def test_create_rag_tool_uses_bearer_token_for_collection_and_search():
    """Verify RAG tool bootstrapping and searches send the Supabase bearer token."""

    async def run_test():
        recorded_requests: list[tuple[str, str]] = []

        async def collection_handler(request: web.Request) -> web.Response:
            recorded_requests.append(("collection", request.headers.get("Authorization")))
            return web.json_response(
                {
                    "id": "collection-123",
                    "name": "LangConnect Collection",
                    "metadata": {"description": "Sample documents"},
                }
            )

        async def search_handler(request: web.Request) -> web.Response:
            recorded_requests.append(("search", request.headers.get("Authorization")))
            payload = await request.json()
            assert payload["query"] == "auth test"
            return web.json_response(
                [
                    {"id": "doc-1", "page_content": "Document 1 content"},
                    {"id": "doc-2", "page_content": "Document 2 content"},
                ]
            )

        app = web.Application()
        app.router.add_get("/collections/collection-123", collection_handler)
        app.router.add_post(
            "/collections/collection-123/documents/search", search_handler
        )

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()

        try:
            port = site._server.sockets[0].getsockname()[1]
            base_url = f"http://127.0.0.1:{port}"

            from tools_agent.utils.tools import create_rag_tool

            rag_tool = await create_rag_tool(
                base_url, "collection-123", "supabase-access-token"
            )

            # Invoke the tool to trigger the search call
            result = await rag_tool.coroutine(query="auth test")

            assert "<document id=\"doc-1\">" in result
            assert "<document id=\"doc-2\">" in result
            assert recorded_requests == [
                ("collection", "Bearer supabase-access-token"),
                ("search", "Bearer supabase-access-token"),
            ]
        finally:
            await runner.cleanup()

    asyncio.run(run_test())
