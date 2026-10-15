"""Game Finder Agent ("Snap It, Find It")

Two agents wired together with LangGraph:
  Agent A  identify    - Gemini vision looks at the photo and names the game
  Agent B  find_links  - searches the web (free DuckDuckGo tool) + builds store-search links
A conditional edge stops the graph early if the photo is not a game / confidence is low.
"""
import base64
import os
from typing import Any, Dict, List, TypedDict
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langserve import add_routes
from pydantic import BaseModel, Field

load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.1-flash-lite-preview")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MIN_CONFIDENCE = 0.4

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini client lazily so the app can start without a key."""
    global _llm
    if _llm is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set on the server.")
        _llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=key, temperature=0)
    return _llm


# ------------------------------------------------------------------ tools
@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """Search the web (DuckDuckGo, free) and return title/url/snippet results."""
    try:
        from ddgs import DDGS

        hits = DDGS().text(query, max_results=5) or []
        return [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits
            if h.get("href")
        ]
    except Exception:
        return []


@tool
def store_search_links(title: str) -> List[Dict[str, str]]:
    """Build direct search links for the game on popular stores."""
    q = quote_plus(title)
    return [
        {"source": "Steam", "url": f"https://store.steampowered.com/search/?term={q}"},
        {"source": "Amazon.in", "url": f"https://www.amazon.in/s?k={q}"},
        {"source": "Flipkart", "url": f"https://www.flipkart.com/search?q={q}"},
        {"source": "Google Shopping", "url": f"https://www.google.com/search?tbm=shop&q={q}"},
    ]


# ------------------------------------------------------------------ state
class GameGuess(BaseModel):
    is_game: bool = Field(description="True if the image shows a video game, board game or its box/cover.")
    title: str = Field(description="Best guess of the game's title, or empty if unknown.")
    platform: str = Field(description="Platform/edition if visible (PS5, PC, Switch, board game...), else empty.")
    confidence: float = Field(description="0.0 to 1.0 confidence in the title.")
    evidence: str = Field(description="One sentence: what in the image led to this guess.")


class AgentState(TypedDict, total=False):
    image_b64: str
    mime: str
    game: Dict[str, Any]
    web_results: List[Dict[str, str]]
    store_links: List[Dict[str, str]]
    message: str


# ------------------------------------------------------------------ agents (graph nodes)
def identify_node(state: AgentState) -> AgentState:
    """Agent A: identify the game from a cover photo or an in-game screenshot."""
    prompt = (
        "You are a game-identification expert. The image is either a box-art/cover photo or a "
        "screenshot from inside a game. Identify the game. If you are not sure, lower the "
        "confidence instead of guessing wildly. If the image is not a game, set is_game=false."
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{state['mime']};base64,{state['image_b64']}"},
        ]
    )
    guess = get_llm().with_structured_output(GameGuess).invoke([msg])
    return {"game": guess.model_dump()}


def route_after_identify(state: AgentState) -> str:
    g = state["game"]
    if g["is_game"] and g["title"].strip() and g["confidence"] >= MIN_CONFIDENCE:
        return "find_links"
    return END


def find_links_node(state: AgentState) -> AgentState:
    """Agent B: use tools to find where the game can be found/bought."""
    g = state["game"]
    title = g["title"].strip()
    query = f"{title} {g['platform']} buy".strip()
    return {
        "web_results": search_web.invoke({"query": query}),
        "store_links": store_search_links.invoke({"title": title}),
    }


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("identify", identify_node)
    g.add_node("find_links", find_links_node)
    g.add_edge(START, "identify")
    g.add_conditional_edges("identify", route_after_identify, {"find_links": "find_links", END: END})
    g.add_edge("find_links", END)
    return g.compile()


graph = build_graph()


# ------------------------------------------------------------------ helpers
def sniff_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image. Please upload a PNG, JPG or WEBP photo.")


def run_agent(raw: bytes) -> Dict[str, Any]:
    if not raw:
        raise ValueError("Empty file.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (max 8 MB).")
    mime = sniff_mime(raw)
    out = graph.invoke({"image_b64": base64.b64encode(raw).decode(), "mime": mime})
    game = out["game"]
    found = "web_results" in out
    return {
        "identified": found,
        "game": game,
        "web_results": out.get("web_results", []),
        "store_links": out.get("store_links", []),
        "message": "" if found else "I couldn't confidently identify a game in this photo. Try a clearer cover shot or a screenshot with a visible title/logo.",
    }


# ------------------------------------------------------------------ API
app = FastAPI(title="Game Finder Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "agent": "game-finder", "playground": "/agent/playground", "docs": "/docs"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    try:
        return run_agent(file.file.read(MAX_IMAGE_BYTES + 1))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="The AI service failed. Please try again.")


class ImageInput(BaseModel):
    image: str = Field(..., json_schema_extra={"widget": {"type": "base64file"}})


def _from_playground(x) -> Dict[str, Any]:
    b64 = x["image"] if isinstance(x, dict) else x.image
    if "," in b64[:64]:  # strip a "data:image/png;base64," prefix if present
        b64 = b64.split(",", 1)[1]
    return run_agent(base64.b64decode(b64))


add_routes(
    app,
    RunnableLambda(_from_playground).with_types(input_type=ImageInput, output_type=dict),
    path="/agent",
    playground_type="default",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
