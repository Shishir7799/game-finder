# Game Finder Agent: "Snap It, Find It"

Upload a photo of a game (box art/cover **or** an in-game screenshot). Two agents identify it and find where to get it.

## How it works (LangGraph)

```
START -> identify (Agent A) --confident game?--> find_links (Agent B) -> END
                              \--no--> END (asks for a clearer photo)
```

| Agent | Job | Tech |
|---|---|---|
| **A: identify** | Looks at the photo and returns title, platform, confidence and evidence as structured output | Gemini vision via `langchain-google-genai` |
| **B: find_links** | Calls tools to find where the game can be found/bought | `search_web` (DuckDuckGo, free) + `store_search_links` (Steam, Amazon.in, Flipkart, Google Shopping) |

- Framework: LangChain (model, tools, structured output) + LangGraph (state, nodes, conditional edge)
- Free tools only: Gemini free tier, DuckDuckGo search, Render free + Netlify free
- No vector database / RAG in this agent
- Served with FastAPI + LangServe. Playground at `/agent/playground` (use the file picker), plain upload API at `POST /analyze`

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your free key from https://aistudio.google.com
uvicorn app:app --reload
```

- Playground: http://localhost:8000/agent/playground
- API docs: http://localhost:8000/docs
- Frontend: open `frontend/index.html` in a browser (it calls `http://localhost:8000`)

## Deploy

1. **GitHub**: push this folder to a new repo.
2. **Render (backend)**: New > Web Service > connect the repo. It reads `render.yaml`. Add env var `GOOGLE_API_KEY`. Your backend URL looks like `https://game-finder-agent.onrender.com` (free tier sleeps, so the first request is slow).
3. **Frontend**: in `frontend/index.html` change `API_URL` to your Render URL, then drag the `frontend` folder onto https://app.netlify.com/drop (or connect the repo with base directory `frontend`).
4. Optional: set `ALLOWED_ORIGINS` on Render to your Netlify URL to lock CORS down.

## Limits

- PNG/JPG/WEBP only, max 8 MB
- Identification depends on the photo: a visible title/logo works best; low confidence returns a message instead of a guess
