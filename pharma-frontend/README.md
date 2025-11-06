# Pharma List Frontend (React + TypeScript)

This is a ready-to-run frontend scaffold for the Pharma List Management system with:
- Modern futuristic UI (Tailwind + glass effects)
- Persistent **ListBot** sidebar (conversational assistant)
- Dashboard, List view, History view
- CSV bulk upload preview (client-side using PapaParse)
- Mocked localStorage backend for rapid front-end development

## Quick start

1. Extract the zip and open the project folder:
```bash
cd pharma-list-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start dev server:
```bash
npm run dev
```

4. Open http://localhost:5173

## Notes for backend integration

The frontend currently uses a small mocked API stored in `src/api/listApi.ts`. When your FastAPI backend is ready, replace those functions with HTTP calls to your real endpoints.

Key endpoints the frontend expects:
- `GET /api/lists?domain=` -> list summaries
- `GET /api/lists/:id` -> full list detail (current_snapshot and versions)
- `POST /api/lists` -> create list (accepts initial_snapshot)
- `POST /api/lists/:id/items/bulk` -> add items (body: items[])
- `GET /api/lists/:id/versions` -> version history
- `POST /api/listbot/query` -> ListBot conversational queries

I'll provide a concrete OpenAPI spec / payload examples you can give to the backend team next.

