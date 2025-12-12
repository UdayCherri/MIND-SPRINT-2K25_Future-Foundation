# BAYMAX ModeM - Web Interface

This is the web interface for the BAYMAX Mental Health Monitoring System (ModeM). Built with React, TypeScript, and Vite.

## Features

- 📊 Real-time mental health analytics dashboard
- 📝 Daily journal entry with behavioral tracking
- 🎯 Multi-dimensional wellness visualization (radar charts)
- 📈 Timeline tracking of emotional patterns
- 🔍 AI-powered insights and recommendations
- 🎨 Futuristic glassmorphism UI design

## Tech Stack

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI
- **Charts:** Recharts
- **Backend:** Python FastAPI (text_embedding_module.py)

## Setup Instructions

### 1. Install Dependencies

```bash
cd baymax-modem
npm install
```

If you encounter TypeScript errors, also install type definitions:

```bash
npm install --save-dev @types/react @types/react-dom
```

### 2. Configure Environment

The `.env` file is already created with default settings:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Modify if your backend runs on a different port.

### 3. Start Backend Server

Open a new terminal in the project root:

```bash
cd ..
python text_embedding_module.py --serve
```

This starts the FastAPI server on `http://localhost:8000`

### 4. Start Development Server

```bash
npm run dev
```

### 5. Open Application

Open your browser and navigate to:

```
http://localhost:5173
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Project Structure

```
baymax-modem/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Main application pages
│   ├── services/       # API integration (baymaxApi.ts)
│   ├── styles/         # Global styles
│   ├── App.tsx         # Main app component
│   └── main.tsx        # Entry point
├── public/             # Static assets
├── .env                # Environment variables
└── package.json        # Dependencies
```

## API Integration

The frontend connects to the Python backend via the `baymaxApi` service:

**Available Endpoints:**

- `GET /health` - Backend health check
- `POST /embed_text` - Get text embeddings
- `POST /build_day_vector` - Build 771-dim feature vector
- `POST /semantic_drift` - Calculate semantic drift
- `POST /analyze` - Full mental health analysis (TODO)

## Current Status

✅ **Implemented:**

- Complete UI/UX with all pages (Dashboard, Analytics, Insights, Settings)
- API service layer for backend communication
- Mock data for development/testing
- Health check and connection status monitoring

⚠️ **In Progress:**

- Backend `/analyze` endpoint integration
- Real-time fusion model predictions
- Audio recording feature
- User authentication
- Data persistence

## Troubleshooting

### Backend Connection Issues

If you see "Backend offline" warning:

1. Ensure Python backend is running: `python text_embedding_module.py --serve`
2. Check backend health: `curl http://localhost:8000/health`
3. Verify `.env` file has correct `VITE_API_BASE_URL`

### TypeScript Errors

If you see type errors:

```bash
npm install --save-dev @types/react @types/react-dom
```

### Port Already in Use

If port 5173 is occupied:

```bash
npm run dev -- --port 3000
```

## Contributing

This project is part of the MIND_SPRINT_2K25 Hackathon by Future_Foundation team.

## License

See LICENSE file in project root.
