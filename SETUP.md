# Setup Instructions

## Prerequisites

Make sure you have the following installed:

- Python 3.11 or higher
- Node.js 18 or higher
- pip (Python package manager)
- npm (Node package manager)

## Backend Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download required models:**
   The fusion model should be in the `models/` directory:

   - `fusion_baseline.h5` - Main prediction model
   - `scaler.joblib` - Feature scaler

3. **Start the backend server:**
   ```bash
   python analyze_api.py
   ```
   The API will run on `http://localhost:8100`

## Frontend Setup

1. **Navigate to frontend directory:**

   ```bash
   cd baymax-modem
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   The app will run on `http://localhost:3000`

## Testing

### Using the Demo Account

- Email: `demo@baymax.ai`
- Password: `demo123`

### Running Your First Analysis

1. Login with demo credentials
2. Fill in the journal entry field
3. Adjust sleep, screen time, and activity sliders
4. Click "Record Audio" (simulated for now)
5. Click "Analyze with BAYMAX"
6. View results in the dashboard

## Troubleshooting

### Backend Issues

**Issue:** Module not found errors

```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue:** Model file not found

```bash
# Solution: Ensure models are in the models/ directory
# Contact team if you need the model files
```

### Frontend Issues

**Issue:** Port 3000 already in use

```bash
# Solution: Kill the process or use a different port
# The dev server will automatically try port 3001
```

**Issue:** CORS errors

```bash
# Solution: Make sure backend is running first
# Check that backend CORS settings include your frontend URL
```

## Environment Variables

Create a `.env` file in the root directory (optional):

```env
# Backend
PORT=8100
DEBUG=True

# Frontend (in baymax-modem/)
VITE_ANALYZE_API_URL=http://localhost:8100
```

## Development Mode

Both servers support hot-reload:

- Backend: Uses `uvicorn` with `--reload` flag
- Frontend: Vite automatically reloads on file changes

## Building for Production

### Backend

```bash
# Use gunicorn for production
gunicorn analyze_api:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend

```bash
cd baymax-modem
npm run build
# Output in baymax-modem/dist/
```

## Need Help?

- Check the [README.md](README.md) for more details
- See [ROADMAP.md](ROADMAP.md) for planned features
- Read [DEVLOG.md](DEVLOG.md) for development notes

---

**Team Future Foundation** | MIND_SPRINT 2K25
