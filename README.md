# BAYMAX: ModeM - Multi-Modal Mental Wellness AI

> **Team:** Future Foundation  
> **Hackathon:** MIND_SPRINT 2K25  
> **Status:** In Development 🚧

## 🎯 Project Vision

BAYMAX: ModeM is an AI-powered mental wellness companion that combines multi-modal analysis (text, audio, behavioral data) to detect early warning signs of burnout and provide personalized mental health insights.

## 🧠 Core Features

### ✅ Implemented

- **Multi-Modal Fusion Model**: Combines text embeddings, audio analysis, and behavioral metrics
- **Pattern Detection**: Identifies 7 mental wellness patterns (stress loops, burnout, emotional volatility, etc.)
- **Real-time Analysis**: FastAPI backend with TensorFlow model inference
- **Explainability Engine**: SHAP-based feature importance for transparent predictions
- **User Authentication**: Secure login/registration system
- **Data Persistence**: Historical analysis tracking and trend detection
- **Interactive Dashboard**: React-based UI with live data visualization

### 🚧 In Progress

- **Personalization Adapters**: User-specific model fine-tuning
- **Voice Analysis**: Enhanced audio emotion recognition
- **Temporal Pattern Detection**: Long-term behavioral trend analysis
- **Mobile Responsiveness**: Optimizing UI for smaller screens

### 📋 Planned Features

- **Intervention Recommendations**: AI-generated wellness action plans
- **Integration with Wearables**: Fitbit, Apple Watch data ingestion
- **Therapist Dashboard**: Professional insights view
- **Group Dynamics**: Team burnout risk assessment

## 🏗️ Architecture

```
├── Backend (Python + FastAPI)
│   ├── analyze_api.py          # Main API endpoints
│   ├── fusion_model.py          # Multi-modal ML model
│   ├── text_embedding_module.py # Text processing
│   ├── explainability_engine.py # Interpretability layer
│   ├── data_storage.py          # Data persistence
│   └── user_auth.py             # Authentication
│
├── Frontend (React + TypeScript)
│   └── baymax-modem/
│       ├── src/
│       │   ├── pages/           # Dashboard, Analytics, Insights
│       │   ├── components/      # Reusable UI components
│       │   ├── services/        # API client
│       │   └── contexts/        # State management
│       └── public/
│
└── Models
    ├── fusion_baseline.h5       # Pre-trained fusion model
    └── scaler.joblib            # Feature normalization
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- 8GB+ RAM (for ML models)

### Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python analyze_api.py
# Runs on http://localhost:8100
```

### Frontend Setup

```bash
cd baymax-modem
npm install
npm run dev
# Runs on http://localhost:3000
```

### Demo Account

- Email: `demo@baymax.ai`
- Password: `demo123`

## 🧪 Testing

### Run Analysis

1. Login with demo credentials
2. Fill in journal entry and behavioral metrics
3. Click "Analyze with BAYMAX"
4. View multi-modal predictions and insights

### API Endpoints

- `GET /health` - Check API status
- `POST /auth/login` - User authentication
- `POST /analyze` - Run mental wellness analysis
- `GET /data/timeline/{user_id}` - Get historical data

## 📊 Tech Stack

**Backend:**

- FastAPI (API framework)
- TensorFlow 2.x (ML models)
- Sentence Transformers (Text embeddings)
- NumPy/Pandas (Data processing)

**Frontend:**

- React 18 (UI framework)
- TypeScript (Type safety)
- Recharts (Data visualization)
- Tailwind CSS (Styling)
- Framer Motion (Animations)

**ML Models:**

- Sentence-BERT for text encoding
- Custom fusion neural network
- SHAP for explainability

## 👥 Team Future Foundation

This project represents our team's work during MIND_SPRINT 2K25. We're passionate about using AI for social good and mental health awareness.

**Contributions:**

- Multi-modal model architecture and training
- Backend API development
- Frontend UI/UX design
- Data pipeline implementation
- Documentation

## 📝 Development Notes

### Current Limitations

- Audio analysis uses simulated embeddings (real wav2vec2 integration pending)
- Limited to 7 pattern types (expanding to 15+)
- Single-user mode (multi-tenancy in development)
- Dataset size: ~1000 samples (scaling to 10K+)

### Known Issues

- [ ] Occasional CORS errors on first load (refresh fixes)
- [ ] Timeline graph needs more data points for accuracy
- [ ] Mobile UI needs optimization
- [ ] Model confidence calibration needed

## 🔒 Privacy & Ethics

- All data stored locally during hackathon
- No PII sent to external services
- User consent required for analysis
- Transparent AI decisions via explainability module

## 📚 References

- RAVDESS Emotional Speech Dataset
- Sentence-Transformers Documentation
- Mental Health Pattern Research Papers
- FastAPI Best Practices

## 🤝 Acknowledgments

Thanks to MIND_SPRINT 2K25 organizers and mentors for this opportunity to work on meaningful AI applications in mental health.

---

**Note:** This is a hackathon prototype. Not intended for clinical use. Always consult mental health professionals for serious concerns.

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ by Team Future Foundation**
