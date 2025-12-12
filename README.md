# Universal Quiz Solver API

An autonomous Quiz-Solver Agent that accepts teacher-delivered tasks via HTTP POST, visits JavaScript-rendered quiz pages, interprets instructions, executes data pipelines, and returns correctly formatted JSON answers.

## 🚀 Features

- **Autonomous Quiz Solving**: Handles various quiz types including CSV, PDF, image OCR, ML inference, and API calls
- **JavaScript Rendering**: Uses Playwright Chromium for full JS execution
- **Quiz Chaining**: Supports recursive quiz chains where correct answers lead to new quizzes
- **Timeout Management**: 3-minute execution limit with graceful handling
- **Rule-based + AI Classification**: Two-layer task classification with optional Gemini fallback
- **Multiple Output Formats**: Numbers, strings, JSON, base64 images, booleans

## 📁 Project Structure

```
quiz_solver/
├── app.py                  # FastAPI server with /solve endpoint
├── solver/
│   ├── __init__.py         # Module exports
│   ├── browser.py          # Playwright wrapper
│   ├── parser.py           # HTML/quiz page parser
│   ├── classifier.py       # Task type classifier
│   ├── downloader.py       # Secure file downloader
│   ├── pdf_utils.py        # PDF extraction
│   ├── csv_utils.py        # CSV/Excel processing
│   ├── ocr_utils.py        # Image OCR
│   ├── ml_utils.py         # ML inference
│   ├── chart_utils.py      # Chart generation
│   ├── api_utils.py        # HTTP & Gemini API
│   └── solver_core.py      # Main solving logic
├── models/
│   └── model.pkl           # Dummy ML model
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

## 🔧 Installation

### Local Setup

1. **Clone and navigate to the project**:
```bash
cd quiz_solver
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Playwright browsers**:
```bash
playwright install chromium
```

5. **Install Tesseract OCR** (for image text extraction):
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# Required: Secret for API authentication
SOLVER_SECRET=your_secret_key_here

# Optional: Gemini API for AI-assisted task interpretation
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
```

## 🏃 Running Locally

```bash
# Development mode with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 7860

# Production mode
uvicorn app:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

## 📡 API Usage

### POST /solve

Starts the quiz solving process asynchronously.

**Request**:
```bash
curl -X POST http://localhost:7860/solve \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/quiz",
    "secret": "your_secret_key_here",
    "timeout": 180
  }'
```

**Response**:
```json
{
  "status": "accepted",
  "message": "Quiz solving started",
  "task_id": "abc12345"
}
```

### GET /status/{task_id}

Check the status of a solving task.

```bash
curl http://localhost:7860/status/abc12345
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:7860/health
```

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t quiz-solver .

# Run the container
docker run -p 7860:7860 \
  -e SOLVER_SECRET=your_secret \
  -e GEMINI_API_KEY=your_key \
  quiz-solver
```

## 🤗 Hugging Face Spaces Deployment

1. **Create a new Space** on Hugging Face with Docker SDK

2. **Upload project files** or connect to a Git repository

3. **Set secrets** in Space Settings:
   - `SOLVER_SECRET`: Your authentication secret
   - `GEMINI_API_KEY`: (Optional) Gemini API key

4. **The Dockerfile is pre-configured** for Hugging Face Spaces:
   - Uses port 7860
   - Installs all dependencies including Playwright and Tesseract
   - Creates a non-root user for security

## 🧩 Solver Modules

### Browser (browser.py)
- Playwright Chromium automation
- JavaScript page rendering
- Content extraction (text, images, links)
- Retry logic for failed loads

### Parser (parser.py)
- Quiz question extraction
- Submit URL detection
- File link extraction
- Table parsing from HTML

### Classifier (classifier.py)
- Rule-based keyword matching
- Task type enumeration
- Optional Gemini refinement
- Compound task detection

### Downloader (downloader.py)
- Secure file downloads
- Content type validation
- Automatic ZIP extraction
- Retry with exponential backoff

### PDF Utils (pdf_utils.py)
- Text extraction
- Table extraction
- Image extraction
- Page-specific operations

### CSV Utils (csv_utils.py)
- CSV/TSV/Excel loading
- Column name cleaning
- Statistical operations (sum, mean, median, etc.)
- GroupBy and merge operations

### OCR Utils (ocr_utils.py)
- Image preprocessing
- Text extraction via Tesseract
- Base64 image support
- Number extraction

### ML Utils (ml_utils.py)
- Model loading (joblib/pickle)
- Prediction interface
- Probability outputs

### Chart Utils (chart_utils.py)
- Histogram generation
- Bar charts
- Scatter plots
- Base64 PNG export

### API Utils (api_utils.py)
- HTTP client with retries
- Gemini API wrapper
- Rate limit handling

## 🔍 Troubleshooting

### Playwright Issues

```bash
# Reinstall Playwright with deps
playwright install --with-deps chromium

# Check browser installation
playwright install --dry-run
```

### Tesseract Not Found

```bash
# Verify installation
tesseract --version

# Set path manually in code
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

### Docker Memory Issues

```bash
# Increase Docker memory limit
docker run -m 4g quiz-solver
```

### Permission Denied in Docker

The Dockerfile creates a non-root user. If you encounter permission issues:
```bash
docker run --user root quiz-solver
```

## 📝 Example Quiz Types

1. **CSV Sum**: "Download the CSV, sum the 'value' column"
2. **PDF Table**: "Extract the table from page 2, calculate the total"
3. **Image OCR**: "Extract the 4-digit code from the image"
4. **Visualization**: "Plot a histogram of column X, return as base64"
5. **API Merge**: "Fetch data from the API, merge with CSV, compute average"
6. **ML Prediction**: "Load the model, predict the category"

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Playwright Python](https://playwright.dev/python/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
