# URL Summarizer API

A secure web API that takes URLs and returns 1-sentence summaries in JSON format.

## Features

- 🔒 SSRF protection (blocks localhost/private IPs)
- 📝 Smart content extraction with priority selectors
- ⚡ Fast response times with request timeouts
- 🛡️ 10MB file size limit protection
- 📊 Comprehensive error handling and logging
- 🌐 CORS enabled for web applications

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Server runs on http://localhost:5000

## API Usage

### POST /summarize

**Request:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "summary": "This webpage titled 'Example Domain' discusses example content...",
  "status": "success"
}
```

### GET /health

Health check endpoint for monitoring.

## Deployment

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Docker
```bash
docker build -t url-summarizer .
docker run -p 5000:5000 url-summarizer
```

## CLI Usage

```bash
python url-summarizer-api.py https://example.com
```

## Security

- Prevents Server-Side Request Forgery (SSRF) attacks
- Blocks access to localhost and private IP ranges
- Implements request timeouts and file size limits
- Comprehensive input validation