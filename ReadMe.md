# Deploy LLM to Edge Device
A browser-based AI chatbot powered by WebLLM that runs W4A16 quantized Llama-3.2-3B-Instruct model locally using WebGPU.

## How to Run

1. **Navigate to the project directory**:
   ```bash
   cd path/to/project-directory
   ```

2. **Start a local HTTP server**:
   ```bash
   python -m http.server 8000
   ```

3. **Open in browser**:
   ```
   http://localhost:8000/test.html
   ```

## File Structure

```
├── test.html          # Main HTML interface
├── test.css           # Styling and layout
├── test.js            # WebLLM logic and GPU monitoring
└── ReadMe.md          # This file
```
