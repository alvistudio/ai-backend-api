import os
import urllib.parse
import requests
import base64
import random
import io
from PIL import Image as PILImage
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from openai import OpenAI
import cohere

app = FastAPI()

# ==========================================
# API Clients Initialization
# ==========================================

gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

openrouter_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key) if openrouter_key else None

hf_key = os.environ.get("HUGGINGFACE_API_KEY")
hf_client = OpenAI(base_url="https://api-inference.huggingface.co/v1/", api_key=hf_key) if hf_key else None

sambanova_key = os.environ.get("SAMBANOVA_API_KEY")
sambanova_client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=sambanova_key) if sambanova_key else None

github_key = os.environ.get("GITHUB_TOKEN")
github_client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=github_key) if github_key else None

cohere_key = os.environ.get("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_key) if cohere_key else None

mistral_key = os.environ.get("MISTRAL_API_KEY")
mistral_client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=mistral_key) if mistral_key else None

# ==========================================
# Data Models & System Instructions
# ==========================================

class ChatRequest(BaseModel):
    prompt: str
    provider: str = "gemini"

SYSTEM_INSTRUCTION = (
    "Your name is Dummy AI. You were created by OpenDev. "
    "Whenever someone asks about your name, identity, or who created you, you must state that your name is Dummy AI and you were created by OpenDev. "
    "Respond in the same language the user uses."
)

# ==========================================
# Routes
# ==========================================

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dummy AI - OpenDev</title>
        <style>
            :root { --bg-main: #1e1e1e; --bg-sidebar: #171717; --bg-chat: #212121; --bg-input: #2f2f2f; --text-main: #ececf1; --text-muted: #9b9b9b; --accent: #19c37d; }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
            body { background-color: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }
            
            /* Sidebar */
            .sidebar { width: 260px; background-color: var(--bg-sidebar); border-right: 1px solid #333; display: flex; flex-direction: column; padding: 15px; transition: all 0.3s ease; }
            .sidebar-header { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            .new-chat-btn { background-color: var(--bg-input); color: #fff; border: 1px solid #424242; padding: 10px; border-radius: 8px; text-align: left; cursor: pointer; font-size: 0.9rem; transition: 0.2s; }
            .new-chat-btn:hover { background-color: #383838; }

            /* Main Chat Area */
            .main-container { flex: 1; display: flex; flex-direction: column; background-color: var(--bg-chat); position: relative; }
            .chat-header { padding: 15px 25px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; background: var(--bg-chat); z-index: 10; }
            .provider-selector { background-color: var(--bg-input); color: #fff; border: 1px solid #424242; padding: 8px 12px; border-radius: 8px; outline: none; cursor: pointer; font-size: 0.9rem; }
            
            #chat-container { flex: 1; overflow-y: auto; padding: 30px 20px; display: flex; flex-direction: column; gap: 24px; scroll-behavior: smooth; }
            .chat-wrapper { max-width: 800px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
            
            .message-row { display: flex; width: 100%; gap: 16px; }
            .message-row.user { justify-content: flex-end; }
            
            .message { padding: 14px 20px; border-radius: 16px; line-height: 1.6; font-size: 0.95rem; word-wrap: break-word; white-space: pre-wrap; max-width: 85%; }
            .user-message { background-color: var(--bg-input); color: #fff; border-bottom-right-radius: 4px; }
            .ai-message { background-color: transparent; color: var(--text-main); border-bottom-left-radius: 4px; padding-left: 0; width: 100%; }
            .ai-content { margin-bottom: 10px; }

            /* Action Buttons (Copy, Regenerate, Like, Dislike) */
            .message-actions { display: flex; gap: 8px; align-items: center; margin-top: 8px; opacity: 0.7; transition: opacity 0.2s; }
            .message-row:hover .message-actions { opacity: 1; }
            .action-btn { background: transparent; border: none; color: var(--text-muted); cursor: pointer; padding: 6px; border-radius: 6px; display: flex; align-items: center; justify-content: center; transition: 0.2s; }
            .action-btn:hover { background: var(--bg-input); color: #fff; }
            .action-btn svg { width: 16px; height: 16px; fill: currentColor; }

            /* Image Styling */
            .image-wrapper { position: relative; display: inline-block; margin-top: 10px; border-radius: 12px; }
            .gen-image { max-width: 100%; display: block; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .download-btn { position: absolute; top: 12px; right: 12px; background: rgba(0,0,0,0.65); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 6px; backdrop-filter: blur(5px); transition: 0.2s; }
            .download-btn:hover { background: rgba(0,0,0,0.9); }

            /* Input Area */
            .input-area { padding: 20px; background: linear-gradient(180deg, rgba(33,33,33,0) 0%, var(--bg-chat) 30%); }
            .input-box-wrapper { max-width: 800px; width: 100%; margin: 0 auto; background-color: var(--bg-input); border: 1px solid #424242; border-radius: 16px; padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
            textarea { width: 100%; background: transparent; border: none; color: #fff; font-size: 1rem; outline: none; resize: none; max-height: 200px; line-height: 1.5; }
            .input-footer { display: flex; justify-content: space-between; align-items: center; }
            .model-badge { font-size: 0.75rem; color: var(--text-muted); }
            .send-btn { background-color: var(--text-main); color: var(--bg-chat); border: none; border-radius: 8px; cursor: pointer; font-weight: 600; padding: 8px 18px; transition: 0.2s; }
            .send-btn:hover { background-color: #fff; transform: translateY(-1px); }
            .loading-text { color: var(--text-muted); font-style: italic; display: flex; align-items: center; gap: 8px; }
            
            /* Responsive */
            @media (max-width: 768px) {
                .sidebar { display: none; }
                .message { max-width: 95%; }
                .chat-header { padding: 12px 15px; }
                .input-area { padding: 15px; }
                .provider-selector { max-width: 150px; font-size: 0.8rem; }
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-header">✨ OpenDev AI</div>
            <button class="new-chat-btn" onclick="location.reload()">+ New Chat</button>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <span style="font-weight: 600;">Dummy AI</span>
                <select id="provider-select" class="provider-selector">
                    <optgroup label="Text Models">
                        <option value="gemini">Gemini API</option>
                        <option value="cohere">Cohere API</option>
                        <option value="openrouter">OpenRouter API</option>
                        <option value="mistral">Mistral API</option>
                        <option value="huggingface">Hugging Face Text</option>
                        <option value="sambanova">SambaNova Cloud</option>
                        <option value="github">GitHub Models</option>
                    </optgroup>
                    <optgroup label="Image Models">
                        <option value="pollinations_img">Pollinations AI (Image)</option>
                        <option value="hf_img">Hugging Face (SDXL Image)</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="chat-container">
                <div class="chat-wrapper" id="chat-wrapper">
                    <div class="message-row">
                        <div class="message ai-message">
                            <div class="ai-content">Hello! I am Dummy AI. I have been upgraded with a Claude-inspired UI, responsive design, and action buttons. How can I help you today?</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-box-wrapper">
                    <textarea id="user-input" rows="1" placeholder="Message Dummy AI..." oninput="autoResize(this)" onkeypress="handleKeyPress(event)"></textarea>
                    <div class="input-footer">
                        <span class="model-badge">Dummy AI v2.0</span>
                        <button class="send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let lastUserPrompt = "";

            function autoResize(textarea) {
                textarea.style.height = 'auto';
                textarea.style.height = textarea.scrollHeight + 'px';
            }

            function downloadImage(dataUrl, filename) {
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }

            function copyText(btn) {
                const contentDiv = btn.closest('.ai-message').querySelector('.ai-content');
                if(contentDiv) {
                    navigator.clipboard.writeText(contentDiv.innerText);
                    const originalHTML = btn.innerHTML;
                    btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';
                    setTimeout(() => btn.innerHTML = originalHTML, 2000);
                }
            }

            function toggleAction(btn) {
                btn.style.color = btn.style.color === 'var(--accent)' ? 'var(--text-muted)' : 'var(--accent)';
            }

            function regenerateLast() {
                if (lastUserPrompt) {
                    document.getElementById('user-input').value = lastUserPrompt;
                    sendMessage(true);
                }
            }

            function getActionButtonsHTML() {
                return `
                <div class="message-actions">
                    <button class="action-btn" onclick="copyText(this)" title="Copy">
                        <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
                    </button>
                    <button class="action-btn" onclick="regenerateLast()" title="Regenerate">
                        <svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
                    </button>
                    <button class="action-btn" onclick="toggleAction(this)" title="Good response">
                        <svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/></svg>
                    </button>
                    <button class="action-btn" onclick="toggleAction(this)" title="Bad response">
                        <svg viewBox="0 0 24 24"><path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/></svg>
                    </button>
                </div>`;
            }

            async function sendMessage(isRegenerate = false) {
                const inputField = document.getElementById('user-input');
                const chatWrapper = document.getElementById('chat-wrapper');
                const chatContainer = document.getElementById('chat-container');
                const providerSelect = document.getElementById('provider-select');
                
                const text = inputField.value.trim();
                const provider = providerSelect.value;
                const providerName = providerSelect.options[providerSelect.selectedIndex].text;

                if (!text) return;
                if (!isRegenerate) lastUserPrompt = text;

                const userRow = document.createElement('div');
                userRow.className = 'message-row user';
                userRow.innerHTML = `<div class="message user-message">${text}</div>`;
                chatWrapper.appendChild(userRow);

                inputField.value = '';
                inputField.style.height = 'auto';
                chatContainer.scrollTop = chatContainer.scrollHeight;

                const aiRow = document.createElement('div');
                aiRow.className = 'message-row';
                const aiMsg = document.createElement('div');
                aiMsg.className = 'message ai-message';
                aiMsg.innerHTML = `<div class="ai-content loading-text">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-loader" style="animation: spin 1s linear infinite;"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line><style>@keyframes spin { 100% { transform: rotate(360deg); } }</style></svg>
                    Generating via ${providerName}...
                </div>`;
                aiRow.appendChild(aiMsg);
                chatWrapper.appendChild(aiRow);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: text, provider: provider })
                    });
                    const data = await response.json();
                    
                    if (response.ok && data.response) {
                        aiMsg.innerHTML = `<div class="ai-content">${data.response}</div>` + getActionButtonsHTML();
                    } else {
                        aiMsg.innerHTML = `<div class="ai-content" style="color: #ff6b6b;">${data.detail || "Error processing request."}</div>` + getActionButtonsHTML();
                    }
                } catch (error) {
                    aiMsg.innerHTML = `<div class="ai-content" style="color: #ff6b6b;">Network error. Please try again.</div>` + getActionButtonsHTML();
                }
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat_with_ai(request: ChatRequest):
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": request.prompt}
    ]
    
    try:
        # ==================== TEXT MODELS ====================
        if request.provider == "gemini":
            if not gemini_client: raise Exception("GEMINI_API_KEY is missing.")
            full_prompt = f"System Instructions: {SYSTEM_INSTRUCTION}\n\nUser: {request.prompt}"
            res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            return {"response": res.text}
            
        elif request.provider == "mistral":
            if not mistral_client: raise Exception("MISTRAL_API_KEY is missing.")
            res = mistral_client.chat.completions.create(model="mistral-tiny", messages=messages)
            return {"response": res.choices[0].message.content}
            
        elif request.provider == "openrouter":
            if not openrouter_client: raise Exception("OPENROUTER_API_KEY is missing.")
            res = openrouter_client.chat.completions.create(model="liquid/lfm-3-1.7b:free", messages=messages)
            return {"response": res.choices[0].message.content}
            
        elif request.provider == "huggingface":
            if not hf_client: raise Exception("HUGGINGFACE_API_KEY is missing.")
            res = hf_client.chat.completions.create(model="mistralai/Mistral-7B-Instruct-v0.2", messages=messages, max_tokens=500)
            return {"response": res.choices[0].message.content}

        elif request.provider == "sambanova":
            if not sambanova_client: raise Exception("SAMBANOVA_API_KEY is missing.")
            res = sambanova_client.chat.completions.create(model="Meta-Llama-3.1-8B-Instruct", messages=messages)
            return {"response": res.choices[0].message.content}

        elif request.provider == "github":
            if not github_client: raise Exception("GITHUB_TOKEN is missing.")
            res = github_client.chat.completions.create(model="meta-llama-3-8b-instruct", messages=messages)
            return {"response": res.choices[0].message.content}

        elif request.provider == "cohere":
            if not cohere_client: raise Exception("COHERE_API_KEY is missing.")
            res = cohere_client.chat(message=request.prompt, preamble=SYSTEM_INSTRUCTION, model="command-r-08-2024")
            return {"response": res.text}

        # ==================== IMAGE MODELS ====================
        elif request.provider == "pollinations_img":
            enhanced_prompt = f"{request.prompt}, highly detailed, 4k resolution, masterpiece, professional lighting, centered"
            encoded_prompt = urllib.parse.quote(enhanced_prompt)
            seed = random.randint(1, 100000)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&nologo=true"
            
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                img = PILImage.open(io.BytesIO(img_response.content))
                width, height = img.size
                img_cropped = img.crop((0, 0, width, height - 45))
                
                buffered = io.BytesIO()
                img_cropped.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                b64_url = f"data:image/jpeg;base64,{img_b64}"
                
                return {"response": f"""
                <div class='image-wrapper'>
                    <img src='{b64_url}' class='gen-image' />
                    <button class='download-btn' onclick='downloadImage("{b64_url}", "DummyAI_Image.jpg")'>
                        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg> Download
                    </button>
                </div>
                """}
            else:
                raise Exception("Image generation failed.")

        elif request.provider == "hf_img":
            if not hf_key: raise Exception("HUGGINGFACE_API_KEY is missing.")
            api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {hf_key}"}
            
            try:
                res = requests.post(api_url, headers=headers, json={"inputs": request.prompt}, timeout=20)
                if res.status_code != 200:
                    raise Exception("Model is loading. Try again in 10 seconds.")
                    
                img_b64 = base64.b64encode(res.content).decode("utf-8")
                b64_url = f"data:image/jpeg;base64,{img_b64}"
                
                return {"response": f"""
                <div class='image-wrapper'>
                    <img src='{b64_url}' class='gen-image' />
                    <button class='download-btn' onclick='downloadImage("{b64_url}", "DummyAI_Image.jpg")'>
                        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg> Download
                    </button>
                </div>
                """}
            except requests.exceptions.RequestException:
                raise Exception("Hugging Face server timeout. Please try Pollinations AI.")

        else:
            raise Exception("Invalid Provider.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))