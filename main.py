import os
import urllib.parse
import requests
import base64
import random
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

deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_key) if deepseek_key else None

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
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
            body { background-color: #1e1e1e; color: #ececf1; display: flex; height: 100vh; overflow: hidden; }
            
            .sidebar { width: 260px; background-color: #171717; border-right: 1px solid #2f2f2f; display: flex; flex-direction: column; padding: 15px; }
            .sidebar-header { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            .new-chat-btn { background-color: #2f2f2f; color: #fff; border: 1px solid #424242; padding: 10px; border-radius: 8px; text-align: left; cursor: pointer; font-size: 0.9rem; transition: background 0.2s; }
            .new-chat-btn:hover { background-color: #383838; }

            .main-container { flex: 1; display: flex; flex-direction: column; justify-content: space-between; background-color: #212121; height: 100vh; }
            .chat-header { padding: 15px 25px; border-bottom: 1px solid #2f2f2f; font-size: 1rem; color: #9b9b9b; background-color: #212121; display: flex; justify-content: space-between; align-items: center; }
            
            .provider-selector { background-color: #2f2f2f; color: #fff; border: 1px solid #424242; padding: 6px 12px; border-radius: 6px; outline: none; cursor: pointer; font-size: 0.9rem; }

            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; max-width: 800px; width: 100%; margin: 0 auto; }
            .message-row { display: flex; gap: 15px; width: 100%; }
            .message-row.user { justify-content: flex-end; }
            .message { padding: 12px 18px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem; max-width: 75%; word-wrap: break-word; white-space: pre-wrap; }
            .user-message { background-color: #2f2f2f; color: #fff; border-top-right-radius: 2px; }
            .ai-message { background-color: transparent; color: #ececf1; border-top-left-radius: 2px; padding-left: 0; }
            
            /* Image Watermark Crop & Button Styling */
            .image-wrapper { position: relative; display: inline-block; margin-top: 10px; border-radius: 8px; }
            .gen-image { max-width: 100%; display: block; border-radius: 8px; clip-path: inset(0px 0px 45px 0px); margin-bottom: -45px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
            .download-btn { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.6); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s; display: flex; align-items: center; gap: 6px; backdrop-filter: blur(4px); }
            .download-btn:hover { background: rgba(0,0,0,0.85); border-color: rgba(255,255,255,0.5); }

            .input-area { padding: 20px; background-color: #212121; display: flex; justify-content: center; }
            .input-box-wrapper { max-width: 800px; width: 100%; background-color: #2f2f2f; border: 1px solid #424242; border-radius: 12px; padding: 12px 15px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
            textarea { width: 100%; background: transparent; border: none; color: #fff; font-size: 1rem; outline: none; resize: none; max-height: 150px; }
            .input-footer { display: flex; justify-content: space-between; align-items: center; }
            .model-badge { font-size: 0.8rem; color: #8e8ea0; background: #1f1f1f; padding: 4px 8px; border-radius: 4px; border: 1px solid #383838; }
            button { background-color: #ececf1; color: #212121; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; padding: 6px 16px; }
            button:hover { opacity: 0.85; }
            .loading { color: #8e8ea0; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-header">✨ OpenDev AI</div>
            <button class="new-chat-btn" onclick="location.reload()">+ New Chat</button>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <span>Dummy AI Workspace</span>
                <select id="provider-select" class="provider-selector">
                    <optgroup label="Text Models">
                        <option value="gemini">Gemini API</option>
                        <option value="cohere">Cohere API</option>
                        <option value="openrouter">OpenRouter API</option>
                        <option value="deepseek">DeepSeek API</option>
                        <option value="mistral">Mistral API</option>
                        <option value="huggingface">Hugging Face Text API</option>
                        <option value="sambanova">SambaNova Cloud API</option>
                        <option value="github">GitHub Models API</option>
                    </optgroup>
                    <optgroup label="Image Models">
                        <option value="pollinations_img">Pollinations AI (Image)</option>
                        <option value="hf_img">Hugging Face (SDXL Image)</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="chat-container">
                <div class="message-row">
                    <div class="message ai-message">Hello! I am Dummy AI. My image generator is now upgraded with a Download Button and watermark removal. Try it out!</div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-box-wrapper">
                    <textarea id="user-input" rows="1" placeholder="Type your message here..." onkeypress="handleKeyPress(event)"></textarea>
                    <div class="input-footer">
                        <span class="model-badge">Dummy AI (Ultimate Hub)</span>
                        <button style="background: #19c37d; color: white;" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Image Download Function
            async function downloadImage(url, filename) {
                try {
                    const response = await fetch(url);
                    const blob = await response.blob();
                    const blobUrl = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = blobUrl;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(blobUrl);
                    document.body.removeChild(a);
                } catch (e) {
                    alert('Download failed. You can right-click the image and select "Save image as..."');
                }
            }

            async function sendMessage() {
                const inputField = document.getElementById('user-input');
                const chatContainer = document.getElementById('chat-container');
                const providerSelect = document.getElementById('provider-select');
                
                const text = inputField.value.trim();
                const provider = providerSelect.value;
                const providerName = providerSelect.options[providerSelect.selectedIndex].text;

                if (!text) return;

                const userRow = document.createElement('div');
                userRow.className = 'message-row user';
                userRow.innerHTML = `<div class="message user-message">${text}</div>`;
                chatContainer.appendChild(userRow);

                inputField.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;

                const aiRow = document.createElement('div');
                aiRow.className = 'message-row';
                const aiMsg = document.createElement('div');
                aiMsg.className = 'message ai-message loading';
                aiMsg.innerText = `Generating via ${providerName}...`;
                aiRow.appendChild(aiMsg);
                chatContainer.appendChild(aiRow);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: text, provider: provider })
                    });
                    const data = await response.json();
                    
                    aiMsg.classList.remove('loading');
                    if (response.ok && data.response) {
                        aiMsg.innerHTML = data.response;
                    } else {
                        aiMsg.innerText = data.detail || "API Error: Please check if the API key is valid.";
                    }
                } catch (error) {
                    aiMsg.classList.remove('loading');
                    aiMsg.innerText = "Error connecting to the server. Check your network or server status.";
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
            if not gemini_client: raise Exception("GEMINI_API_KEY is not set.")
            full_prompt = f"System Instructions: {SYSTEM_INSTRUCTION}\n\nUser: {request.prompt}"
            res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            return {"response": res.text}
            
        elif request.provider == "deepseek":
            if not deepseek_client: raise Exception("DEEPSEEK_API_KEY is not set.")
            res = deepseek_client.chat.completions.create(model="deepseek-chat", messages=messages)
            return {"response": res.choices[0].message.content}
            
        elif request.provider == "mistral":
            if not mistral_client: raise Exception("MISTRAL_API_KEY is not set.")
            res = mistral_client.chat.completions.create(model="mistral-tiny", messages=messages)
            return {"response": res.choices[0].message.content}
            
        elif request.provider == "openrouter":
            if not openrouter_client: raise Exception("OPENROUTER_API_KEY is not set.")
            res = openrouter_client.chat.completions.create(model="liquid/lfm-3-1.7b:free", messages=messages)
            return {"response": res.choices[0].message.content}
            
        elif request.provider == "huggingface":
            if not hf_client: raise Exception("HUGGINGFACE_API_KEY is not set.")
            res = hf_client.chat.completions.create(model="microsoft/Phi-3-mini-4k-instruct", messages=messages, max_tokens=500)
            return {"response": res.choices[0].message.content}

        elif request.provider == "sambanova":
            if not sambanova_client: raise Exception("SAMBANOVA_API_KEY is not set.")
            res = sambanova_client.chat.completions.create(model="Meta-Llama-3.1-8B-Instruct", messages=messages)
            return {"response": res.choices[0].message.content}

        elif request.provider == "github":
            if not github_client: raise Exception("GITHUB_TOKEN is not set.")
            res = github_client.chat.completions.create(model="meta-llama-3-8b-instruct", messages=messages)
            return {"response": res.choices[0].message.content}

        elif request.provider == "cohere":
            if not cohere_client: raise Exception("COHERE_API_KEY is not set.")
            res = cohere_client.chat(message=request.prompt, preamble=SYSTEM_INSTRUCTION, model="command-r-08-2024")
            return {"response": res.text}

        # ==================== IMAGE MODELS ====================

        elif request.provider == "pollinations_img":
            enhanced_prompt = f"{request.prompt}, highly detailed, 4k resolution, masterpiece, professional lighting, centered"
            encoded_prompt = urllib.parse.quote(enhanced_prompt)
            seed = random.randint(1, 100000)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&nologo=true"
            
            # Download Button HTML Structure
            html_response = f"""
            <div style="margin-bottom: 10px;">Here is your image:</div>
            <div class='image-wrapper'>
                <img src='{image_url}' class='gen-image' crossorigin='anonymous' />
                <button class='download-btn' onclick='downloadImage("{image_url}", "DummyAI_Image.jpg")'>
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg> Download
                </button>
            </div>
            """
            return {"response": html_response}

        elif request.provider == "hf_img":
            if not hf_key: raise Exception("HUGGINGFACE_API_KEY is not set.")
            api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {hf_key}"}
            
            res = requests.post(api_url, headers=headers, json={"inputs": request.prompt})
            if res.status_code != 200:
                raise Exception("Hugging Face Image API is currently overloaded. Please try Pollinations AI.")
                
            img_b64 = base64.b64encode(res.content).decode("utf-8")
            b64_url = f"data:image/jpeg;base64,{img_b64}"
            
            html_response = f"""
            <div style="margin-bottom: 10px;">Here is your image:</div>
            <div class='image-wrapper'>
                <img src='{b64_url}' class='gen-image' />
                <button class='download-btn' onclick='downloadImage("{b64_url}", "DummyAI_Image.jpg")'>
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg> Download
                </button>
            </div>
            """
            return {"response": html_response}

        else:
            raise Exception("Invalid AI Provider Selected!")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))