import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Assistant - Claude Style</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
            body { background-color: #1e1e1e; color: #ececf1; display: flex; height: 100vh; overflow: hidden; }
            
            /* Sidebar */
            .sidebar { width: 260px; background-color: #171717; border-right: 1px solid #2f2f2f; display: flex; flex-direction: column; padding: 15px; }
            .sidebar-header { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            .new-chat-btn { background-color: #2f2f2f; color: #fff; border: 1px solid #424242; padding: 10px; border-radius: 8px; text-align: left; cursor: pointer; font-size: 0.9rem; transition: background 0.2s; }
            .new-chat-btn:hover { background-color: #383838; }

            /* Main Chat Area */
            .main-container { flex: 1; display: flex; flex-direction: column; justify-content: space-between; background-color: #212121; height: 100vh; }
            .chat-header { padding: 15px 25px; border-bottom: 1px solid #2f2f2f; font-size: 1rem; color: #9b9b9b; background-color: #212121; }
            
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; max-width: 800px; width: 100%; margin: 0 auto; }
            .message-row { display: flex; gap: 15px; width: 100%; }
            .message-row.user { justify-content: flex-end; }
            .message { padding: 12px 18px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem; max-width: 75%; word-wrap: break-word; }
            .user-message { background-color: #2f2f2f; color: #fff; border-top-right-radius: 2px; }
            .ai-message { background-color: transparent; color: #ececf1; border-top-left-radius: 2px; padding-left: 0; }

            /* Input Box Area */
            .input-area { padding: 20px; background-color: #212121; display: flex; justify-content: center; }
            .input-box-wrapper { max-width: 800px; width: 100%; background-color: #2f2f2f; border: 1px solid #424242; border-radius: 12px; padding: 12px 15px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
            textarea { width: 100%; background: transparent; border: none; color: #fff; font-size: 1rem; outline: none; resize: none; max-height: 150px; }
            .input-footer { display: flex; justify-content: space-between; align-items: center; }
            .model-badge { font-size: 0.8rem; color: #8e8ea0; background: #1f1f1f; padding: 4px 8px; border-radius: 4px; border: 1px solid #383838; }
            send-btn { background-color: #ececf1; color: #212121; border: none; border-radius: 6px; width: 32px; height: 32px; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: opacity 0.2s; }
            button:hover { opacity: 0.85; }
            .loading { color: #8e8ea0; font-style: italic; }
        </style>
    </head>
    <body>
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">✨ AI Workspace</div>
            <button class="new-chat-btn" onclick="location.reload()">+ নতুন চ্যাট শুরু করুন</button>
        </div>

        <!-- Main Workspace -->
        <div class="main-container">
            <div class="chat-header">Gemini Pro Assistant</div>
            
            <div id="chat-container">
                <div class="message-row">
                    <div class="message ai-message">হ্যালো আলবি! আজ আপনাকে কীভাবে সাহায্য করতে পারি?</div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-box-wrapper">
                    <textarea id="user-input" rows="1" placeholder="আপনার বার্তা এখানে লিখুন..." onkeypress="handleKeyPress(event)"></textarea>
                    <div class="input-footer">
                        <span class="model-badge">Gemini Flash</span>
                        <button style="background: #19c37d; color: white; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-weight: 600;" onclick="sendMessage()">পাঠান</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const inputField = document.getElementById('user-input');
                const chatContainer = document.getElementById('chat-container');
                const text = inputField.value.trim();
                if (!text) return;

                // User Message
                const userRow = document.createElement('div');
                userRow.className = 'message-row user';
                userRow.innerHTML = `<div class="message user-message">${text}</div>`;
                chatContainer.appendChild(userRow);

                inputField.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;

                // Loading message
                const aiRow = document.createElement('div');
                aiRow.className = 'message-row';
                const aiMsg = document.createElement('div');
                aiMsg.className = 'message ai-message loading';
                aiMsg.innerText = 'উত্তর তৈরি হচ্ছে...';
                aiRow.appendChild(aiMsg);
                chatContainer.appendChild(aiRow);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: text })
                    });
                    const data = await response.json();
                    
                    aiMsg.classList.remove('loading');
                    aiMsg.innerText = data.response || "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।";
                } catch (error) {
                    aiMsg.classList.remove('loading');
                    aiMsg.innerText = "সার্ভারে সংযোগ স্থাপন করতে সমস্যা হয়েছে।";
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
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))