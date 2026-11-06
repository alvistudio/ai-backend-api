import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# Render-এর Environment Variable থেকে এপিআই কি রিড করা
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class ChatRequest(BaseModel):
    prompt: str

# ব্রাউজারের জন্য প্রিমিয়াম চ্যাট ইন্টারফেস (HTML, CSS & JS)
@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Assistant - Gemini Powered</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            body { background-color: #0D1117; color: #C9D1D9; display: flex; flex-direction: column; height: 100vh; justify-content: space-between; }
            header { background-color: #161B22; padding: 15px 20px; text-align: center; border-bottom: 1px solid #30363D; font-size: 1.2rem; font-weight: bold; color: #58A6FF; }
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; word-wrap: break-word; font-size: 0.95rem; }
            .user-message { background-color: #238636; color: #ffffff; align-self: flex-end; border-bottom-right-radius: 2px; }
            .ai-message { background-color: #21262D; color: #C9D1D9; align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #30363D; }
            .input-container { background-color: #161B22; padding: 15px 20px; display: flex; gap: 10px; border-top: 1px solid #30363D; }
            input { flex: 1; background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; padding: 12px 15px; color: #C9D1D9; font-size: 1rem; outline: none; }
            input:focus { border-color: #58A6FF; }
            button { background-color: #238636; color: white; border: none; border-radius: 8px; padding: 0 20px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
            button:hover { background-color: #2ea043; }
            .loading { font-style: italic; color: #8B949E; }
        </style>
    </head>
    <body>
        <header>✨ AI Assistant (Gemini)</header>
        <div id="chat-container">
            <div class="message ai-message">হ্যালো! আমি আপনার এআই অ্যাসিস্ট্যান্ট। আজ আপনাকে কীভাবে সাহায্য করতে পারি?</div>
        </div>
        <div class="input-container">
            <input type="text" id="user-input" placeholder="আপনার বার্তা এখানে লিখুন..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">পাঠান</button>
        </div>

        <script>
            async function sendMessage() {
                const inputField = document.getElementById('user-input');
                const chatContainer = document.getElementById('chat-container');
                const text = inputField.value.trim();
                if (!text) return;

                // ইউজারের মেসেজ স্ক্রিনে যোগ করা
                const userMsgDiv = document.createElement('div');
                userMsgDiv.className = 'message user-message';
                userMsgDiv.innerText = text;
                chatContainer.appendChild(userMsgDiv);

                inputField.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;

                // লোডিং মেসেজ তৈরি করা
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message ai-message loading';
                loadingDiv.innerText = 'উত্তর তৈরি হচ্ছে...';
                chatContainer.appendChild(loadingDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: text })
                    });
                    const data = await response.json();
                    
                    chatContainer.removeChild(loadingDiv);

                    const aiMsgDiv = document.createElement('div');
                    aiMsgDiv.className = 'message ai-message';
                    aiMsgDiv.innerText = data.response || "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।";
                    chatContainer.appendChild(aiMsgDiv);
                } catch (error) {
                    chatContainer.removeChild(loadingDiv);
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message ai-message';
                    errorDiv.innerText = "সার্ভারে সংযোগ স্থাপন করতে সমস্যা হচ্ছে।";
                    chatContainer.appendChild(errorDiv);
                }
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """

# চ্যাট এন্ডপয়েন্ট (যা জেমিনি থেকে উত্তর এনে দেবে)
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