import { useState } from 'react';
import axios from 'axios'; // 1. 匯入 axios
import './App.css';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([
    { id: 1, text: '你好！請提出一個關於演算法的問題，我將嘗試從文件中為您尋找答案。', sender: 'ai' },
  ]);
  // 新增一個狀態來處理「載入中」
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => { // 2. 將函式改為 async (非同步)
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    const question = inputValue;
    setInputValue('');
    setIsLoading(true); // 3. 開始載入

    // 4. 使用 try...catch 來捕捉 API 請求的潛在錯誤
    try {
      // 發送 POST 請求到您的後端 API
      // **注意：** 這裡我們先假設後端會有一個 /ask 的端點
      const response = await axios.post('http://127.0.0.1:8000/ask', {
        question: question // 將問題作為請求內容發送
      });

      // 從後端回應中取得 AI 的回答
      const aiResponse = {
        id: Date.now() + 1,
        text: response.data.answer, // 假設後端回傳的 JSON 中有 answer 欄位
        sender: 'ai',
      };
      setMessages(prevMessages => [...prevMessages, aiResponse]);

    } catch (error) {
      console.error("呼叫 API 時發生錯誤:", error);
      // 如果出錯，也顯示一則錯誤訊息
      const errorResponse = {
        id: Date.now() + 1,
        text: '抱歉，與伺服器連線時發生錯誤，請檢查後端服務是否正常運行。',
        sender: 'ai',
      };
      setMessages(prevMessages => [...prevMessages, errorResponse]);
    } finally {
      setIsLoading(false); // 5. 結束載入
    }
  };

  return (
    <div className="chat-container">
      <div className="message-list">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        {/* 當正在載入時，顯示 "思考中..." 的提示 */}
        {isLoading && (
          <div className="message ai">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => { if (e.key === 'Enter') { handleSendMessage(); } }}
          placeholder={isLoading ? "AI 正在思考中..." : "請在這裡輸入你的問題..."}
          disabled={isLoading} // 正在載入時禁用輸入框
        />
        <button onClick={handleSendMessage} disabled={isLoading}>
          {isLoading ? "思考中" : "發送"}
        </button>
      </div>
    </div>
  );
}

export default App;