import React, { useState, useEffect, useRef } from 'react';

const ChatPopup = ({ onClose, messages, setMessages }) => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const apiKey = '84da3af36aa9a8e61cd8a40addb32bc56e71b35e8ae3a3008fc8bf82740f40ce';
  const popupRef = useRef(null);
  const messagesEndRef = useRef(null);

  const handleSend = async () => {
    if (!message.trim()) return;
    
    setIsLoading(true);
    try {
      // Add user message
      const updatedMessages = [...messages, { role: 'user', content: message }];
      setMessages(updatedMessages);
      
      const response = await fetch('https://api.together.xyz/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'meta-llama/Llama-3-70b-chat-hf',
          messages: updatedMessages,
          temperature: 0.7,
          max_tokens: 1000
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      if (data.choices && data.choices.length > 0) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.choices[0].message.content }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I couldn't process your request. Please try again." }]);
    } finally {
      setIsLoading(false);
      setMessage('');
    }
  };

  // Scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleCopyMessage = (text) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="chat-popup" ref={popupRef}>
      <div className="chat-header">
        <h3>Chat with Video</h3>
        <button onClick={onClose} className="close-button">×</button>
      </div>
      <div className="chat-messages">
        {messages.slice(1).map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
            {msg.role === 'assistant' && (
              <button 
                onClick={() => handleCopyMessage(msg.content)} 
                className="copy-message-button"
                title="Copy text"
              >
                📋
              </button>
            )}
          </div>
        ))}
        {isLoading && <div className="loading">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask about the video..."
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading || !message}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatPopup; 