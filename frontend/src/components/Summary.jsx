import React, { useState } from 'react';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css'; // Import Quill styles
import htmlDocx from 'html-docx-js/dist/html-docx';
import ChatPopup from './ChatPopup'; // We'll create this component
import './Summary.css';

function Summary({ summary, audioUrl }) {
  const [editableSummary, setEditableSummary] = useState(summary);
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);

  const handleCopy = () => {
    navigator.clipboard.writeText(editableSummary);
  };

  const handleChatClick = () => {
    if (!showChat && chatMessages.length === 0) {
      setChatMessages([{
        role: 'system',
        content: `You are a helpful assistant that knows about this video summary: ${editableSummary}. Answer questions based on this context.`
      }]);
    }
    setShowChat(!showChat);
  };

  const handleDownloadPDF = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/download_pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ summary: editableSummary, filename: 'summary' }),
      });

      if (!response.ok) {
        throw new Error('Failed to download PDF');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'summary.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (error) {
      console.error('Error downloading PDF:', error);
    }
  };

  const exportToWord = () => {
    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
        </head>
        <body>
          ${editableSummary}
        </body>
      </html>
    `;

    const converted = htmlDocx.asBlob(html);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(converted);
    link.download = "video_summary.docx";
    link.click();
  };

  return (
    <div className="summary-container">
      {audioUrl && (
        <div className="audio-player-container">
          <h2>Hear the summary </h2>
          <audio controls className="audio-player" src={audioUrl}>
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
      <div className="summary-text">
        <div className="summary-header">
          <h2>
            Summary
            <div className="action-buttons">
              <button 
                onClick={handleCopy} 
                className="action-button" 
              >
                📋 <span>Copy Text</span>
              </button>
              <button 
                onClick={handleChatClick} 
                className="action-button" 
              >
                💬 <span>Chat With Video</span>
              </button>
              <button 
                onClick={exportToWord} 
                className="action-button" 
              >
                📝 <span>Export as .docx</span>
              </button>
            </div>
          </h2>
        </div>
        <ReactQuill
          value={editableSummary}
          onChange={setEditableSummary}
          modules={{
            toolbar: [
              [{ 'header': [1, 2, 3, false] }],
              ['bold', 'italic', 'underline', 'strike'],
              [{ 'list': 'ordered' }, { 'list': 'bullet' }],
              ['link', 'image'],
              ['clean']
            ],
          }}
          theme="snow"
        />
      </div>
      {showChat && (
        <ChatPopup 
          onClose={() => setShowChat(false)} 
          messages={chatMessages}
          setMessages={setChatMessages}
        />
      )}
    </div>
  );
}

export default Summary;