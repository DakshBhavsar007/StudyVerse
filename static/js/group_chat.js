document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // UX Logic (Auto-resize, Enter key)
    // ----------------------------------------------------
    const chatInput = document.getElementById('groupChatInput');
    const sendButton = document.getElementById('groupSendButton');
    const messagesContainer = document.getElementById('groupMessagesContainer');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const chatForm = document.getElementById('groupChatForm');

    if (chatInput) {
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
        });

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                // Instead of clicking button, dispatch submit event to handle via socket
                if (chatForm) chatForm.dispatchEvent(new Event('submit'));
            }
        });
    }

    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', () => {
            fileInput.click();
        });

        const previewContainer = document.getElementById('filePreviewContainer');
        const previewName = document.getElementById('previewFileName');
        const removeFileBtn = document.getElementById('removeFileBtn');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                uploadBtn.classList.add('text-primary');

                // Show preview
                if (previewContainer && previewName) {
                    previewContainer.style.display = 'flex';
                    previewName.textContent = file.name;

                    // Optional: Image preview if needed
                    // const reader = new FileReader(); ...
                }
            } else {
                clearFileSelection();
            }
        });

        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', () => {
                clearFileSelection();
            });
        }

        function clearFileSelection() {
            fileInput.value = '';
            uploadBtn.classList.remove('text-primary');
            if (previewContainer) previewContainer.style.display = 'none';
        }

        // Expose to submit handler scope if needed, or attach to form element
        chatForm.clearFileSelection = clearFileSelection;
    }

    // ----------------------------------------------------
    // Socket.IO Logic
    // ----------------------------------------------------
    if (typeof io !== 'undefined' && GROUP_ID) {
        const socket = io();

        socket.on('connect', () => {
            console.log('Connected to SocketIO server');
            socket.emit('join', { group_id: GROUP_ID });
        });

        socket.on('receive_message', (data) => {
            appendMessage(data);
        });

        if (chatForm) {
            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();

                const message = chatInput.value.trim();
                const file = fileInput.files[0];

                if (!message && !file) return;

                let filePath = null;

                // Handle File Upload first if exists
                if (file) {
                    const formData = new FormData();
                    formData.append('file', file);

                    try {
                        sendButton.disabled = true;
                        const response = await fetch('/group/upload', {
                            method: 'POST',
                            body: formData
                        });

                        if (response.ok) {
                            const result = await response.json();
                            filePath = result.url;
                        } else {
                            console.error('File upload failed');
                            alert('Failed to upload file');
                            sendButton.disabled = false;
                            return;
                        }
                    } catch (err) {
                        console.error('Error uploading file:', err);
                        sendButton.disabled = false;
                        return;
                    }
                }

                // Emit message via SocketIO
                socket.emit('send_message', {
                    group_id: GROUP_ID,
                    content: message,
                    file_path: filePath
                });

                // Reset UI
                chatInput.value = '';
                chatInput.style.height = 'auto';
                if (chatForm.clearFileSelection) chatForm.clearFileSelection();
                sendButton.disabled = false;
            });
        }
    }

    function appendMessage(data) {
        if (!messagesContainer) return;

        const isMe = String(data.user_id) === String(CURRENT_USER_ID);
        const isAI = data.role === 'assistant';

        // Create message container with same styling as template
        const msgDiv = document.createElement('div');
        msgDiv.style.display = 'flex';
        msgDiv.style.gap = '10px';
        msgDiv.style.alignSelf = isMe ? 'flex-end' : 'flex-start';
        msgDiv.style.maxWidth = '70%';
        if (isMe) {
            msgDiv.style.flexDirection = 'row-reverse';
        }

        // Avatar HTML
        let avatarHtml = '';
        if (isAI) {
            avatarHtml = `
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: grid; place-items: center; color: white; font-size: 0.8rem;">
                    <i class="fa-solid fa-robot"></i>
                </div>
            `;
        } else {
            const avatarUrl = data.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(data.username || 'User')}&background=random`;
            avatarHtml = `<img src="${avatarUrl}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover;">`;
        }

        // Name label
        const nameLabel = isAI ? 'AI Coach' : (isMe ? 'You' : data.username);

        // Message bubble styling
        const bubbleBg = isMe ? 'var(--accent-green)' : (isAI ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.1)');
        const bubbleColor = isMe ? 'black' : 'white';
        const borderRadius = isMe ? '12px 0 12px 12px' : '0 12px 12px 12px';
        const borderColor = isAI ? 'rgba(59, 130, 246, 0.3)' : 'transparent';

        // Attachment HTML
        let attachmentHtml = '';
        if (data.file_path) {
            attachmentHtml = `
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(0,0,0,0.1);">
                    <a href="${data.file_path}" target="_blank" style="text-decoration: none; display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: inherit; font-weight: 600;">
                        <i class="fa-solid fa-paperclip"></i> Attachment
                    </a>
                </div>
            `;
        }

        // Build the complete message structure matching the template
        msgDiv.innerHTML = `
            ${avatarHtml}
            <div style="display: flex; flex-direction: column; gap: 4px; align-items: ${isMe ? 'flex-end' : 'flex-start'};">
                <div style="font-size: 0.75rem; color: var(--text-secondary); margin: 0 4px;">
                    ${nameLabel} • <span>${data.created_at}</span>
                </div>
                <div style="background: ${bubbleBg}; color: ${bubbleColor}; padding: 12px; border-radius: ${borderRadius}; font-weight: 500; font-size: 0.95rem; border: 1px solid ${borderColor};">
                    ${escapeHtml(data.content)}
                    ${attachmentHtml}
                </div>
            </div>
        `;

        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});