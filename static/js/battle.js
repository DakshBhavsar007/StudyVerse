// BYTE BATTLE FRONTEND LOGIC

document.addEventListener('DOMContentLoaded', () => {
    // --- Socket Init ---
    const socket = io({ transports: ['websocket', 'polling'] });
    let currentRoom = null;
    let isHost = false;
    let battleTimer = null;

    // --- Elements ---
    const screens = {
        entry: document.getElementById('screen-entry'),
        battle: document.getElementById('screen-battle')
    };

    const modals = {
        joinReq: document.getElementById('modal-join-req'),
        result: document.getElementById('modal-result')
    };

    const display = {
        room: document.getElementById('room-display'),
        timer: document.getElementById('timer-display'),
        status: document.getElementById('status-display'),
        lang: document.getElementById('lang-display'),
        chat: document.getElementById('chat-log')
    };

    const problem = {
        title: document.getElementById('problem-title'),
        desc: document.getElementById('problem-desc'),
        details: document.getElementById('problem-details'),
        in: document.getElementById('prob-in'),
        out: document.getElementById('prob-out')
    };

    const inputs = {
        code: document.getElementById('code-editor'),
        chat: document.getElementById('chat-input')
    };

    const buttons = {
        create: document.getElementById('btn-create'),
        join: document.getElementById('btn-join'),
        submit: document.getElementById('btn-submit'),
        accept: document.getElementById('btn-accept'),
        reject: document.getElementById('btn-reject'),
        voteYes: document.getElementById('btn-vote-yes'),
        voteNo: document.getElementById('btn-vote-no')
    };

    // --- Helpers ---
    function showScreen(name) {
        Object.values(screens).forEach(el => el.classList.add('hidden'));
        if (screens[name]) {
            screens[name].classList.remove('hidden');
            screens[name].style.display = 'flex'; // Ensure flex layout applies
        }
    }

    function addChatMsg(sender, text, type = 'user') {
        const div = document.createElement('div');
        div.className = `chat-msg ${type}`;
        if (type === 'system') div.className += ' system';
        if (sender === 'ByteBot') div.className = 'chat-msg bot';
        if (sender === 'You') div.className = 'chat-msg user';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        // Handle bolding for system msgs (markdown-ish)
        if (type === 'system' || sender === 'ByteBot') {
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/\n/g, '<br>');
        }

        bubble.innerHTML = (sender && type !== 'system' && sender !== 'You' ? `<strong>${sender}:</strong> ` : '') + text;
        div.appendChild(bubble);
        display.chat.appendChild(div);
        display.chat.scrollTop = display.chat.scrollHeight;
    }

    function setStatus(text, active = false) {
        display.status.textContent = text;
        if (active) display.status.classList.add('active');
        else display.status.classList.remove('active');
    }

    // --- 1. Entry Logic ---

    buttons.create.addEventListener('click', () => {
        socket.emit('battle_create', {});
    });

    buttons.join.addEventListener('click', () => {
        const code = document.getElementById('join-code').value.trim();
        if (!code) return alert("Enter a room code!");

        // Change button to loading state
        buttons.join.textContent = "Requesting...";
        buttons.join.disabled = true;

        socket.emit('battle_join_request', { room_code: code });
        currentRoom = code;
    });

    socket.on('battle_created', (data) => {
        currentRoom = data.room_code;
        isHost = true;
        display.room.textContent = currentRoom;

        showScreen('battle');
        setStatus("WAITING FOR PLAYER");
        addChatMsg("ByteBot", "Room created. Waiting for opponent to join...");
        addChatMsg("ByteBot", `Invite Code: **${currentRoom}**`, 'system');
    });

    socket.on('battle_rejoined', (data) => {
        // Handle rejoin (simple version)
        currentRoom = data.room_code;
        showScreen('battle');
        display.room.textContent = currentRoom;
        addChatMsg("ByteBot", "Reconnected to battle.", 'system');
    });

    socket.on('battle_error', (data) => {
        alert(data.message);
        buttons.join.textContent = "Join";
        buttons.join.disabled = false;

        // Hide modal if open
        modals.joinReq.style.display = 'none';
    });

    // --- 2. Join Request Flow (Host Side) ---

    socket.on('battle_join_request_notify', (data) => {
        document.getElementById('req-name').textContent = data.player_name;
        modals.joinReq.style.display = 'flex';
    });

    buttons.accept.addEventListener('click', () => {
        socket.emit('battle_join_response', { room_code: currentRoom, accepted: true });
        modals.joinReq.style.display = 'none';
    });

    buttons.reject.addEventListener('click', () => {
        socket.emit('battle_join_response', { room_code: currentRoom, accepted: false });
        modals.joinReq.style.display = 'none';
    });

    // --- 3. Join Accepted (Guest Side) ---

    socket.on('join_accepted', (data) => {
        currentRoom = data.room_code;
        socket.emit('battle_confirm_join', { room_code: currentRoom });
    });

    // --- 4. Setup Phase ---

    socket.on('battle_entered', (data) => {
        if (!currentRoom) currentRoom = data.room_code; // Ensure set for guest
        display.room.textContent = currentRoom;
        showScreen('battle');
        setStatus("SETUP");
    });

    socket.on('battle_chat_message', (data) => {
        addChatMsg(data.sender, data.message, data.type);
    });

    // Chat Input
    inputs.chat.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const msg = inputs.chat.value.trim();
            if (msg) {
                // Optimistic UI
                addChatMsg("You", msg, 'user');
                socket.emit('battle_chat_send', { room_code: currentRoom, message: msg });
                inputs.chat.value = '';
            }
        }
    });

    // --- 5. Battle Phase ---

    socket.on('battle_started', (data) => {
        setStatus("BATTLE IN PROGRESS", true);

        // Update Language Display
        if (data.language) {
            display.lang.textContent = data.language;
        }

        // Update Problem
        problem.title.textContent = data.problem.title;
        problem.desc.textContent = data.problem.description;
        problem.in.textContent = data.problem.input_format;
        problem.out.textContent = data.problem.output_format;
        problem.details.classList.remove('hidden');

        // Reset Editor
        inputs.code.value = "";
        buttons.submit.disabled = false;
        buttons.submit.textContent = "SUBMIT CODE";

        // Start Timer
        startTimer(data.duration);
    });

    socket.on('battle_notification', (data) => {
        // Play sound
        const audio = document.getElementById('sound-bell');
        if (audio) audio.play().catch(e => { });

        // Flash border
        document.body.style.borderColor = 'var(--battle-accent)';
        setTimeout(() => document.body.style.borderColor = 'transparent', 500);
    });

    buttons.submit.addEventListener('click', () => {
        const code = inputs.code.value;
        if (!code.trim()) return alert("Write some code first!");

        if (confirm("Submit solution?")) {
            socket.emit('battle_submit', { room_code: currentRoom, code: code });
            buttons.submit.disabled = true;
            buttons.submit.textContent = "Submitted ✅";
        }
    });

    socket.on('battle_state_change', (data) => {
        if (data.state === 'judging') {
            setStatus("JUDGING...");
            stopTimer();
        }
    });

    // --- 6. Result & Rematch ---

    socket.on('battle_result', (data) => {
        modals.result.style.display = 'flex';
        document.getElementById('res-winner').textContent = `Winner: ${data.winner}`;
        document.getElementById('res-reason').textContent = data.reason;

        // Reset vote buttons
        buttons.voteYes.disabled = false;
        buttons.voteNo.disabled = false;
        document.getElementById('vote-status').textContent = "";
    });

    buttons.voteYes.addEventListener('click', () => {
        socket.emit('battle_rematch_vote', { room_code: currentRoom, vote: 'yes' });
        disableVotes("Waiting for opponent...");
    });

    buttons.voteNo.addEventListener('click', () => {
        socket.emit('battle_rematch_vote', { room_code: currentRoom, vote: 'no' });
        disableVotes("You declined. Notifying opponent...");
        // Don't close modal here - let backend's battle_rematch_declined event handle it
    });

    function disableVotes(msg) {
        buttons.voteYes.disabled = true;
        buttons.voteNo.disabled = true;
        document.getElementById('vote-status').textContent = msg;
    }

    socket.on('battle_restart', () => {
        modals.result.style.display = 'none';
        setStatus("SETUP");
        problem.title.textContent = "Waiting for host configuration...";
        problem.desc.textContent = "Host is selecting new difficulty/language via chat.";
        problem.details.classList.add('hidden');
        inputs.code.value = "";
    });

    socket.on('battle_rematch_declined', () => {
        // Close result modal and return both players to entry screen
        modals.result.style.display = 'none';
        showScreen('entry');

        // Reset room state
        currentRoom = null;
        isHost = false;

        // Clear chat
        display.chat.innerHTML = '';
    });

    // --- Timer Util ---
    function startTimer(duration) {
        let timeLeft = duration;
        stopTimer();
        updateTimerDisplay(timeLeft);

        battleTimer = setInterval(() => {
            timeLeft--;
            updateTimerDisplay(timeLeft);
            if (timeLeft <= 0) {
                stopTimer();
                // Auto submit or end
            }
        }, 1000);
    }

    function stopTimer() {
        if (battleTimer) clearInterval(battleTimer);
    }

    function updateTimerDisplay(seconds) {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        display.timer.textContent = `${m}:${s}`;

        if (seconds < 60) display.timer.style.color = '#ef4444'; // Red
        else display.timer.style.color = 'inherit';
    }
});