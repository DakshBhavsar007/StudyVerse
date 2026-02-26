/**
 * StudyVerse – Byte Battle (Unified JS)
 * Fixed: battleActive flag prevents enterWaiting() from overwriting battle screen
 */
document.addEventListener('DOMContentLoaded', () => {

    const socket = io();
    window.socket = socket;

    let currentRoom  = sessionStorage.getItem('battle_room_code') || null;
    let isHost       = false;
    let selMode      = '1v1';
    let selVis       = 'public';
    let battleTimer  = null;
    let heartbeat    = null;
    let pendInvite   = null;
    let battleActive = false;  // KEY FIX: true after battle_started — blocks all enterWaiting() calls

    const $  = id => document.getElementById(id);
    const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };

    function showScreen(name) {
        ['screen-entry','screen-waiting','screen-battle'].forEach(s => {
            const el = $(s); if (!el) return;
            if (s === name) { el.classList.remove('hidden'); el.style.display = 'flex'; }
            else            { el.classList.add('hidden');    el.style.display = 'none'; }
        });
        dbg('Screen → ' + name);
    }

    const pill = document.createElement('div');
    pill.style.cssText = 'position:fixed;bottom:10px;right:10px;background:rgba(0,0,0,.85);color:lime;padding:3px 9px;font-size:10px;border-radius:4px;z-index:9999;pointer-events:none;';
    pill.innerHTML = 'Status: <span id="_dbg">Init</span>';
    document.body.appendChild(pill);
    function dbg(msg) { const el = $('_dbg'); if (el) el.textContent = msg; console.log('[Battle]', msg); }

    function toast(msg, type) {
        const c = {info:'#3b82f6',success:'#4ade80',error:'#ef4444',warn:'#f59e0b'};
        const t = document.createElement('div');
        t.style.cssText = `position:fixed;top:80px;right:20px;background:${c[type]||c.info};color:#fff;padding:12px 20px;border-radius:8px;z-index:9999;font-size:.9rem;box-shadow:0 4px 12px rgba(0,0,0,.4);max-width:320px;word-break:break-word;`;
        t.textContent = msg; document.body.appendChild(t); setTimeout(() => t.remove(), 4000);
    }

    function chatMsg(sender, text, type) {
        const log = $('chat-log'); if (!log) return;
        const wrap = document.createElement('div');
        wrap.className = 'chat-msg ' + (sender === 'ByteBot' ? 'bot' : type === 'system' ? 'system' : (type||'user'));
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        const safe = (text||'').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
        bubble.innerHTML = (sender && type !== 'system' ? `<strong>${sender}:</strong> ` : '') + safe;
        wrap.appendChild(bubble); log.appendChild(wrap); log.scrollTop = log.scrollHeight;
    }

    function setStatus(t) { const el = $('status-display'); if (el) el.textContent = t; }

    document.querySelectorAll('.mode-btn').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(x => x.classList.remove('active'));
        b.classList.add('active'); selMode = b.dataset.mode;
    }));
    document.querySelectorAll('.vis-btn').forEach(b => b.addEventListener('click', () => {
        document.querySelectorAll('.vis-btn').forEach(x => x.classList.remove('active'));
        b.classList.add('active'); selVis = b.dataset.vis;
    }));

    ['btn-copy-room','btn-copy-code'].forEach(id => on(id,'click', () => {
        const code = currentRoom || ($('room-display') && $('room-display').textContent);
        if (code && code !== '---') navigator.clipboard.writeText(code).then(() => toast('Copied!','success'));
    }));

    /* ── CONNECT ── */
    socket.on('connect', () => {
        dbg('Connected');
        socket.emit('join_personal_room', {});
        if (currentRoom && !battleActive) {
            dbg('Rejoin ' + currentRoom);
            socket.emit('battle_rejoin_attempt', { room_code: currentRoom });
        }
        startHB();
    });
    socket.on('connect_error', () => { dbg('Conn Error'); stopHB(); });
    socket.on('disconnect',    () => { dbg('Disconnected'); stopHB(); });

    function startHB() { stopHB(); heartbeat = setInterval(() => { if (currentRoom) socket.emit('battle_heartbeat', { room_code: currentRoom }); }, 30000); }
    function stopHB()  { if (heartbeat) { clearInterval(heartbeat); heartbeat = null; } }

    /* ── WAITING ROOM HELPERS ── */
    function enterWaiting(data) {
        if (battleActive) { dbg('enterWaiting blocked — battle is active'); return; }
        currentRoom = data.room_code;
        const rc = $('wr-room-code'); if (rc) rc.textContent = data.room_code;
        const ml = $('wr-mode-label');
        if (ml) ml.textContent = `Mode: ${data.mode||'1v1'} · ${data.visibility==='private'?'🔒 Private':'🌐 Public'} · ${data.slots_filled||1}/${data.slots_total||2} players`;
        buildSlots(data.slots_filled||1, data.slots_total||2);
        showScreen('screen-waiting');
    }

    function buildSlots(filled, total) {
        const c = $('wr-slots'); if (!c) return; c.innerHTML = '';
        for (let i = 0; i < total; i++) {
            const s = document.createElement('div');
            s.style.cssText = `width:40px;height:40px;border-radius:50%;border:2px solid ${i<filled?'#4ade80':'#444'};background:${i<filled?'rgba(74,222,128,.15)':'transparent'};display:flex;align-items:center;justify-content:center;font-size:1.1rem;`;
            s.textContent = i < filled ? '✅' : '⌛'; c.appendChild(s);
        }
    }

    function setTeams(players) {
        const aEl = $('team-a-list'), bEl = $('team-b-list');
        if (!players) return;
        const a = Object.values(players).filter(p=>p.team==='A').map(p=>p.name);
        const b = Object.values(players).filter(p=>p.team==='B').map(p=>p.name);
        if (aEl) aEl.innerHTML = a.length ? a.map(n=>`<div>• ${n}</div>`).join('') : '—';
        if (bEl) bEl.innerHTML = b.length ? b.map(n=>`<div>• ${n}</div>`).join('') : '—';
    }

    function showTeamBadge(myTeam) {
        if (!myTeam || battleActive) return;
        const wr = $('wr-slots'); if (!wr) return;
        const existing = wr.parentNode.querySelector('.team-badge');
        if (existing) existing.remove();
        const badge = document.createElement('div');
        badge.className = 'team-badge';
        badge.style.cssText = 'margin-top:0.75rem;padding:5px 14px;border-radius:12px;font-size:0.85rem;font-weight:700;display:inline-block;';
        badge.style.background = myTeam === 'A' ? 'rgba(96,165,250,0.2)' : 'rgba(248,113,113,0.2)';
        badge.style.color = myTeam === 'A' ? '#60a5fa' : '#f87171';
        badge.style.border = `1px solid ${myTeam === 'A' ? '#60a5fa' : '#f87171'}`;
        badge.textContent = `You are on Team ${myTeam}`;
        wr.parentNode.insertBefore(badge, wr.nextSibling);
    }

    /* ── CREATE ROOM ── */
    on('btn-create','click', () => {
        const btn = $('btn-create');
        btn.textContent = 'Creating…'; btn.disabled = true;
        dbg('Create ' + selMode + ' / ' + selVis);
        socket.emit('battle_create', { mode: selMode, visibility: selVis });
        setTimeout(() => { if (btn.disabled) { btn.textContent='Create Room'; btn.disabled=false; toast('Timeout – try again','error'); } }, 10000);
    });

    socket.on('battle_created', data => {
        dbg('Room created: ' + data.room_code);
        currentRoom = data.room_code;
        battleActive = false;
        sessionStorage.setItem('battle_room_code', currentRoom);
        isHost = true;
        const btn = $('btn-create'); if (btn) { btn.textContent='Create Room'; btn.disabled=false; }
        enterWaiting(data);
        showTeamBadge('A');
        socket.emit('battle_entered', { room_code: currentRoom });
    });

    /* ── AUTO MATCHMAKING ── */
    on('btn-auto-mm','click', () => {
        const btn = $('btn-auto-mm'); if (btn) btn.disabled = true;
        const ms = $('mm-status'); if (ms) ms.style.display = 'block';
        const mt = $('mm-status-text'); if (mt) mt.textContent = `Searching for ${selMode} opponents…`;
        socket.emit('battle_queue_join', { mode: selMode });
    });

    on('btn-cancel-mm','click', () => {
        socket.emit('battle_queue_leave', { mode: selMode });
        const btn = $('btn-auto-mm'); if (btn) btn.disabled = false;
        const ms = $('mm-status'); if (ms) ms.style.display = 'none';
    });

    socket.on('battle_queue_status', data => {
        if (data.status === 'searching') {
            const mt = $('mm-status-text'); if (mt) mt.textContent = `In queue… (${data.queue_size} waiting for ${data.mode})`;
        } else {
            const btn = $('btn-auto-mm'); if (btn) btn.disabled = false;
            const ms = $('mm-status'); if (ms) ms.style.display = 'none';
        }
    });

    socket.on('battle_match_found', data => {
        if (battleActive) return;
        const btn = $('btn-auto-mm'); if (btn) btn.disabled = false;
        const ms  = $('mm-status');  if (ms) ms.style.display = 'none';
        currentRoom = data.room_code;
        sessionStorage.setItem('battle_room_code', currentRoom);
        const total = data.players ? data.players.length : 2;
        enterWaiting({ room_code: data.room_code, mode: data.mode, visibility:'public', slots_total: total, slots_filled: total });
        socket.emit('battle_entered', { room_code: currentRoom });
    });

    /* ── MANUAL JOIN ── */
    on('btn-join','click', () => {
        const ci = $('join-code'); if (!ci || !ci.value.trim()) { toast('Enter a room code!','warn'); return; }
        const code = ci.value.trim().toUpperCase();
        const btn  = $('btn-join'); if (btn) { btn.textContent='Requesting…'; btn.disabled=true; }
        currentRoom = code;
        socket.emit('battle_join_request', { room_code: code });
    });

    socket.on('battle_waiting_approval', data => {
        toast('⏳ ' + (data.message || 'Waiting for host…'), 'info');
        const btn = $('btn-join'); if (btn) { btn.textContent='Join'; btn.disabled=false; }
    });

    socket.on('battle_player_joined', data => {
        if (battleActive) return;
        buildSlots(data.slots_filled, data.slots_total);
        setTeams(data.players||{});
        const ml = $('wr-mode-label');
        if (ml) ml.textContent = ml.textContent.replace(/\d+\/\d+/, `${data.slots_filled}/${data.slots_total}`);
        toast(`👋 ${data.name} joined!`, 'success');
        dbg(`Players: ${data.slots_filled}/${data.slots_total}`);
    });

    socket.on('battle_enter_room', data => {
        if (battleActive) { dbg('battle_enter_room ignored — battle active'); return; }
        currentRoom = data.room_code;
        sessionStorage.setItem('battle_room_code', currentRoom);
        const btn = $('btn-join'); if (btn) { btn.textContent='Join'; btn.disabled=false; }
        enterWaiting({ room_code:data.room_code, mode:data.mode, visibility:data.visibility||'public', slots_total:data.slots_total, slots_filled:data.slots_filled });
        setTeams(data.players||{});
        showTeamBadge(data.your_team);
        socket.emit('battle_entered', { room_code: data.room_code });
    });

    socket.on('battle_team_assigned', data => {
        if (battleActive) return;
        if (data.players) setTeams(data.players);
        const ml = $('wr-mode-label');
        if (ml) ml.textContent = `Mode: ${data.mode||'1v1'} · ${(data.slots_filled||1)}/${data.slots_total||2} players`;
        buildSlots(data.slots_filled||1, data.slots_total||2);
        showTeamBadge(data.team);
    });

    socket.on('battle_rejoined', data => {
        currentRoom = data.room_code;
        sessionStorage.setItem('battle_room_code', currentRoom);
        if (data.state && data.state !== 'waiting') {
            battleActive = true;
            showScreen('screen-battle');
            const rd = $('room-display'); if (rd) rd.textContent = currentRoom;
            setStatus('BATTLE IN PROGRESS');
            chatMsg('ByteBot','Reconnected.','system');
        } else {
            battleActive = false;
            enterWaiting({ room_code:data.room_code, mode:data.mode||'1v1', visibility:'public', slots_total:data.slots_total||2, slots_filled:data.slots_filled||1 });
            setTeams(data.players||{});
            showTeamBadge(data.your_team);
        }
    });

    on('btn-leave-waiting','click', () => {
        if (currentRoom) socket.emit('battle_leave', { room_code: currentRoom });
        currentRoom = null; battleActive = false; sessionStorage.removeItem('battle_room_code');
        showScreen('screen-entry');
    });

    /* ── INVITE FRIENDS ── */
    on('btn-invite-friends','click', () => { const m=$('modal-invite'); if(m) m.style.display='flex'; });

    document.querySelectorAll('.btn-invite-friend').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!currentRoom) { toast('Create a room first!','warn'); return; }
            socket.emit('battle_invite_friend', { room_code: currentRoom, friend_id: parseInt(btn.dataset.fid) });
            btn.textContent = 'Invited ✅'; btn.disabled = true;
        });
    });

    socket.on('battle_invite_sent', () => toast('📨 Invite sent!','success'));

    socket.on('battle_invite_received', data => {
        pendInvite = data;
        const t=$('invite-title'), m=$('invite-msg');
        if (t) t.textContent = '🎮 Battle Invite!';
        if (m) m.textContent = `${data.host_name} invited you to a ${data.mode} battle. Room: ${data.room_code}`;
        const modal=$('modal-invite-received'); if(modal) modal.style.display='flex';
    });

    on('btn-accept-invite','click', () => {
        const m=$('modal-invite-received'); if(m) m.style.display='none';
        if (pendInvite) { currentRoom=pendInvite.room_code; socket.emit('battle_join_request',{room_code:pendInvite.room_code}); pendInvite=null; }
    });
    on('btn-decline-invite','click', () => { const m=$('modal-invite-received'); if(m) m.style.display='none'; pendInvite=null; });

    /* ── HOST JOIN REQUEST MODAL ── */
    socket.on('battle_join_request_notify', data => {
        const n=$('req-name'); if(n) n.textContent = data.name || data.player_name || 'Someone';
        const m=$('modal-join-req'); if(m) m.style.display='flex';
    });

    on('btn-accept','click', () => { const m=$('modal-join-req'); if(m) m.style.display='none'; socket.emit('battle_join_response', { room_code: currentRoom, accepted: true }); });
    on('btn-reject','click', () => { const m=$('modal-join-req'); if(m) m.style.display='none'; socket.emit('battle_join_response', { room_code: currentRoom, accepted: false }); });

    /* ════════════════════════════════════
       BATTLE STARTED — THE CRITICAL FIX
       Set battleActive=true FIRST so no
       subsequent events can call enterWaiting
    ════════════════════════════════════ */
    socket.on('battle_started', data => {
        battleActive = true;  // LOCK immediately — blocks all enterWaiting() calls
        dbg('Battle started! → screen-battle');

        const mode = data.mode || '1v1';
        const badge=$('mode-badge'), rd=$('room-display'), ld=$('lang-display'), sb=$('team-scorebar');
        if(badge) badge.textContent = mode;
        if(rd) rd.textContent = currentRoom;
        if(data.config && ld) ld.textContent = data.config.language || 'Python';
        setStatus('BATTLE IN PROGRESS');

        if(mode !== '1v1' && sb) {
            sb.style.display = 'flex';
            if(data.teams) chatMsg('ByteBot',`⚔️ Teams – 🔵 A: ${(data.teams.A||[]).join(', ')} | 🔴 B: ${(data.teams.B||[]).join(', ')}`,'system');
        }

        if(data.problem) {
            const pt=$('problem-title'),pd=$('problem-desc'),pp=$('problem-details'),pi=$('prob-in'),po=$('prob-out');
            if(pt) pt.textContent = data.problem.title || 'Problem';
            if(pd) pd.textContent = data.problem.description || '';
            if(data.problem.examples && data.problem.examples.length && pi && po) {
                pi.textContent = data.problem.examples[0].input  || '';
                po.textContent = data.problem.examples[0].output || '';
                if(pp) pp.classList.remove('hidden');
            }
        }

        const ce=$('code-editor'),sb2=$('btn-submit');
        if(ce) ce.value='';
        if(sb2) { sb2.disabled=false; sb2.textContent='SUBMIT CODE'; }
        if(data.duration) startTimer(data.duration);

        showScreen('screen-battle');  // switch screen LAST
    });

    socket.on('battle_chat_message', data => chatMsg(data.sender, data.message, data.type));

    on('chat-input','keypress', e => {
        if(e.key !== 'Enter') return;
        const ci=$('chat-input'), msg=ci&&ci.value.trim();
        if(msg && currentRoom) { chatMsg('You', msg, 'user'); socket.emit('battle_chat_send', { room_code: currentRoom, message: msg }); ci.value = ''; }
    });

    on('btn-submit','click', () => {
        const code = $('code-editor') && $('code-editor').value;
        if(!code || !code.trim()) { toast('Write some code first!','warn'); return; }
        if(confirm('Submit solution?')) {
            socket.emit('battle_submit', { room_code: currentRoom, code });
            const sb=$('btn-submit'); if(sb){sb.disabled=true; sb.textContent='Submitted ✅';}
        }
    });

    socket.on('battle_team_scores', data => {
        const a=$('score-team-a'),b=$('score-team-b');
        if(a) a.textContent=data.A||0; if(b) b.textContent=data.B||0;
    });

    socket.on('battle_state_change', data => {
        if(data.state==='judging') { setStatus('JUDGING…'); stopTimer(); const sd=$('status-display'); if(sd) sd.classList.add('judging'); }
    });

    socket.on('battle_notification', () => { const a=$('sound-bell'); if(a) a.play().catch(()=>{}); });

    /* ── RESULT MODAL ── */
    socket.on('battle_result', data => {
        const sd=$('status-display'); if(sd){ sd.textContent='RESULT'; sd.classList.remove('judging'); }
        stopTimer();
        const wEl=$('res-winner'),rEl=$('res-reason'),xEl=$('res-xp'),tsEl=$('res-team-scores'),saEl=$('res-score-a'),sbEl=$('res-score-b');
        if(wEl) wEl.textContent = data.winning_team ? `🏆 Team ${data.winning_team} wins!` : `🏆 ${data.winner||'Draw'}`;
        if(rEl) rEl.textContent = data.summary || (data.feedback ? JSON.stringify(data.feedback,null,2) : '');
        if(xEl && data.xp_awarded) xEl.textContent = Object.entries(data.xp_awarded).map(([n,v])=>`${n}: +${v} XP`).join(' · ');
        if(data.team_scores && tsEl && saEl && sbEl){ tsEl.style.display='flex'; saEl.textContent=data.team_scores.A||0; sbEl.textContent=data.team_scores.B||0; }
        const vy=$('btn-vote-yes'),vn=$('btn-vote-no'),vs=$('vote-status');
        if(vy) vy.disabled=false; if(vn) vn.disabled=false; if(vs) vs.textContent='';
        const mr=$('modal-result'); if(mr) mr.style.display='flex';
    });

    on('btn-vote-yes','click', () => {
        socket.emit('battle_rematch_vote',{room_code:currentRoom,vote:'yes'});
        const vy=$('btn-vote-yes'); if(vy) vy.disabled=true;
        const vs=$('vote-status'); if(vs) vs.textContent='Voted YES – waiting for others…';
    });
    on('btn-vote-no','click', () => { socket.emit('battle_rematch_vote',{room_code:currentRoom,vote:'no'}); const vn=$('btn-vote-no'); if(vn) vn.disabled=true; });

    socket.on('battle_rematch_declined', () => {
        battleActive = false;
        const mr=$('modal-result'); if(mr) mr.style.display='none';
        sessionStorage.removeItem('battle_room_code'); currentRoom=null; isHost=false;
        const cl=$('chat-log'); if(cl) cl.innerHTML='';
        showScreen('screen-entry');
    });

    socket.on('battle_restart', () => {
        battleActive = false;
        const mr=$('modal-result'); if(mr) mr.style.display='none';
        const ce=$('code-editor'),pt=$('problem-title'),pd=$('problem-desc'),pp=$('problem-details');
        if(ce) ce.value=''; if(pt) pt.textContent='Waiting for host…'; if(pd) pd.textContent='Not started.'; if(pp) pp.classList.add('hidden');
        setStatus('WAITING'); showScreen('screen-waiting');
    });

    /* ── ERRORS / ROOM CLOSED ── */
    socket.on('battle_error', data => {
        const msg = data.message || 'Error.';
        toast('❌ '+msg,'error'); dbg('Err: '+msg);
        if(msg.includes('Invalid')||msg.includes('expired')||msg.includes('not in this room')) {
            sessionStorage.removeItem('battle_room_code'); currentRoom=null; battleActive=false; showScreen('screen-entry');
        }
        const bj=$('btn-join'); if(bj&&bj.disabled){bj.textContent='Join';bj.disabled=false;}
    });

    socket.on('battle_room_closed', data => {
        sessionStorage.removeItem('battle_room_code'); currentRoom=null; isHost=false; battleActive=false; stopTimer();
        ['modal-result','modal-join-req'].forEach(id=>{const el=$(id);if(el)el.style.display='none';});
        const cl=$('chat-log'); if(cl) cl.innerHTML='';
        showScreen('screen-entry');
        toast('🚫 '+(data.reason||'Room closed.'),'error');
    });

    socket.on('battle_room_expired',()=>{ sessionStorage.removeItem('battle_room_code'); currentRoom=null; battleActive=false; showScreen('screen-entry'); });

    /* ── TIMER ── */
    function startTimer(dur) {
        let t = dur||1800; stopTimer(); tickTimer(t);
        battleTimer = setInterval(()=>{ t--; tickTimer(t); if(t<=0) stopTimer(); }, 1000);
    }
    function stopTimer() { if(battleTimer){clearInterval(battleTimer);battleTimer=null;} }
    function tickTimer(s) {
        const el=$('timer-display'); if(!el) return;
        const m=Math.floor(s/60).toString().padStart(2,'0'), sec=(s%60).toString().padStart(2,'0');
        el.textContent=`${m}:${sec}`; el.style.color=s<60?'#ef4444':'inherit';
    }

    window.leaveRoom = function() {
        if(!confirm('Leave? This closes the room for everyone.')) return;
        if(currentRoom) socket.emit('battle_leave',{room_code:currentRoom});
        sessionStorage.removeItem('battle_room_code'); currentRoom=null; battleActive=false;
        window.location.href='/battle';
    };

    showScreen('screen-entry');
});