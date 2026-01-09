// Pomodoro Timer JavaScript

let timerInterval = null;
let timeLeft = 25 * 60; // seconds (used for countdown)
let elapsedSeconds = 0; // seconds (used for stopwatch)
let isRunning = false;
let currentMode = 'focus'; // 'focus', 'shortBreak', 'stopwatch'

const timerConfig = {
    focus: { duration: 25 * 60, label: 'Focus', color: 'var(--accent-green)' },
    shortBreak: { duration: 5 * 60, label: 'Short Break', color: 'var(--accent-green)' },
    stopwatch: { duration: 0, label: 'Stopwatch', color: 'var(--accent-green)' }
};

document.addEventListener('DOMContentLoaded', () => {
    // New IDs from inspiration template
    const timerDisplay = document.getElementById('timer-display');
    const startBtn = document.getElementById('start-btn');
    const resetBtn = document.getElementById('reset-btn');

    // Mode buttons
    const focusMode = document.getElementById('focusMode');
    const shortBreakMode = document.getElementById('shortBreakMode');
    const stopwatchMode = document.getElementById('stopwatchMode');

    // --- Persistence Logic ---

    function saveState() {
        const state = {
            mode: currentMode,
            isRunning: isRunning,
            timeLeft: timeLeft,
            elapsedSeconds: elapsedSeconds,
            lastUpdated: Date.now()
        };

        // If running, we store absolute timestamps to handle time passing while away
        if (isRunning) {
            if (currentMode === 'stopwatch') {
                state.startTime = Date.now() - (elapsedSeconds * 1000);
            } else {
                state.targetTime = Date.now() + (timeLeft * 1000);
            }
        }

        localStorage.setItem('pomodoroState', JSON.stringify(state));
    }

    function loadState() {
        const saved = localStorage.getItem('pomodoroState');
        if (!saved) return;

        try {
            const state = JSON.parse(saved);

            // Validate mode
            if (state.mode && timerConfig[state.mode]) {
                currentMode = state.mode;
            } else {
                currentMode = 'focus';
            }

            isRunning = state.isRunning;

            // Restore visual mode selection
            updateModeButtons(currentMode);

            if (isRunning) {
                const now = Date.now();
                if (currentMode === 'stopwatch') {
                    if (state.startTime) {
                        elapsedSeconds = Math.floor((now - state.startTime) / 1000);
                    } else {
                        elapsedSeconds = state.elapsedSeconds || 0;
                    }
                    startTimer(true); // resume
                } else {
                    // Countdown modes
                    if (state.targetTime) {
                        const remaining = Math.floor((state.targetTime - now) / 1000);
                        if (remaining > 0) {
                            timeLeft = remaining;
                            startTimer(true); // resume
                        } else {
                            // Timer finished while we were away
                            timeLeft = 0;
                            isRunning = false;
                            updateDisplay();
                            if (startBtn) startBtn.textContent = 'Start';
                            saveState(); // Update to stopped
                        }
                    } else {
                        timeLeft = state.timeLeft || timerConfig[currentMode].duration;
                        startTimer(true);
                    }
                }
            } else {
                // Paused state
                if (currentMode === 'stopwatch') {
                    elapsedSeconds = state.elapsedSeconds || 0;
                } else {
                    timeLeft = state.timeLeft || timerConfig[currentMode].duration;
                }

                // Safety checks for NaN
                if (isNaN(elapsedSeconds)) elapsedSeconds = 0;
                if (isNaN(timeLeft)) timeLeft = timerConfig[currentMode].duration;

                updateDisplay();
                if (startBtn) startBtn.textContent = 'Start';
            }
        } catch (e) {
            console.error("Error loading pomodoro state", e);
            localStorage.removeItem('pomodoroState');
        }
    }

    function updateModeButtons(mode) {
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        if (mode === 'focus' && focusMode) focusMode.classList.add('active');
        if (mode === 'shortBreak' && shortBreakMode) shortBreakMode.classList.add('active');
        if (mode === 'stopwatch' && stopwatchMode) stopwatchMode.classList.add('active');
    }

    // --- Core Timer Logic ---

    function updateDisplay() {
        let secondsToDisplay;
        if (currentMode === 'stopwatch') {
            secondsToDisplay = elapsedSeconds;
        } else {
            secondsToDisplay = timeLeft;
        }

        // Safety check
        if (isNaN(secondsToDisplay)) {
            secondsToDisplay = (currentMode === 'stopwatch') ? 0 : timerConfig[currentMode].duration;
        }

        const minutes = Math.floor(secondsToDisplay / 60);
        const seconds = secondsToDisplay % 60;

        if (timerDisplay) {
            timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }

        // Update browser tab title
        document.title = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} - StudyVerse`;
    }

    function startTimer(resume = false) {
        if (isRunning && !resume) return;

        isRunning = true;
        if (startBtn) startBtn.textContent = 'Pause';

        // Save state immediately when starting
        saveState();

        if (timerInterval) clearInterval(timerInterval);

        timerInterval = setInterval(() => {
            if (currentMode === 'stopwatch') {
                elapsedSeconds++;
                updateDisplay();
            } else {
                timeLeft--;
                updateDisplay();

                if (timeLeft <= 0) {
                    clearInterval(timerInterval);
                    isRunning = false;

                    // Save the completed session
                    const completedMode = currentMode;
                    saveSession(completedMode, timerConfig[completedMode].duration);

                    // Auto switch logic
                    if (currentMode === 'focus') {
                        switchMode('shortBreak');
                    } else {
                        switchMode('focus');
                    }
                    if (startBtn) startBtn.textContent = 'Start';
                    document.title = "Time's Up! - StudyVerse";
                }
            }
            // Save state periodically
            saveState();
        }, 1000);
    }

    function pauseTimer() {
        if (!isRunning) return;

        clearInterval(timerInterval);
        isRunning = false;
        if (startBtn) startBtn.textContent = 'Start';
        saveState();
        document.title = "StudyVerse";
    }

    function resetTimer() {
        // Calculate time spent based on mode
        let timeSpent = 0;
        if (currentMode === 'stopwatch') {
            timeSpent = elapsedSeconds;
        } else {
            // For countdown modes (Focus/Break), time spent is duration minus time left
            const totalDuration = timerConfig[currentMode].duration;
            timeSpent = totalDuration - timeLeft;
        }

        // Keep save threshold at 60 seconds
        if (timeSpent > 60) {
            if (confirm(`You have focused for ${Math.floor(timeSpent / 60)} minutes. Save this session before resetting?`)) {
                saveSession(currentMode, timeSpent);
            }
        }

        clearInterval(timerInterval);
        isRunning = false;

        if (currentMode === 'stopwatch') {
            elapsedSeconds = 0;
        } else {
            timeLeft = timerConfig[currentMode].duration;
        }

        updateDisplay();
        if (startBtn) startBtn.textContent = 'Start';
        document.title = "StudyVerse";

        // Clear persistence
        localStorage.removeItem('pomodoroState');
        // Save the reset state to overwrite any running state
        saveState();
    }

    function switchMode(mode) {
        if (isRunning) pauseTimer();

        // Save stopwatch session if switching away
        if (currentMode === 'stopwatch' && elapsedSeconds > 60) {
            saveSession('stopwatch', elapsedSeconds);
        }

        currentMode = mode;

        if (mode === 'stopwatch') {
            elapsedSeconds = 0;
        } else {
            timeLeft = timerConfig[mode].duration;
        }

        updateDisplay();
        updateModeButtons(mode);
        saveState();
    }

    function saveSession(mode, durationInSeconds) {
        // Don't save very short sessions (less than 1 min)
        if (durationInSeconds < 60) return;

        const form = new FormData();
        const durationMinutes = Math.floor(durationInSeconds / 60);
        form.append('duration', String(durationMinutes));
        form.append('mode', mode);

        // Add subject_id
        const subjectSelect = document.getElementById('subject-select');
        if (subjectSelect && subjectSelect.value) {
            form.append('subject_id', subjectSelect.value);
        }

        console.log(`Saving ${mode} session: ${durationMinutes} minutes`);

        fetch('/pomodoro/sessions', {
            method: 'POST',
            body: form
        })
            .then(response => response.json())
            .then(data => {
                console.log('Session saved successfully:', data);
                if (window.StudyVerse && window.StudyVerse.showToast) {
                    window.StudyVerse.showToast(`${mode} session (${durationMinutes}m) saved!`, 'success');
                }
            })
            .catch(error => {
                console.error('Error saving session:', error);
            });
    }

    // --- Subject Fetching ---
    function loadSubjects() {
        const select = document.getElementById('subject-select');
        if (!select) return;

        fetch('/api/subjects')
            .then(res => res.json())
            .then(data => {
                data.forEach(sub => {
                    const opt = document.createElement('option');
                    opt.value = sub.id;
                    opt.textContent = sub.name;
                    opt.style.background = '#1e1e1e';
                    select.appendChild(opt);
                });

                // Restore selection if saved? Optional.
                const savedSub = localStorage.getItem('pomodoroSubject');
                if (savedSub) select.value = savedSub;

                select.addEventListener('change', () => {
                    localStorage.setItem('pomodoroSubject', select.value);
                });
            })
            .catch(err => console.error("Failed to load subjects:", err));
    }
    loadSubjects();

    // Event listeners
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (isRunning) {
                pauseTimer();
            } else {
                startTimer();
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetTimer);
    }

    if (focusMode) {
        focusMode.addEventListener('click', () => switchMode('focus'));
    }

    if (shortBreakMode) {
        shortBreakMode.addEventListener('click', () => switchMode('shortBreak'));
    }

    if (stopwatchMode) {
        stopwatchMode.addEventListener('click', () => switchMode('stopwatch'));
    }

    // Save state before leaving the page
    window.addEventListener('beforeunload', () => {
        saveState();
    });

    // Handle tab visibility changes (optional sync)
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            loadState();
        }
    });

    // Initialize
    loadState(); // Check for persisted timer
    if (!isRunning) {
        updateDisplay();
    }

    // --- Brain Dump Logic ---
    const dumpArea = document.getElementById('brain-dump-area');
    const clearBtn = document.getElementById('clear-notes');
    const saveStatus = document.getElementById('save-status');

    if (dumpArea) {
        // Load saved dump
        const savedDump = localStorage.getItem('StudyVerse_brainDump');
        if (savedDump) dumpArea.value = savedDump;

        // Auto-save on input
        let saveTimeout;
        dumpArea.addEventListener('input', () => {
            if (saveStatus) {
                saveStatus.style.opacity = '1';
                saveStatus.textContent = 'Saving...';
            }

            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                localStorage.setItem('StudyVerse_brainDump', dumpArea.value);
                if (saveStatus) {
                    saveStatus.textContent = 'Saved';
                    setTimeout(() => { saveStatus.style.opacity = '0'; }, 2000);
                }
            }, 1000);
        });
    }

    if (clearBtn && dumpArea) {
        clearBtn.addEventListener('click', () => {
            if (confirm("Clear your Brain Dump? This cannot be undone.")) {
                dumpArea.value = '';
                localStorage.removeItem('StudyVerse_brainDump');
                if (saveStatus) {
                    saveStatus.textContent = 'Cleared';
                    saveStatus.style.opacity = '1';
                    setTimeout(() => { saveStatus.style.opacity = '0'; }, 2000);
                }
            }
        });
    }

    // --- Session Goals Logic ---
    const goalsList = document.getElementById('session-goals-list');
    const newGoalInput = document.getElementById('new-goal-input');
    const addGoalBtn = document.getElementById('add-goal-btn');

    function fetchSessionGoals() {
        if (!goalsList) return;

        fetch('/pomodoro/goals')
            .then(res => res.json())
            .then(goals => renderGoals(goals))
            .catch(err => console.error('Error fetching goals:', err));
    }

    function renderGoals(goals) {
        goalsList.innerHTML = '';
        if (goals.length === 0) {
            goalsList.innerHTML = `
                <div style="text-align: center; color: var(--text-secondary); padding: 20px; font-size: 0.9rem;">
                    No active goals. Add one above!
                </div>`;
            return;
        }

        goals.forEach(goal => {
            const el = document.createElement('div');
            el.style.cssText = `
                display: flex; align-items: center; gap: 10px; 
                padding: 8px 12px; background: rgba(255,255,255,0.05); 
                border-radius: 8px; border: 1px solid var(--border);
                transition: all 0.2s ease;
            `;

            const isCompleted = goal.completed;
            const textStyle = isCompleted ? 'text-decoration: line-through; color: var(--text-secondary);' : 'color: white;';
            const iconClass = isCompleted ? 'fa-circle-check' : 'fa-circle';
            const iconColor = isCompleted ? 'var(--accent-green)' : 'var(--text-secondary)';

            el.innerHTML = `
                <i class="fa-regular ${iconClass} goal-toggle" data-id="${goal.id}" style="cursor: pointer; color: ${iconColor}; font-size: 1.1rem;"></i>
                <div class="goal-title" data-id="${goal.id}" style="flex: 1; ${textStyle} cursor: text;">${goal.title}</div>
                <div style="display: flex; gap: 8px; opacity: 0.6;">
                    <i class="fa-solid fa-pen goal-edit" data-id="${goal.id}" style="cursor: pointer; font-size: 0.8rem;"></i>
                    <i class="fa-solid fa-trash goal-delete" data-id="${goal.id}" style="cursor: pointer; font-size: 0.8rem; color: #ff6b6b;"></i>
                </div>
            `;
            goalsList.appendChild(el);
        });

        // Re-attach listeners
        attachGoalListeners();
    }

    function attachGoalListeners() {
        document.querySelectorAll('.goal-toggle').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                toggleGoal(id);
            });
        });

        document.querySelectorAll('.goal-delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                deleteGoal(id);
            });
        });

        document.querySelectorAll('.goal-edit').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                startEditingGoal(id);
            });
        });

        // Also allow clicking text to edit if not completed
        document.querySelectorAll('.goal-title').forEach(div => {
            div.addEventListener('click', (e) => {
                // if (e.target.style.textDecoration.includes('line-through')) return;
                const id = e.target.dataset.id;
                startEditingGoal(id);
            });
        });
    }

    function startEditingGoal(id) {
        const titleDiv = document.querySelector(`.goal-title[data-id="${id}"]`);
        if (!titleDiv) return;

        const currentText = titleDiv.textContent;
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentText;
        input.style.cssText = `
            width: 100%; bg: transparent; border: none; color: white; 
            border-bottom: 1px solid var(--accent-primary); outline: none;
            background: rgba(0,0,0,0.5); padding: 2px 4px; border-radius: 4px;
        `;

        titleDiv.replaceWith(input);
        input.focus();

        const saveEdit = () => {
            const newTitle = input.value.trim();
            if (newTitle && newTitle !== currentText) {
                updateGoal(id, newTitle);
            } else {
                fetchSessionGoals(); // Revert
            }
        };

        input.addEventListener('blur', saveEdit);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                input.blur();
            }
        });
    }

    function addGoal() {
        const title = newGoalInput.value.trim();
        if (!title) return;

        fetch('/pomodoro/goals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: title })
        })
            .then(res => res.json())
            .then(data => {
                newGoalInput.value = '';
                fetchSessionGoals();
            });
    }

    function toggleGoal(id) {
        fetch(`/pomodoro/goals/${id}/toggle`, { method: 'POST' })
            .then(res => res.json())
            .then(() => fetchSessionGoals());
    }

    function deleteGoal(id) {
        if (!confirm('Delete this goal?')) return;
        fetch(`/pomodoro/goals/${id}/delete`, { method: 'POST' })
            .then(res => res.json())
            .then(() => fetchSessionGoals());
    }

    function updateGoal(id, newTitle) {
        fetch(`/pomodoro/goals/${id}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        })
            .then(res => res.json())
            .then(() => fetchSessionGoals());
    }

    if (addGoalBtn && newGoalInput) {
        addGoalBtn.addEventListener('click', addGoal);
        newGoalInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') addGoal();
        });
    }

    // Initialize Goals
    fetchSessionGoals();

});