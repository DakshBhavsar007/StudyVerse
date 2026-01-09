<<<<<<< HEAD
// Main JavaScript file for StudyVerse

// Sidebar toggle
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarLogo = document.getElementById('sidebarLogo');

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }

    // Set active navigation item based on current page
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (currentPath === href || (href === '/dashboard' && currentPath === '/')) {
            item.classList.add('active');
        }
    });
});

// Toast notification system
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Utility functions
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric'
    });
}

// API helper functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'An error occurred');
        }

        return data;
    } catch (error) {
        console.error('API call error:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

// Export for use in other scripts
window.StudyVerse = {
    showToast,
    formatTime,
    formatDate,
    apiCall
};

// Level Up Logic
function checkLevelUp() {
    if (typeof window.currentUserLevel === 'undefined') return;

    const serverLevel = window.currentUserLevel;
    const storedLevel = localStorage.getItem('userLevel');

    // If no stored level or stored is null/undefined, set it and return (don't show on first ever load)
    // Or actually, user might want to see it if they just leveled up? 
    // Let's stick to the plan: if stored exists and server > stored -> Show.
    // If not stored, just set it.

    if (!storedLevel) {
        localStorage.setItem('userLevel', serverLevel);
        return;
    }

    if (parseInt(serverLevel) > parseInt(storedLevel)) {
        showLevelUpModal(serverLevel);
        localStorage.setItem('userLevel', serverLevel);
    }
}

function showLevelUpModal(level) {
    const modal = document.getElementById('levelUpModal');
    const display = document.getElementById('new-level-display');
    if (!modal || !display) return;

    display.innerText = level;
    modal.classList.add('show');
    startConfetti();
}

window.closeLevelUpModal = function () {
    const modal = document.getElementById('levelUpModal');
    if (modal) modal.classList.remove('show');
};

// Simple Confetti Implementation
// Simple Confetti Implementation
function startConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = canvas.parentElement.offsetHeight;

    const particles = [];
    const colors = ['#4ade80', '#ffffff', '#f59e0b', '#ef4444', '#3b82f6'];

    // Create initial particles
    for (let i = 0; i < 100; i++) {
        particles.push(createParticle(canvas, colors, true));
    }

    function animate() {
        // Stop if modal is closed
        const modal = document.getElementById('levelUpModal');
        if (!modal || !modal.classList.contains('show')) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach((p, index) => {
            p.x += p.vw;
            p.y += p.vh;
            p.vh += 0.2; // gravity
            p.rotation += p.velRotation;
            p.vw *= p.decay;
            p.vh *= p.decay; // air resistance

            // Render
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation * Math.PI / 180);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
            ctx.restore();

            // Reset if out of bounds
            if (p.y > canvas.height + 20) {
                particles[index] = createParticle(canvas, colors, false);
            }
        });

        requestAnimationFrame(animate);
    }
    animate();
}

function createParticle(canvas, colors, initial) {
    return {
        x: Math.random() * canvas.width,
        y: initial ? Math.random() * canvas.height : -20,
        w: Math.random() * 10 + 5,
        h: Math.random() * 10 + 5,
        vw: (Math.random() - 0.5) * 10,
        vh: Math.random() * 5 + 2, // consistently down
        color: colors[Math.floor(Math.random() * colors.length)],
        rotation: Math.random() * 360,
        velRotation: (Math.random() - 0.5) * 10,
        decay: Math.random() * 0.05 + 0.95 // slower decay
    };
}

// Call checkLevelUp on load
document.addEventListener('DOMContentLoaded', () => {
    checkLevelUp();
});
=======
// Main JavaScript file for StudyVerse

// Sidebar toggle
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarLogo = document.getElementById('sidebarLogo');

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }

    // Set active navigation item based on current page
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (currentPath === href || (href === '/dashboard' && currentPath === '/')) {
            item.classList.add('active');
        }
    });
});

// Toast notification system
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Utility functions
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric'
    });
}

// API helper functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'An error occurred');
        }

        return data;
    } catch (error) {
        console.error('API call error:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

// Export for use in other scripts
window.StudyVerse = {
    showToast,
    formatTime,
    formatDate,
    apiCall
};

// Level Up Logic
function checkLevelUp() {
    if (typeof window.currentUserLevel === 'undefined') return;

    const serverLevel = window.currentUserLevel;
    const storedLevel = localStorage.getItem('userLevel');

    // If no stored level or stored is null/undefined, set it and return (don't show on first ever load)
    // Or actually, user might want to see it if they just leveled up? 
    // Let's stick to the plan: if stored exists and server > stored -> Show.
    // If not stored, just set it.

    if (!storedLevel) {
        localStorage.setItem('userLevel', serverLevel);
        return;
    }

    if (parseInt(serverLevel) > parseInt(storedLevel)) {
        showLevelUpModal(serverLevel);
        localStorage.setItem('userLevel', serverLevel);
    }
}

function showLevelUpModal(level) {
    const modal = document.getElementById('levelUpModal');
    const display = document.getElementById('new-level-display');
    if (!modal || !display) return;

    display.innerText = level;
    modal.classList.add('show');
    startConfetti();
}

window.closeLevelUpModal = function () {
    const modal = document.getElementById('levelUpModal');
    if (modal) modal.classList.remove('show');
};

// Simple Confetti Implementation
// Simple Confetti Implementation
function startConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = canvas.parentElement.offsetHeight;

    const particles = [];
    const colors = ['#4ade80', '#ffffff', '#f59e0b', '#ef4444', '#3b82f6'];

    // Create initial particles
    for (let i = 0; i < 100; i++) {
        particles.push(createParticle(canvas, colors, true));
    }

    function animate() {
        // Stop if modal is closed
        const modal = document.getElementById('levelUpModal');
        if (!modal || !modal.classList.contains('show')) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach((p, index) => {
            p.x += p.vw;
            p.y += p.vh;
            p.vh += 0.2; // gravity
            p.rotation += p.velRotation;
            p.vw *= p.decay;
            p.vh *= p.decay; // air resistance

            // Render
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation * Math.PI / 180);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
            ctx.restore();

            // Reset if out of bounds
            if (p.y > canvas.height + 20) {
                particles[index] = createParticle(canvas, colors, false);
            }
        });

        requestAnimationFrame(animate);
    }
    animate();
}

function createParticle(canvas, colors, initial) {
    return {
        x: Math.random() * canvas.width,
        y: initial ? Math.random() * canvas.height : -20,
        w: Math.random() * 10 + 5,
        h: Math.random() * 10 + 5,
        vw: (Math.random() - 0.5) * 10,
        vh: Math.random() * 5 + 2, // consistently down
        color: colors[Math.floor(Math.random() * colors.length)],
        rotation: Math.random() * 360,
        velRotation: (Math.random() - 0.5) * 10,
        decay: Math.random() * 0.05 + 0.95 // slower decay
    };
}

// Call checkLevelUp on load
document.addEventListener('DOMContentLoaded', () => {
    checkLevelUp();
});
>>>>>>> f5971551499e078e14bc7548b7a15e1f97eb6644
