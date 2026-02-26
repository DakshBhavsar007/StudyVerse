
/**
 * StudyVerse Procedural Particle System
 * Renders infinite falling background animations based on the active theme.
 * Uses HTML5 Canvas for performance and crisp vectors.
 */

class ParticleSystem {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.activeTheme = 'dark'; // Default
        this.animationFrame = null;
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.init();
    }

    init() {
        // Setup Canvas
        this.canvas.id = 'theme-particles';
        this.canvas.style.position = 'fixed';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '9997'; // Just behind the CSS overlay (9998) if any, or matching it
        document.body.appendChild(this.canvas);

        // Resize Listener
        window.addEventListener('resize', () => this.resize());
        this.resize();

        // Theme Observer
        this.observeTheme();

        // Start Loop
        this.animate();
    }

    resize() {
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.canvas.width = this.width;
        this.canvas.height = this.height;
    }

    observeTheme() {
        // Observer for body class changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'class') {
                    this.updateTheme();
                }
            });
        });
        observer.observe(document.body, { attributes: true });

        // Initial check
        this.updateTheme();
    }

    updateTheme() {
        const bodyClasses = document.body.classList;
        let newTheme = 'dark'; // Default no animation

        // Detect theme from class
        if (bodyClasses.contains('theme_cyberpunk')) newTheme = 'cyberpunk';
        else if (bodyClasses.contains('theme_matrix')) newTheme = 'matrix';
        else if (bodyClasses.contains('theme_sunset')) newTheme = 'sunset';
        else if (bodyClasses.contains('theme_ocean')) newTheme = 'ocean';
        else if (bodyClasses.contains('theme_midnight')) newTheme = 'midnight';
        else if (bodyClasses.contains('theme_forest')) newTheme = 'forest';
        else if (bodyClasses.contains('theme_aurora')) newTheme = 'aurora';
        else if (bodyClasses.contains('theme_lava')) newTheme = 'lava';
        else if (bodyClasses.contains('theme_synthwave')) newTheme = 'synthwave';
        else if (bodyClasses.contains('theme_retro')) newTheme = 'retro';
        else if (bodyClasses.contains('theme_blood_moon')) newTheme = 'blood_moon';
        else if (bodyClasses.contains('theme_toxic')) newTheme = 'toxic';
        else if (bodyClasses.contains('theme_neon_city')) newTheme = 'neon_city';
        else if (bodyClasses.contains('theme_sakura')) newTheme = 'sakura';

        if (this.activeTheme !== newTheme) {
            this.activeTheme = newTheme;
            this.resetParticles();
        }
    }

    resetParticles() {
        this.particles = [];

        // Dark/Light themes have no particles
        if (this.activeTheme === 'dark' || this.activeTheme === 'light') {
            return;
        }

        // Initial spawn density
        const density = this.getThemeDensity();
        for (let i = 0; i < density; i++) {
            this.particles.push(this.createParticle(true));
        }
    }

    getThemeDensity() {
        switch (this.activeTheme) {
            case 'matrix': return 150;
            case 'neon_city': return 200;
            case 'sakura': return 40; // Flowers
            case 'synthwave': return 50; // Music notes
            default: return 80;
        }
    }

    createParticle(randomY = false) {
        const p = {
            x: Math.random() * this.width,
            y: randomY ? Math.random() * this.height : -50,
            speed: 1 + Math.random() * 2,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.3,
            rotation: Math.random() * 360,
            rotationSpeed: (Math.random() - 0.5) * 2,
            oscillation: Math.random() * Math.PI * 2,
            type: this.activeTheme
        };

        if (this.activeTheme === 'matrix') {
            const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            p.char = chars[Math.floor(Math.random() * chars.length)];
            p.speed = 2 + Math.random() * 3;
            p.size = 10 + Math.random() * 10;
            p.color = '#22c55e';
        } else if (this.activeTheme === 'synthwave') {
            // Music Symbols
            const notes = ["♪", "♫", "♩", "♬", "♭"];
            p.char = notes[Math.floor(Math.random() * notes.length)];
            p.color = Math.random() > 0.5 ? '#e879f9' : '#22d3ee'; // Pink/Cyan
            p.size = 14 + Math.random() * 10;
            p.speed = 1.5 + Math.random();
        } else if (this.activeTheme === 'sakura') {
            p.color = '#fbcfe8'; // Pink
            p.size = 6 + Math.random() * 5;
            p.speed = 1 + Math.random();
            p.oscillationSpeed = 0.02 + Math.random() * 0.02;
        } else if (this.activeTheme === 'lava') {
            p.color = Math.random() > 0.5 ? '#ef4444' : '#f97316'; // Red/Orange
            p.speed = 2 + Math.random() * 2;
            p.size = 3 + Math.random() * 3;
        } else if (this.activeTheme === 'blood_moon') {
            p.color = '#b91c1c'; // Deep Red
            p.speed = 3 + Math.random() * 2; // Fast drip
            p.size = 3 + Math.random() * 2;
        } else if (this.activeTheme === 'toxic') {
            p.char = "☢";
            p.color = '#bef264'; // Lime
            p.size = 12 + Math.random() * 8;
        } else if (this.activeTheme === 'sunset') {
            p.char = "☀";
            p.color = Math.random() > 0.5 ? '#fb923c' : '#fcd34d'; // Orange/Yellow
            p.size = 12 + Math.random() * 8;
            p.speed = 0.5 + Math.random();
        } else if (this.activeTheme === 'midnight') {
            p.char = Math.random() > 0.5 ? "🌙" : "✨"; // Mix moons and sparkles
            p.color = '#e2e8f0'; // Slate 200
            p.size = 10 + Math.random() * 6;
            p.speed = 0.5 + Math.random();
        } else if (this.activeTheme === 'aurora') {
            p.char = "✨";
            p.color = Math.random() > 0.5 ? '#a78bfa' : '#34d399'; // Violet/Emerald
            p.size = 8 + Math.random() * 6;
            p.speed = 1 + Math.random(); // Floating
        }

        // ... (Keep existing logic for others or default) ...
        // Re-adding default fallback for others to ensure specific overrides work
        if (!p.color && !p.char) {
            // Fallbacks for other themes
            if (this.activeTheme === 'cyberpunk') {
                p.color = Math.random() > 0.5 ? '#d946ef' : '#06b6d4';
                p.speed = 3 + Math.random() * 4;
                p.w = Math.random() * 20 + 5;
                p.h = Math.random() * 2 + 1;
            } else if (this.activeTheme === 'neon_city') {
                p.color = '#06b6d4';
                p.speed = 15 + Math.random() * 10;
                p.w = 1;
                p.h = 20 + Math.random() * 20;
            } else if (this.activeTheme === 'retro') {
                p.color = '#4ade80';
                p.size = 4 + Math.random() * 4;
            } else if (this.activeTheme === 'ocean') {
                p.color = Math.random() > 0.5 ? '#38bdf8' : '#7dd3fc';
                p.size = 2 + Math.random() * 3;
            } else {
                // Defaults for Forest, etc.
                p.color = '#ffffff';
                if (this.activeTheme === 'forest') p.color = '#4ade80';
            }
        }

        return p;
    }

    update() {
        if (!this.particles.length) return;

        for (let i = 0; i < this.particles.length; i++) {
            let p = this.particles[i];

            // Movement
            p.y += p.speed;
            p.rotation += p.rotationSpeed;

            // Custom sway logic
            if (this.activeTheme === 'sakura' || this.activeTheme === 'forest') {
                p.oscillation += p.oscillationSpeed || 0.02;
                p.x += Math.sin(p.oscillation) * 1.5; // Gentle sway
            } else if (this.activeTheme === 'blood_moon' || this.activeTheme === 'neon_city' || this.activeTheme === 'lava') {
                // Straight down, minimal sway
            } else {
                p.oscillation += 0.02;
                p.x += Math.sin(p.oscillation) * 0.5;
            }

            // Reset if out of bounds
            if (p.y > this.height + 50) {
                this.particles[i] = this.createParticle();
            }
        }
    }

    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        if (!this.particles.length) return;

        this.ctx.save();

        for (let p of this.particles) {
            this.ctx.globalAlpha = p.opacity;

            if (this.activeTheme === 'matrix') {
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px monospace`;
                this.ctx.fillText(p.char, p.x, p.y);
            } else if (this.activeTheme === 'synthwave') {
                // Draw Music Notes
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px sans-serif`;
                this.ctx.fillText(p.char, p.x, p.y);
            } else if (this.activeTheme === 'toxic') {
                // Radioactive Symbol
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px sans-serif`;
                this.ctx.fillText("☢", p.x, p.y);
            } else if (this.activeTheme === 'sunset') {
                // Sun Symbol
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px sans-serif`;
                this.ctx.fillText("☀", p.x, p.y);
            } else if (this.activeTheme === 'midnight') {
                // Moon Symbol
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px sans-serif`;
                this.ctx.fillText("🌙", p.x, p.y);
            } else if (this.activeTheme === 'aurora') {
                // Sparkle/Star Symbol
                this.ctx.fillStyle = p.color;
                this.ctx.font = `${p.size}px sans-serif`;
                this.ctx.fillText("✨", p.x, p.y);
            } else if (this.activeTheme === 'sakura') {
                // Draw Flower
                this.ctx.fillStyle = p.color;
                this.ctx.translate(p.x, p.y);
                this.ctx.rotate(p.rotation * Math.PI / 180);
                // Simple 5-petal flower
                for (let i = 0; i < 5; i++) {
                    this.ctx.beginPath();
                    this.ctx.ellipse(0, -p.size / 2, p.size / 3, p.size / 1.5, 0, 0, Math.PI * 2);
                    this.ctx.fill();
                    this.ctx.rotate((Math.PI * 2) / 5);
                }
                this.ctx.beginPath(); // Center
                this.ctx.fillStyle = '#fff';
                this.ctx.arc(0, 0, p.size / 4, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.setTransform(1, 0, 0, 1, 0, 0);
            } else if (this.activeTheme === 'lava') {
                // Draw Fire Spark (Diamond/Kite shape)
                this.ctx.fillStyle = p.color;
                this.ctx.beginPath();
                this.ctx.moveTo(p.x, p.y - p.size);
                this.ctx.lineTo(p.x + p.size / 2, p.y);
                this.ctx.lineTo(p.x, p.y + p.size);
                this.ctx.lineTo(p.x - p.size / 2, p.y);
                this.ctx.fill();
            } else if (this.activeTheme === 'blood_moon') {
                // Draw Blood Drop
                this.ctx.fillStyle = p.color;
                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, p.size, 0, Math.PI, false); // Bottom half circle
                this.ctx.lineTo(p.x, p.y - p.size * 2.5); // Pointy top
                this.ctx.lineTo(p.x + p.size, p.y);
                this.ctx.fill();
            } else if (this.activeTheme === 'cyberpunk') {
                this.ctx.fillStyle = p.color;
                this.ctx.translate(p.x, p.y);
                this.ctx.rotate(p.rotation * Math.PI / 180);
                this.ctx.fillRect(0, 0, p.w, p.h);
                this.ctx.setTransform(1, 0, 0, 1, 0, 0);
            } else if (this.activeTheme === 'neon_city') {
                this.ctx.fillStyle = p.color;
                this.ctx.fillRect(p.x, p.y, p.w, p.h);
            } else if (this.activeTheme === 'retro') {
                this.ctx.fillStyle = p.color;
                this.ctx.fillRect(p.x, p.y, p.size, p.size);
            } else if (this.activeTheme === 'forest') {
                this.ctx.fillStyle = '#4ade80';
                this.ctx.beginPath();
                // Simpler leaf shape
                this.ctx.ellipse(p.x, p.y, p.size, p.size * 2, p.rotation * Math.PI / 180, 0, Math.PI * 2);
                this.ctx.fill();
            } else if (this.activeTheme === 'ocean') {
                // Bubbles (already default circle but ensure color)
                this.ctx.fillStyle = p.color || '#38bdf8';
                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                this.ctx.fill();
                // Add shine
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                this.ctx.beginPath();
                this.ctx.arc(p.x - p.size / 3, p.y - p.size / 3, p.size / 4, 0, Math.PI * 2);
                this.ctx.fill();
            } else {
                // Default circles
                this.ctx.fillStyle = p.color;
                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }

        this.ctx.restore();
    }

    animate() {
        this.update();
        this.draw();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.particleSystem = new ParticleSystem();
});
