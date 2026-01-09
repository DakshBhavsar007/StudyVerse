// Authentication JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabTriggers = document.querySelectorAll('.tabs-trigger');
    const tabContents = {
        'signin': document.getElementById('signin-tab'),
        'signup': document.getElementById('signup-tab')
    };

    tabTriggers.forEach(trigger => {
        trigger.addEventListener('click', () => {
            const tab = trigger.dataset.tab;
            
            // Update active state
            tabTriggers.forEach(t => t.classList.remove('active'));
            trigger.classList.add('active');
            
            // Show/hide content
            Object.values(tabContents).forEach(content => {
                if (content) content.style.display = 'none';
            });
            if (tabContents[tab]) {
                tabContents[tab].style.display = 'block';
            }
        });
    });

    // NOTE: Auth uses standard HTML forms posting to Flask routes (/signin, /signup).
    // We intentionally avoid JSON-based fetch here to keep the semester project simple.

    // Google sign in (placeholder)
    const googleSignIn = document.getElementById('googleSignIn');
    if (googleSignIn) {
        googleSignIn.addEventListener('click', () => {
            alert('Google sign in is not yet implemented. Please use email sign in.');
        });
    }
});
