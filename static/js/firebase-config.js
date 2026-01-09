// Firebase Configuration Template
// Replace the values below with your actual Firebase project credentials

const firebaseConfig = {
    apiKey: "YOUR_API_KEY_HERE",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
    appId: "YOUR_APP_ID",
    measurementId: "YOUR_MEASUREMENT_ID" // Optional
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Initialize Firebase Authentication
const auth = firebase.auth();

// Google Sign-In Provider
const googleProvider = new firebase.auth.GoogleAuthProvider();

// Google Sign-In Function
function signInWithGoogle() {
    auth.signInWithPopup(googleProvider)
        .then((result) => {
            const user = result.user;
            console.log('Google Sign-In successful:', user);

            // Send user data to Flask backend
            fetch('/api/auth/google', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: user.email,
                    displayName: user.displayName,
                    photoURL: user.photoURL,
                    uid: user.uid,
                    idToken: result.credential.idToken
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        window.location.href = '/dashboard';
                    } else {
                        console.error('Backend authentication failed:', data);
                    }
                })
                .catch(error => {
                    console.error('Error sending to backend:', error);
                });
        })
        .catch((error) => {
            console.error('Google Sign-In error:', error);
            alert('Sign-in failed: ' + error.message);
        });
}

// Auth State Observer
auth.onAuthStateChanged((user) => {
    if (user) {
        console.log('User is signed in:', user.email);
    } else {
        console.log('User is signed out');
    }
});

// Sign Out Function
function signOut() {
    auth.signOut().then(() => {
        console.log('User signed out');
        window.location.href = '/auth';
    }).catch((error) => {
        console.error('Sign out error:', error);
    });
}
