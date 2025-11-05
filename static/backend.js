document.addEventListener('DOMContentLoaded', function() {
    // ----------------------------------------------
    // 1. VARIABLE DECLARATION (Used across both pages)
    // ----------------------------------------------
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    
    const idInput = document.getElementById('inputField');
    const passwordInput = document.getElementById('passwordField');
    const submitbtn = document.getElementById('submitButton'); // Login button
    const errorbox = document.getElementById('errorbox');
    
    const submitbtn2 = document.getElementById('submitButton-2'); // Sign-up button
    const errorbox2 = document.getElementById('errorbox-2');


    // ----------------------------------------------
    // 2. INITIAL SETUP
    // ----------------------------------------------
    if (errorbox) {
        errorbox.classList.add('hidden');
    }
    if (errorbox2) {
        errorbox2.classList.add('hidden');
    }

async function loadProtectedPage(url) {
    const token = localStorage.getItem('authToken');

    if (!token) {
        window.location.href = '/';
        return;
    }

    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const pageHtml = await response.text();
            document.open();
            document.write(pageHtml);
            document.close();
        } else {
            localStorage.removeItem('authToken');
            alert('Your session has expired. Please log in again.');
            window.location.href = '/';
        }
    } catch (error) {
        console.error('Failed to load protected page:', error);
        alert('An error occurred. Please try again.');
    }
}

    // ----------------------------------------------
    // 3. LOGIN LOGIC (index.html) : ( -- No work
    // ----------------------------------------------
if (loginForm) {
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault(); 

        const formData = new FormData(loginForm);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify(data) // Blyat
            });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.message || 'Login failed.');
                }
                
                localStorage.setItem('authToken', result.token);
                
                // --- REDIRECT LOGIC ---
                switch (result.role) {
                    case 'admin':
                        loadProtectedPage('/admin-page'); // Use the new function
                        break;
                    case 'specialist':
                        loadProtectedPage('/specialist-page'); // Use the new function
                        break;
                    default:
                        loadProtectedPage('/user_page'); // Use the new function
                        break;
}

            } catch (error) {
                const errorbox = document.getElementById('errorbox');
                if (errorbox) {
                    errorbox.querySelector('.errortext').textContent = error.message;
                    errorbox.classList.remove('hidden');
                }
            }
        });
    }


    // ----------------------------------------------
    // 4. SIGN-UP LOGIC (sign-up.html)
    // ----------------------------------------------
    if (submitbtn2 && signupForm) { 
        submitbtn2.addEventListener('click', async function(e) { 
            e.preventDefault(); 
            
            // ------------------------------------------------------------
            // Password confirmation section

            const password = document.getElementById('passwordField-2').value;
            const confpass = document.getElementById('passwordField-3').value;
            if (password!= confpass) {
                if (errorbox2) {
                    errorbox2.querySelector('.errortext-2').textContent = 'Error: Passwords do not match!';
                    errorbox2.classList.remove('hidden');
                }
                return;
            }
            if (password.length>1) {
                if (password.length<6) {
                    errorbox2.querySelector('.errortext-2').textContent = 'Password is too short!';
                    errorbox2.classList.remove('hidden');
                    return;
                }
            }

            if (errorbox2) {
                errorbox2.classList.add('hidden');
            }

            // ------------------------------------------------------------

            const formData = new FormData(signupForm); 
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/sign-up', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (response.ok && password.length>2) {
                    alert('Sign-up successful! Please log in.');
                    window.location.href = '/'; 
                } else {
                    const result = await response.json();
                    throw new Error(result.message || `Sign-up failed: ${response.status}`)
                }

            } catch (error) {
                const errorbox = document.getElementById('errorbox-2');
                if (errorbox) {
                    errorbox.querySelector('.errortext-2').textContent = error.message;
                    errorbox.classList.remove('hidden');
                }
            }
        });
    }

    // ----------------------------------------------
    // 5. KEYBOARD NAVIGATION (Login Page Only)
    // ----------------------------------------------
    if (idInput && passwordInput) {
        idInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                passwordInput.focus();
            }
        });
        passwordInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                // Submitting the form by clicking the login button
                if (submitbtn) {
                    submitbtn.click(); 
                }
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                idInput.focus();
            }
        });
    }
});