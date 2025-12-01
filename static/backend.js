document.addEventListener('DOMContentLoaded', function() {
    // ----------------------------------------------
    // 1. VARIABLE DECLARATION (Used across both pages)
    // ----------------------------------------------
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    
    const idInput = document.getElementById('inputField');
    const passwordInput = document.getElementById('passwordField');
    const submitbtn = document.getElementById('submitButton');
    const errorbox = document.getElementById('errorbox');
    
    const submitbtn2 = document.getElementById('submitButton-2');
    const errorbox2 = document.getElementById('errorbox-2');

    const catselect = document.getElementById('catSelect');
    const roleSelect = document.getElementById('roleSelect');
    const categoryGroup = document.getElementById('categoryGroup');

    // Sign-up specific inputs
    const idInput2 = document.getElementById('inputField-2');
    const passwordInput2 = document.getElementById('passwordField-2');
    const confpassInput2 = document.getElementById('passwordField-3');

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
        window.location.href = '/login_page';
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
            window.location.href = '/login_page';
        }
    } catch (error) {
        console.error('Failed to load protected page:', error);
        alert('An error occurred. Please try again.');
    }
}

    // ----------------------------------------------
    // 3. LOGIN LOGIC (index.html)
    // ----------------------------------------------
if (loginForm) {
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault(); 
        
        // Manual data construction to fix the JSON parsing error (415)
        const rememberCheckbox = document.getElementById('rememberme');

        const data = {
            username: idInput.value,
            password: passwordInput.value,
            remember: rememberCheckbox ? rememberCheckbox.checked : false 
        };

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || 'Login failed.');
            }
            
            localStorage.setItem('authToken', result.token);
            
            // --- REDIRECT LOGIC ---
            switch (result.role) {
                case 'admin':
                    loadProtectedPage('/admin-page');
                    break;
                case 'specialist':
                    loadProtectedPage('/specialist-page');
                    break;
                default:
                    loadProtectedPage('/user_page');
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
    // SIGN-UP ROLE CHANGE LOGIC (Runs on load and change)
    // ----------------------------------------------
    if (roleSelect && categoryGroup && catselect) {
        function handleRoleChange() {
            const isInvestor = roleSelect.value === 'investor';
            
            if (isInvestor) {
                categoryGroup.classList.add('hidden');                        
                catselect.value = ""; 
            } else {
                categoryGroup.classList.remove('hidden');
            }
        }
        handleRoleChange();
        roleSelect.addEventListener('change', handleRoleChange);
    }

    // ----------------------------------------------
    // 4. SIGN-UP LOGIC (sign-up.html)
    // ----------------------------------------------
    if (submitbtn2 && signupForm) { 
        submitbtn2.addEventListener('click', async function(e) { 
            e.preventDefault(); 
            
            // --- Collect all values for validation and submission ---
            const username = idInput2.value;
            const password = passwordInput2.value;
            const confpass = confpassInput2.value;
            const currentRole = roleSelect.value;
            const companyCategory = catselect.value;

            // --- Validation Checks ---
            if (password != confpass) {
                if (errorbox2) {
                    errorbox2.querySelector('.errortext-2').textContent = 'Error: Passwords do not match';
                    errorbox2.classList.remove('hidden');
                }
                return;
            }
            
            if (password.length > 1 && password.length < 6) {
                errorbox2.querySelector('.errortext-2').textContent = 'Error: Password is too short (min 6 characters)';
                errorbox2.classList.remove('hidden');
                return;
            }

            // Check if a category was selected ONLY if not an investor
            if (currentRole !== 'investor' && (companyCategory == "" || companyCategory == "Select Category")) {
                errorbox2.querySelector('.errortext-2').textContent = 'Error: Please select a Company Category.';
                errorbox2.classList.remove('hidden');
                return;
            }
            
            if (errorbox2) {
                errorbox2.classList.add('hidden');
            }

            // --- Data Construction for API Payload ---
            const data = {
                username: username,
                password: password,
                role: currentRole,
                company_category: companyCategory 
            };

            try {
                const response = await fetch('/sign-up', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (response.ok) {
                    alert('Sign-up successful! Please log in.');
                    window.location.href = '/login_page'; 
                } else {
                    const result = await response.json();
                    throw new Error(result.message || `Sign-up failed: ${response.status}`)
                }

            } catch (error) {
                if (errorbox2) { 
                    const errorTextElement = errorbox2.querySelector('.errortext-2');
                    if (errorTextElement) {
                        errorTextElement.textContent = error.message;
                        errorbox2.classList.remove('hidden');
                    } else {
                        console.error('Missing .errortext-2 element in #errorbox-2', error);
                    }
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