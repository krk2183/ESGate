document.addEventListener('DOMContentLoaded', function() {
    // --- STATE & CHART INSTANCES ---
    let interestRateChart, defaultRateChart, sustainabilityScoreChart;

    // --- DOM ELEMENT SELECTORS (from Analysis Page) ---
    const calculateButton = document.getElementById('calculateButton');
    const fetchHistoryButton = document.getElementById('fetchHistoryButton');
    const companyNameInput = document.getElementById('companyName');
    
    // Result display elements
    const intRateResultEl = document.getElementById('intRateResult');
    const defaultRateResultEl = document.getElementById('defaultRateResult');
    const susScoreResultEl = document.getElementById('susScoreResult');
    const esgateScoreResultEl = document.getElementById('esgatescore');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // AI Analysis card elements
    const aiAnalysisCard = document.getElementById('aiAnalysisCard');
    const aiSummaryEl = document.getElementById('aiSummary');
    const aiStrengthsEl = document.getElementById('aiStrengths');
    const aiWeaknessesEl = document.getElementById('aiWeaknesses');
    const aiRecommendationsEl = document.getElementById('aiRecommendations');

    const accordionHeaders = document.querySelectorAll('.accordion-header');

    // Historical comparison elements
    const historyAnalysisCard = document.getElementById('historyAnalysisCard');
    const historySpinner = document.getElementById('historySpinner'); 
    const intRateChangeEl = document.getElementById('intRateChange');
    const defaultRateChangeEl = document.getElementById('defaultRateChange');
    const susScoreChangeEl = document.getElementById('susScoreChange');
    const improvementMessageEl = document.getElementById('improvementMessage');
    const aiIntRateCommentEl = document.getElementById('aiIntRateComment');
    const aiDefaultRateCommentEl = document.getElementById('aiDefaultRateComment');
    const aiSusScoreCommentEl = document.getElementById('aiSusScoreComment');
    const aiOverallHistorySummaryEl = document.getElementById('aiOverallHistorySummary');
    const aiHistoryErrorEl = document.getElementById('aiHistoryError');

    // Home page news
    const card_1_title = document.getElementById('card-1-title');
    const card_1_summary = document.getElementById('card-1-summary');
    const read_1 = document.getElementById('read-1');
    const card_2_title = document.getElementById('card-2-title');
    const card_2_summary = document.getElementById('card-2-summary');
    const read_2 = document.getElementById('read-2');
    const card_3_title = document.getElementById('card-3-title');
    const card_3_summary = document.getElementById('card-3-summary');
    const read_3 = document.getElementById('read-3');

    // --- NEW: PROTECTED PAGE LOADER FUNCTION ---
    // (This function was missing from your user-js.js)
    async function loadProtectedPage(url) {
        const token = localStorage.getItem('authToken');
        if (!token) {
            alert("Session expired or token not found. Please log in again.");
            window.location.href = '/login_page'; // Redirect to login
            return;
        }

        console.log(`Navigating to protected page: ${url}`); // Debug log
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}` // Send the token
                }
            });

            if (response.ok) {
                const pageHtml = await response.text();
                // Replace the entire document content
                document.open();
                document.write(pageHtml);
                document.close();
            } else {
                // Handle 401 Unauthorized (token expired, etc.)
                localStorage.removeItem('authToken');
                alert('Your session has expired. Please log in again.');
                window.location.href = '/login_page'; // Redirect to login
            }
        } catch (error) {
            console.error('Failed to load protected page:', error);
            alert('An error occurred while loading the page. Please try again.');
        }
    }


    // --- NEW: DELEGATED EVENT LISTENER FOR NAVIGATION ---
    // This one listener handles all protected nav links
    document.body.addEventListener('click', function(event) {
        // Find the closest <a> tag that has the 'protected-nav-link' class
        const protectedLink = event.target.closest('a.protected-nav-link'); 

        if (protectedLink) {
            event.preventDefault(); // Stop the link from navigating normally
            const url = protectedLink.getAttribute('href');
            
            // Check if the link is not just a placeholder
            if (url && url !== '#') {
                loadProtectedPage(url); // Use our function to load with token
            }
        }
        // If the click wasn't on a .protected-nav-link, it's ignored
    });


    // --- EVENT LISTENERS (for Analysis Page buttons) ---

    // Note: querySelectorAll is fine here as it runs once on load
    if (accordionHeaders) {
        accordionHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                header.classList.toggle('active');
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                    content.style.padding = "0 1rem";
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                    content.style.padding = "1rem";
                }
            });
        });
    }

    if (calculateButton){
        calculateButton.addEventListener('click', handleFullCalculation);
    }

    if (fetchHistoryButton){
        fetchHistoryButton.addEventListener('click', handleFetchHistory);
    }
    // HOME PAGE NEWS
    async function set_news_elements() {
        // V2 higher performance implementation
        const promise1 = fetch('/get_gemini_news',{});
        const promise2 = fetch('/get_gemini_news',{});
        const promise3 = fetch('/get_gemini_news',{});
        const [result1,result2,result3] = await Promise.all([
            promise1,promise2,promise3
        ]);
        const [data1,data2,data3]  = await Promise.all([
            result1.json(),result2.json(),result3.json()
        ]);
        card_1_title.innerHTML = data1.title;
        card_1_summary.innerHTML = data1.summary;
        read_1.href = data1.link;
        card_2_title.innerHTML = data2.title;
        card_2_summary.innerHTML = data2.summary;
        read_2.href = data2.link;
        card_3_title.innerHTML = data3.title;
        card_3_summary.innerHTML = data3.summary;
        read_3.href = data3.link; 

    }

    if (card_1_title) {
        set_news_elements();
    }


    // --- CORE FUNCTIONS (for Analysis Page) ---
    
    async function handleFullCalculation() {
        showLoading(true);
        clearResults();

        try {
            const [intRateData, defaultRateData, susScoreData] = await Promise.all([
                predictInterestRate(),
                predictDefaultRate(),
                predictSustainabilityScore()
            ]);

            // Check for errors from individual predictions
            if (intRateData.error) throw new Error(`Interest Rate: ${intRateData.error}`);
            if (defaultRateData.error) throw new Error(`Default Rate: ${defaultRateData.error}`);
            if (susScoreData.error) throw new Error(`Sustainability: ${susScoreData.error}`);
            if (susScoreData.sus_score === undefined) throw new Error("Missing 'sus_score' from AI.");

            updateResultsUI(intRateData.int_rate, defaultRateData.default_rate, susScoreData.sus_score);
            
            await fetchAiAnalysis(intRateData.int_rate, defaultRateData.default_rate, susScoreData.sus_score);

        } catch (error) {
            console.error("Calculation failed:", error.message);
            alert(`An error occurred during calculation: ${error.message}`);
        } finally {
            showLoading(false);
        }
    }
    
    async function handleFetchHistory() {
        // const companyName = companyNameInput.value.trim();
        // if (!companyName) {
        //     alert("Please enter a company name to fetch its history.");
        //     return;
        // }

        if(historySpinner) historySpinner.classList.remove('hidden'); 
        historyAnalysisCard.classList.add('hidden'); 
        if (aiHistoryErrorEl) aiHistoryErrorEl.textContent = ''; 

        try {
            const analysisData = await makeApiCall('/history', 'POST', {});

            if (analysisData.message) {
                 alert(analysisData.message);
                 if (interestRateChart) interestRateChart.destroy();
                 if (defaultRateChart) defaultRateChart.destroy();
                 if (sustainabilityScoreChart) sustainabilityScoreChart.destroy();
                 clearHistoryAnalysisUI();
                 return; 
            }

            if (analysisData.history && analysisData.history.length > 0) {
                renderHistoryCharts(analysisData.history);

            } else {
                 alert("No historical prediction data found for this company.");
                 if (interestRateChart) interestRateChart.destroy();
                 if (defaultRateChart) defaultRateChart.destroy();
                 if (sustainabilityScoreChart) sustainabilityScoreChart.destroy();
                 clearHistoryAnalysisUI();
                 return;
            }

            if (analysisData.improvement_metrics && analysisData.ai_analysis) {
                updateHistoryAnalysisUI(analysisData.improvement_metrics, analysisData.ai_analysis);
            }

        } catch (error) {
            console.error("Failed to fetch history:", error); 
            alert(`Could not fetch history: ${error.message}`);
            clearHistoryAnalysisUI();
            if (aiHistoryErrorEl) aiHistoryErrorEl.textContent = `Error: ${error.message}`;
            if (historyAnalysisCard) historyAnalysisCard.classList.remove('hidden');
        } finally {
             if(historySpinner) historySpinner.classList.add('hidden');
        }
    }
    
    
    function formatPercentageChange(value, higherIsBetter) {
        if (value === null || value === undefined || !isFinite(value)) {
            return 'N/A';
        }
        
        const isPositive = value > 0;
        const isNegative = value < 0;

        let prefix = isPositive ? '+' : '';
        let arrow = ''; 
        if (isPositive) {
            arrow = higherIsBetter ? '▲ Improvement' : '▼ Decline';
        } else if (isNegative) {
             arrow = higherIsBetter ? '▼ Decline' : '▲ Improvement';
        } else {
            arrow = '● No Change';
        }
        return `${prefix}${value.toFixed(1)}% ${arrow}`;
    }

    function clearResults() {
        if (intRateResultEl) intRateResultEl.textContent = '- %';
        if (defaultRateResultEl) defaultRateResultEl.textContent = '- %';
        if (susScoreResultEl) susScoreResultEl.textContent = '- / 10';
        if (esgateScoreResultEl) esgateScoreResultEl.textContent = '- / 100'; // New
        if (aiAnalysisCard) aiAnalysisCard.classList.add('hidden');
        clearHistoryAnalysisUI(); // Call the helper
    }

    function clearHistoryAnalysisUI() {
        if (!historyAnalysisCard) return; // Exit if elements aren't on this page
        intRateChangeEl.textContent = '-';
        defaultRateChangeEl.textContent = '-';
        susScoreChangeEl.textContent = '-';
        improvementMessageEl.textContent = '';
        aiIntRateCommentEl.textContent = 'Awaiting analysis...';
        aiDefaultRateCommentEl.textContent = 'Awaiting analysis...';
        aiSusScoreCommentEl.textContent = 'Awaiting analysis...';
        aiOverallHistorySummaryEl.textContent = 'Awaiting analysis...';
        aiHistoryErrorEl.textContent = '';
        historyAnalysisCard.classList.add('hidden');
    }

    function updateHistoryAnalysisUI(improvements, aiAnalysis) {
        if (!historyAnalysisCard) return; // Exit if elements aren't on this page
        improvementMessageEl.textContent = '';
        aiHistoryErrorEl.textContent = '';
        
        if (improvements) {
            if (improvements.message) { 
                improvementMessageEl.textContent = improvements.message;
                intRateChangeEl.textContent = 'N/A';
                defaultRateChangeEl.textContent = 'N/A';
                susScoreChangeEl.textContent = 'N/A';
            } else {
                intRateChangeEl.textContent = formatPercentageChange(improvements.int_rate_change, false); 
                defaultRateChangeEl.textContent = formatPercentageChange(improvements.default_rate_change, false); 
                susScoreChangeEl.textContent = formatPercentageChange(improvements.sus_score_change, true); 
            }
        } else {
            improvementMessageEl.textContent = 'Improvement data unavailable.';
            intRateChangeEl.textContent = 'N/A';
            defaultRateChangeEl.textContent = 'N/A';
            susScoreChangeEl.textContent = 'N/A';
        }

        if (aiAnalysis) {
            if (aiAnalysis.error) {
                aiHistoryErrorEl.textContent = `AI Analysis Error: ${aiAnalysis.error}`;
                aiIntRateCommentEl.textContent = '-';
                aiDefaultRateCommentEl.textContent = '-';
                aiSusScoreCommentEl.textContent = '-';
                aiOverallHistorySummaryEl.textContent = '-';
            } else if (aiAnalysis.message) {
                 aiHistoryErrorEl.textContent = aiAnalysis.message; 
                 aiIntRateCommentEl.textContent = '-';
                 aiDefaultRateCommentEl.textContent = '-';
                 aiSusScoreCommentEl.textContent = '-';
                 aiOverallHistorySummaryEl.textContent = '-';
            } else {
                aiIntRateCommentEl.textContent = aiAnalysis.interest_rate_comment || 'No comment available.';
                aiDefaultRateCommentEl.textContent = aiAnalysis.default_rate_comment || 'No comment available.';
                aiSusScoreCommentEl.textContent = aiAnalysis.sus_score_comment || 'No comment available.';
                aiOverallHistorySummaryEl.textContent = aiAnalysis.overall_summary || 'No overall summary available.';
            }
        } else {
             aiHistoryErrorEl.textContent = 'AI analysis data unavailable.';
             aiIntRateCommentEl.textContent = '-';
             aiDefaultRateCommentEl.textContent = '-';
             aiSusScoreCommentEl.textContent = '-';
             aiOverallHistorySummaryEl.textContent = '-';
        }
        historyAnalysisCard.classList.remove('hidden');
    }

    // --- API CALL FUNCTIONS ---
    function getValidatedNumber(elementId, isFloat = true) {
        const element = document.getElementById(elementId);
        if (!element) {
             throw new Error(`Element with ID '${elementId}' not found.`); // Add check
        }
        const value = isFloat ? parseFloat(element.value) : parseInt(element.value);
        if (isNaN(value)) {
            const label = document.querySelector(`label[for='${elementId}']`)?.textContent || elementId;
            throw new Error(`Invalid input for "${label}". Please enter a valid number.`);
        }
        return value;
    }

    async function predictInterestRate() {
        const data = {
            operation_years: getValidatedNumber('operation_years'),
            revenue: getValidatedNumber('revenue'),
            loan_amt: getValidatedNumber('loan_amt'),
            team_exp: getValidatedNumber('team_exp'),
            cred_hist_len: getValidatedNumber('cred_hist_len'),
            default_hist: getValidatedNumber('default_hist', false),
            office_own: getValidatedNumber('office_own', false),
            repayment_status: 0 
        };
        return makeApiCall('/predict_int_rate', 'POST', data);
    }

    async function predictDefaultRate() {
        const officeOwnValue = document.getElementById('office_own').value;
        const homeOwnershipString = officeOwnValue === '1' ? 'Own Home' : 'Rent';
        
        const data = {
            'Annual Income': Math.min(getValidatedNumber('annual_income')/10,400000),
            'Credit Score': getValidatedNumber('credit_score'),
            'Years of Credit History': getValidatedNumber('years_credit_history'),
            'Current Loan Amount': Math.min(getValidatedNumber('current_loan_amount')/10,300000),
            'Home Ownership': homeOwnershipString,
            'Tax Liens': 0, 'Number of Open Accounts': 10, 'Number of Credit Problems': 0,
            'Months since last delinquent': 0, 'Bankruptcies': getValidatedNumber('default_hist'), 'Current Credit Balance': 5000,
            'Maximum Open Credit': 20000, 'Monthly Debt': 1000, 'Years in current job': '10+ years',
            'Purpose': 'debt consolidation', 'Term': 'Short Term'
        };
        return makeApiCall('/predict_default', 'POST', data);
    }
    
    async function predictSustainabilityScore() {
        const data = {
            energy_efficiency: getValidatedNumber('energy_ef'),
            carbon_intensity: getValidatedNumber('carbon_int'),
            water_usage: getValidatedNumber('water_usg')
        };
        return makeApiCall('/sustainability_prediction', 'POST', data);
    }

    async function fetchAiAnalysis(intRate, defaultRate, susScore) {
        if (typeof intRate !== 'number' || typeof defaultRate !== 'number' || typeof susScore !== 'number') {
            updateAiAnalysisUI({ error: "Cannot generate analysis, one or more predictions failed." });
            return;
        }

        const data = {
            int_rate: intRate,
            default_rate: defaultRate,
            sus_score: susScore,
            notes: "Client-side calculation."
        };
        
        const analysisData = await makeApiCall('/company_summary', 'POST', data);
        
        if (analysisData.esgatescore !== undefined && typeof analysisData.esgatescore === 'number') {
            if (esgateScoreResultEl) { 
                esgateScoreResultEl.textContent = `${analysisData.esgatescore.toFixed(2)} / 100`;
            }
        } //else if (esgateScoreResultEl =='None' | esgateScoreResultEl == 'none') { 
            // esgateScoreResultEl.textContent = 'Error';
        // }

        if (analysisData.error) { 
            updateAiAnalysisUI({ error: `AI summary failed: ${analysisData.error}` });
        } else if (analysisData.mistral_summary) {
            updateAiAnalysisUI(analysisData.mistral_summary); 
        } else {
            updateAiAnalysisUI({ error: "AI summary response was malformed." });
        }
    }

    /**
    * helper function for making authenticated API calls.
    * This was also likely missing from user-js.js
    */
    async function makeApiCall(endpoint, method = 'GET', body = null) {
        const token = localStorage.getItem('authToken');
        if (!token) {
            alert("Session expired. Please log in again.");
            window.location.href = '/login_page'; // Point to login page
            throw new Error("Authentication token not found.");
        }

        const options = {
            method,
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(endpoint, options);
        if (!response.ok) {
            // Check for 401 Unauthorized specifically
            if (response.status === 401) {
                 localStorage.removeItem('authToken');
                 alert('Your session has expired. Please log in again.');
                 window.location.href = '/login_page'; // Redirect to login
            }
            const errorData = await response.json();
            throw new Error(errorData.message || errorData.error || `Request failed with status ${response.status}`);
        }
        return response.json();
    }
    
    // --- UI UPDATE & CHARTING FUNCTIONS ---
    
    function showLoading(isLoading) {
        if (loadingSpinner) loadingSpinner.classList.toggle('hidden', !isLoading);
        if (calculateButton) calculateButton.disabled = isLoading;
    }

    function updateResultsUI(intRate, defaultRate, susScore) {
        if (intRateResultEl && typeof intRate === 'number' && isFinite(intRate)) {
            intRateResultEl.textContent = `${(intRate * 100).toFixed(2)} %`;
        }
        if (defaultRateResultEl && typeof defaultRate === 'number' && isFinite(defaultRate)) {
            defaultRateResultEl.textContent = `${(defaultRate * 100).toFixed(2)} %`;
        }
        if (susScoreResultEl && typeof susScore === 'number' && isFinite(susScore)) {
            susScoreResultEl.textContent = `${susScore.toFixed(2)} / 10`;
        }
    }

    function updateAiAnalysisUI(analysis) {
        if (!aiAnalysisCard) return; // Exit if elements not on page
        if (analysis.error) {
            aiSummaryEl.textContent = `Error getting AI analysis: ${analysis.error}`;
            aiStrengthsEl.innerHTML = '<li>-</li>';
            aiWeaknessesEl.innerHTML = '<li>-</li>';
            aiRecommendationsEl.innerHTML = '<li>-</li>';
            aiAnalysisCard.classList.remove('hidden');
            return;
        }
        aiSummaryEl.textContent = analysis.summary || 'No summary available.';
        aiStrengthsEl.innerHTML = (analysis.strengths && analysis.strengths.length > 0) ? analysis.strengths.map(item => `<li>${item}</li>`).join('') : '<li>-</li>';
        aiWeaknessesEl.innerHTML = (analysis.weaknesses && analysis.weaknesses.length > 0) ? analysis.weaknesses.map(item => `<li>${item}</li>`).join('') : '<li>-</li>';
        aiRecommendationsEl.innerHTML = (analysis.recommendations && analysis.recommendations.length > 0) ? analysis.recommendations.map(item => `<li>${item}</li>`).join('') : '<li>-</li>';
        aiAnalysisCard.classList.remove('hidden');
    }

    function renderHistoryCharts(predictions) {
        const labels = predictions.map(p => new Date(p.created_at).toLocaleString());
        
        const intRateData = predictions.map(p => 
            p.int_rate !== null ? parseFloat((p.int_rate * 100).toFixed(2)) : null
        );
        const defaultRateData = predictions.map(p => 
            p.default_rate !== null ? parseFloat((p.default_rate * 100).toFixed(2)) : null
        );
        const susScoreData = predictions.map(p => 
            p.sus_score !== null ? parseFloat(p.sus_score.toFixed(2)) : null 
        ); 

        console.log("Labels for charts:", labels);
        console.log("Interest Rate Data for chart:", intRateData);
        console.log("Default Rate Data for chart:", defaultRateData);
        console.log("Sustainability Score Data for chart:", susScoreData);

        const intRateCtx = document.getElementById('interestRateChart')?.getContext('2d');
        if (intRateCtx) {
            if (interestRateChart) interestRateChart.destroy();
            interestRateChart = new Chart(intRateCtx, createChartConfig('Interest Rate (%)', labels, intRateData, 'rgba(13, 71, 161, 0.6)'));
        }
        
        const defaultRateCtx = document.getElementById('defaultRateChart')?.getContext('2d');
        if (defaultRateCtx) {
            if (defaultRateChart) defaultRateChart.destroy();
            defaultRateChart = new Chart(defaultRateCtx, createChartConfig('Default Probability (%)', labels, defaultRateData, 'rgba(211, 47, 47, 0.6)')); 
        }
        
        const susScoreCtx = document.getElementById('sustainabilityScoreChart')?.getContext('2d');
        if (susScoreCtx) {
            if (sustainabilityScoreChart) sustainabilityScoreChart.destroy();
            sustainabilityScoreChart = new Chart(susScoreCtx, createChartConfig('Sustainability Score (/10)', labels, susScoreData, 'rgba(76, 175, 80, 0.6)'));
        }
    }

    function createChartConfig(label, labels, data, color) {
        return {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    borderColor: color,
                    backgroundColor: color.replace('0.6', '0.2'),
                    fill: true,
                    tension: 0.3,
                    spanGaps: true 
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        };
    }

    // --- Add robust checks for elements not present on every page ---
    // (This helps prevent errors when this script runs on `home-page.html`
    // which doesn't have all the analysis elements)
    function initializePageElements() {
        if (!historyAnalysisCard) console.warn("History analysis card not found on this page.");
        if (!calculateButton) console.warn("Calculate button not found on this page.");
    }
    
    initializePageElements(); 

});
