        document.addEventListener('DOMContentLoaded', function () {
            
            // --- Mobile Menu Toggle ---
            const menuButton = document.getElementById('mobile-menu-button');
            const mobileMenu = document.getElementById('mobile-menu');
            const iconOpen = document.getElementById('menu-icon-open');
            const iconClose = document.getElementById('menu-icon-close');

            if (menuButton && mobileMenu && iconOpen && iconClose) {
                menuButton.addEventListener('click', function () {
                    mobileMenu.classList.toggle('hidden');
                    iconOpen.classList.toggle('hidden');
                    iconClose.classList.toggle('hidden');
                    const isExpanded = menuButton.getAttribute('aria-expanded') === 'true';
                    menuButton.setAttribute('aria-expanded', !isExpanded);
                });
            }

            // --- Protected Link Navigation (Delegated) ---
            document.body.addEventListener('click', function(event) {
                const protectedLink = event.target.closest('a.protected-nav-link'); 
                if (protectedLink) {
                    event.preventDefault(); 
                    const url = protectedLink.getAttribute('href');
                    if (url && url !== '#') {
                         loadProtectedPage(url); 
                    }
                }
            });

            // --- Accordion Toggle ---
            const accordionHeader = document.getElementById('accordion-header-methodology');
            const accordionContent = document.getElementById('accordion-content-methodology');
            
            if (accordionHeader && accordionContent) {
                accordionHeader.addEventListener('click', () => {
                    const icon = accordionHeader.querySelector('.accordion-icon');
                    accordionHeader.classList.toggle('active');
                    if (icon) icon.classList.toggle('rotate-180');

                    if (accordionContent.style.maxHeight) {
                        // Collapse
                        accordionContent.style.maxHeight = null;
                        setTimeout(() => { 
                            accordionContent.style.paddingTop = "0"; 
                            accordionContent.style.paddingBottom = "0"; 
                        }, 300); // Match transition duration
                    } else {
                        // Expand
                        accordionContent.style.paddingTop = "0.5rem"; // pt-2
                        accordionContent.style.paddingBottom = "1.5rem"; // pb-6
                        accordionContent.style.maxHeight = accordionContent.scrollHeight + "px";
                    } 
                });
            }
            
            // --- Table Row Expansion ---
            const expandableRows = document.querySelectorAll('tr.expandable-row');
            expandableRows.forEach(row => {
                row.addEventListener('click', () => {
                    const targetId = row.getAttribute('data-target-id');
                    const targetPanel = document.getElementById(targetId);

                    if (targetPanel) {
                        const isHidden = targetPanel.classList.toggle('hidden');
                        row.setAttribute('aria-expanded', !isHidden);
                        
                        // --- Chart Creation Logic (Only runs when expanding) ---
                        if (!isHidden && targetId === 'details-1') {
                            const detailCtx = document.getElementById('detail-chart-1')?.getContext('2d');
                            
                            if (detailCtx) {
                                // 🔑 FIX: Destroy the previous chart instance before creating a new one
                                if (detailChart1Instance) {
                                    detailChart1Instance.destroy();
                                }
                                
                                // 🔑 FIX: Create the new chart and save the instance to the global variable
                                detailChart1Instance = new Chart(detailCtx, {
                                    type: 'radar',
                                    data: {
                                        labels: ['Environmental', 'Social', 'Governance', 'Ethics', 'Transparency'],
                                        datasets: [
                                            {
                                                label: 'Agritech Solutions',
                                                data: [85, 75, 90, 80, 78],
                                                backgroundColor: 'rgba(30, 58, 138, 0.2)',
                                                borderColor: 'rgba(30, 58, 138, 1)',
                                                borderWidth: 2
                                            },
                                            {
                                                label: 'Sector Average',
                                                data: [60, 65, 70, 68, 70],
                                                backgroundColor: 'rgba(209, 213, 219, 0.2)',
                                                borderColor: 'rgba(107, 114, 128, 1)',
                                                borderWidth: 1,
                                                borderDash: [5, 5]
                                            }
                                        ]
                                    },
                                    options: {
                                        responsive: true, maintainAspectRatio: false,
                                        scales: { r: { beginAtZero: true, max: 100, ticks: { display: false, stepSize: 25 } } },
                                        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }
                                    }
                                });
                            }
                        }
                    }
                });
            });
            // --- loadProtectedPage Function ---
            async function loadProtectedPage(url) {
                 const token = localStorage.getItem('authToken');
                 if (!token) {
                     console.log("No auth token found, redirecting to login.");
                     window.location.href = "{{ url_for('login_page') }}"; // Make sure 'login_page' is correct
                     return;
                 }
                 try {
                     const response = await fetch(url, {
                         method: 'GET',
                         headers: { 'Authorization': `Bearer ${token}` }
                     });
                     if (response.ok) {
                         const pageHtml = await response.text();
                         document.open(); 
                         document.write(pageHtml);
                         document.close();
                     } else {
                         console.log("Response not OK, removing token and redirecting.");
                         localStorage.removeItem('authToken');
                         alert('Session expired or invalid. Please log in again.');
                         window.location.href = "{{ url_for('login_page') }}";
                     }
                 } catch (error) {
                     console.error('Failed to load protected page:', error);
                 }
             }
             
             // --- Placeholder Chart.js Data ---
             // Placeholder for expanded row radar chart
            let detailChartInstance = null;
            const detailCtx = document.getElementById('detail-chart-1')?.getContext('2d');

             if (detailCtx) {
                 new Chart(detailCtx, {
                    type: 'radar',
                    data: {
                        labels: ['Environmental', 'Social', 'Governance', 'Ethics', 'Transparency'],
                        datasets: [
                            {
                                label: 'Agritech Solutions',
                                data: [85, 75, 90, 80, 78],
                                backgroundColor: 'rgba(30, 58, 138, 0.2)', // blue-800
                                borderColor: 'rgba(30, 58, 138, 1)',
                                borderWidth: 2
                            },
                            {
                                label: 'Sector Average',
                                data: [60, 65, 70, 68, 70],
                                backgroundColor: 'rgba(209, 213, 219, 0.2)', // gray-300
                                borderColor: 'rgba(107, 114, 128, 1)', // gray-500
                                borderWidth: 1,
                                borderDash: [5, 5]
                            }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: { r: { beginAtZero: true, max: 100, ticks: { display: false, stepSize: 25 } } },
                        plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }
                    }
                });
             }
             
             // Placeholder for comparison bar chart
             const compareCtx = document.getElementById('comparison-chart')?.getContext('2d');
             if (compareCtx) {
                 new Chart(compareCtx, {
                     type: 'bar',
                     data: {
                         labels: ['ESG Score', 'Default Risk', 'Compliance'],
                         datasets: [
                             {
                                 label: 'Agritech Solutions',
                                 data: [82, 10.5, 90],
                                 backgroundColor: 'rgba(30, 58, 138, 0.8)', // blue-800
                                 borderColor: 'rgba(30, 58, 138, 1)',
                                 borderWidth: 1
                             },
                             {
                                 label: 'Baku Fabrix',
                                 data: [42, 51.5, 30],
                                 backgroundColor: 'rgba(156, 163, 175, 0.8)', // gray-400
                                 borderColor: 'rgba(156, 163, 175, 1)',
                                 borderWidth: 1
                             }
                         ]
                     },
                     options: {
                         responsive: true, maintainAspectRatio: false,
                         scales: { y: { beginAtZero: true, max: 100 } },
                         plugins: { legend: { position: 'bottom' } }
                     }
                 });
             }




             

        }); 