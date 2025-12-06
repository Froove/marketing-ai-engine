// Smooth scroll
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

// API endpoint (à adapter selon votre configuration)
const API_URL = 'http://localhost:8000/generate-script';

// Form submission
document.getElementById('script-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const generateBtn = document.getElementById('generate-btn');
    const btnText = generateBtn.querySelector('.btn-text');
    const btnLoader = generateBtn.querySelector('.btn-loader');
    const outputContainer = document.getElementById('output-container');
    
    // Disable button and show loading
    generateBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    
    // Get form values
    const params = {
        brand: document.getElementById('brand').value,
        platform: document.getElementById('platform').value,
        audience: document.getElementById('audience').value,
        tone: document.getElementById('tone').value,
        angle_main: document.getElementById('angle_main').value || undefined,
    };
    
    // Remove undefined values
    Object.keys(params).forEach(key => params[key] === undefined && delete params[key]);
    
    try {
        // Call API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display result
        displayResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        outputContainer.innerHTML = `
            <div class="output-result" style="color: #ef4444;">
                <h4>Erreur</h4>
                <p>Impossible de générer le script. Vérifiez que l'API est démarrée sur ${API_URL}</p>
                <p style="font-size: 12px; margin-top: 8px;">${error.message}</p>
            </div>
        `;
    } finally {
        // Re-enable button
        generateBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function displayResult(data) {
    const outputContainer = document.getElementById('output-container');
    
    // Extract the first variant if available
    const variant = data.variants && data.variants[0] ? data.variants[0] : data;
    
    let html = '<div class="output-result">';
    
    // Hook
    if (variant.hook) {
        html += `<div class="hook">${escapeHtml(variant.hook)}</div>`;
    }
    
    // Script
    if (variant.script && Array.isArray(variant.script)) {
        html += '<div class="script">';
        variant.script.forEach(item => {
            html += `
                <div class="script-item">
                    <div class="timing">${item.timing_sec || 'N/A'}</div>
                    <div class="text">${escapeHtml(item.text)}</div>
                    ${item.visual ? `<div class="visual" style="font-size: 12px; color: #6b7280; margin-top: 4px;">📹 ${escapeHtml(item.visual)}</div>` : ''}
                </div>
            `;
        });
        html += '</div>';
    }
    
    // CTA
    if (variant.cta) {
        html += `<div class="cta">${escapeHtml(variant.cta)}</div>`;
    }
    
    // Scores (if available)
    if (variant.scores) {
        html += '<div style="margin-top: 20px; padding: 16px; background: #f9fafb; border-radius: 8px;">';
        html += '<strong>Scores:</strong><br>';
        Object.entries(variant.scores).forEach(([key, value]) => {
            html += `<span style="font-size: 14px; color: #6b7280;">${key}: ${value}</span><br>`;
        });
        html += '</div>';
    }
    
    // Raw JSON (collapsible)
    html += `
        <details style="margin-top: 20px;">
            <summary style="cursor: pointer; color: #6b7280; font-size: 14px;">Voir le JSON complet</summary>
            <pre style="margin-top: 12px;">${JSON.stringify(data, null, 2)}</pre>
        </details>
    `;
    
    html += '</div>';
    
    outputContainer.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

