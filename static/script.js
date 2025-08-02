document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('modelSelect');
    const commonFields = document.getElementById('commonFields');
    const predictBtn = document.getElementById('predictBtn');
    const form = document.getElementById('predictionForm');
    const result = document.getElementById('result');
    const resultText = document.getElementById('resultText');
    const modelInfo = document.getElementById('modelInfo');
    const buttonText = document.getElementById('buttonText');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Model-specific field groups
    const fieldGroups = {
        'model1': document.getElementById('model1Fields'),
        'model2': document.getElementById('model2Fields'),
        'model3': document.getElementById('model3Fields'),
        'model4': document.getElementById('model4Fields')
    };

    // Model descriptions
    const modelDescriptions = {
        'model1': 'Random Forest Regressor using 10 features including derived engagement metrics to predict view count.',
        'model2': 'Advanced Random Forest Regressor with 26 features including comprehensive engagement analysis to predict subscriber count.',
        'model3': 'Random Forest Classifier using 8 core features to classify video popularity into Low, Medium, or High categories.',
        'model4': 'Random Forest Classifier with 14 features including engagement metrics to determine trending success probability.'
    };

    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        
        // Hide all field groups
        Object.values(fieldGroups).forEach(group => {
            if (group) {
                group.style.display = 'none';
            }
        });
        
        // Hide result and reset form
        result.style.display = 'none';
        form.reset();
        modelSelect.value = selectedModel; // Keep the selection
        
        if (selectedModel) {
            // Show common fields with animation
            commonFields.style.display = 'block';
            commonFields.style.animation = 'slideIn 0.5s ease-out';
            
            // Show model-specific fields
            if (fieldGroups[selectedModel]) {
                fieldGroups[selectedModel].style.display = 'block';
                fieldGroups[selectedModel].style.animation = 'slideIn 0.5s ease-out 0.2s both';
            }
            
            // Show predict button
            predictBtn.style.display = 'block';
            predictBtn.style.animation = 'slideIn 0.5s ease-out 0.4s both';
        } else {
            // Hide all fields and button
            commonFields.style.display = 'none';
            predictBtn.style.display = 'none';
        }
    });

    // Add input validation and user feedback
    const inputs = form.querySelectorAll('input[required], select[required]');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateField(this);
        });
        
        input.addEventListener('input', function() {
            if (this.classList.contains('error')) {
                validateField(this);
            }
        });
    });

    function validateField(field) {
        const isValid = field.checkValidity();
        
        if (isValid) {
            field.classList.remove('error');
            field.style.borderColor = '#4caf50';
        } else {
            field.classList.add('error');
            field.style.borderColor = '#f44336';
        }
        
        // Reset border color after animation
        setTimeout(() => {
            if (isValid) {
                field.style.borderColor = '';
            }
        }, 2000);
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedModel = modelSelect.value;
        if (!selectedModel) return;

        // Show loading state
        buttonText.style.display = 'none';
        loadingSpinner.style.display = 'inline';
        predictBtn.disabled = true;
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        // Convert form data to object with proper types
        for (let [key, value] of formData.entries()) {
            if (key === 'region' || key === 'title_sentiment') {
                data[key] = value.toLowerCase();
            } else if (key !== 'model' && value !== '') {
                data[key] = parseFloat(value) || 0;
            }
        }
        
        // Handle model-specific view_count field names
        const viewCountFields = {
            'model1': 'viewCount1',
            'model2': 'viewCount2', 
            'model3': 'viewCount3',
            'model4': 'viewCount4'
        };
        
        if (viewCountFields[selectedModel] && data.view_count === undefined) {
            const viewCountField = document.getElementById(viewCountFields[selectedModel]);
            if (viewCountField && viewCountField.value !== '') {
                data.view_count = parseFloat(viewCountField.value) || 0;
            }
        }
        
        // Handle comment_count for models that need it
        const commentCountFields = {
            'model2': 'commentCount2',
            'model3': 'commentCount3',
            'model4': 'commentCount4'
        };
        
        if (commentCountFields[selectedModel] && data.comment_count === undefined) {
            const commentField = document.getElementById(commentCountFields[selectedModel]);
            if (commentField && commentField.value !== '') {
                data.comment_count = parseFloat(commentField.value) || 0;
            }
        }
        
        // Handle like_view_ratio_percent for models that need it
        const ratioFields = {
            'model1': 'likeViewRatio1',
            'model2': 'likeViewRatio2'
        };
        
        if (ratioFields[selectedModel] && data.like_view_ratio_percent === undefined) {
            const ratioField = document.getElementById(ratioFields[selectedModel]);
            if (ratioField && ratioField.value !== '') {
                data.like_view_ratio_percent = parseFloat(ratioField.value) || 0;
            }
        }

        // Validate required fields
        const requiredFields = ['video_age_days', 'title_length_words', 'title_sentiment', 'category_id', 'like_count', 'region', 'subscriber_count'];
        const missingFields = requiredFields.filter(field => data[field] === undefined || data[field] === '');
        
        if (missingFields.length > 0) {
            showError(`Please fill in all required fields: ${missingFields.join(', ')}`);
            resetButton();
            return;
        }
        
        // Make prediction request
        fetch(`/predict?model=${selectedModel}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(responseData => {
            resetButton();
            
            if (responseData.error) {
                showError(`Prediction Error: ${responseData.error}`);
            } else {
                showResult(responseData.prediction, selectedModel);
            }
        })
        .catch(error => {
            resetButton();
            showError(`Network Error: ${error.message}`);
            console.error('Prediction error:', error);
        });
    });

    function resetButton() {
        buttonText.style.display = 'inline';
        loadingSpinner.style.display = 'none';
        predictBtn.disabled = false;
    }

    function showResult(prediction, modelType) {
        resultText.textContent = prediction;
        resultText.className = 'success';
        
        // Add model information
        if (modelDescriptions[modelType]) {
            modelInfo.innerHTML = `<strong>Model Used:</strong> ${modelDescriptions[modelType]}`;
            modelInfo.style.display = 'block';
        }
        
        result.style.display = 'block';
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Add celebration animation for successful predictions
        if (prediction.includes('Views:') || prediction.includes('Subscribers:') || prediction.includes('successful') || prediction.includes('high')) {
            addCelebrationEffect();
        }
    }

    function showError(errorMessage) {
        resultText.textContent = errorMessage;
        resultText.className = 'error';
        
        // Style the result box for errors
        result.style.background = 'linear-gradient(135deg, #ffebee, #fce4ec)';
        result.style.borderColor = '#f44336';
        
        modelInfo.style.display = 'none';
        result.style.display = 'block';
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Reset error styling after 5 seconds
        setTimeout(() => {
            result.style.background = '';
            result.style.borderColor = '';
        }, 5000);
    }

    function addCelebrationEffect() {
        // Add sparkle effect
        const sparkles = ['✨', '🎉', '⭐', '💫', '🌟'];
        
        for (let i = 0; i < 6; i++) {
            setTimeout(() => {
                const sparkle = document.createElement('div');
                sparkle.textContent = sparkles[Math.floor(Math.random() * sparkles.length)];
                sparkle.style.cssText = `
                    position: fixed;
                    font-size: 1.5rem;
                    pointer-events: none;
                    z-index: 1000;
                    animation: sparkle 2s ease-out forwards;
                    left: ${Math.random() * window.innerWidth}px;
                    top: ${Math.random() * window.innerHeight}px;
                `;
                
                document.body.appendChild(sparkle);
                
                setTimeout(() => {
                    sparkle.remove();
                }, 2000);
            }, i * 200);
        }
    }

    // Add sparkle animation CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes sparkle {
            0% {
                opacity: 0;
                transform: scale(0) rotate(0deg);
            }
            50% {
                opacity: 1;
                transform: scale(1) rotate(180deg);
            }
            100% {
                opacity: 0;
                transform: scale(0.5) rotate(360deg);
            }
        }
        
        .error {
            color: #f44336 !important;
            background: rgba(244, 67, 54, 0.1) !important;
            border-left: 4px solid #f44336 !important;
        }
        
        .success {
            color: #1b5e20 !important;
            background: rgba(255, 255, 255, 0.7) !important;
            border-left: 4px solid #4caf50 !important;
        }
        
        input.error, select.error {
            border-color: #f44336 !important;
            box-shadow: 0 0 0 4px rgba(244, 67, 54, 0.1) !important;
        }
    `;
    document.head.appendChild(style);

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (predictBtn.style.display !== 'none' && !predictBtn.disabled) {
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape') {
            if (confirm('Clear the form and start over?')) {
                form.reset();
                modelSelect.dispatchEvent(new Event('change'));
                result.style.display = 'none';
            }
        }
    });

    // Add tooltips for better UX
    const helpTexts = {
        'video_age_days': 'Enter the number of days since the video was published. Newer videos typically have different performance patterns.',
        'title_length_words': 'Count the words in your video title. Optimal length varies by content type and audience.',
        'category_id': 'Select the primary category that best describes your video content.',
        'like_count': 'Current number of likes. This helps calculate engagement ratios.',
        'subscriber_count': 'Total channel subscribers. Larger channels often have different trending patterns.'
    };

    Object.keys(helpTexts).forEach(fieldName => {
        const field = document.querySelector(`[name="${fieldName}"]`);
        if (field) {
            field.addEventListener('focus', function() {
                showTooltip(this, helpTexts[fieldName]);
            });
            
            field.addEventListener('blur', function() {
                hideTooltip();
            });
        }
    });

    let currentTooltip = null;

    function showTooltip(element, text) {
        hideTooltip();
        
        const tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            max-width: 250px;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transform: translateY(-10px);
            transition: all 0.3s ease;
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
        
        setTimeout(() => {
            tooltip.style.opacity = '1';
            tooltip.style.transform = 'translateY(0)';
        }, 100);
        
        currentTooltip = tooltip;
    }

    function hideTooltip() {
        if (currentTooltip) {
            currentTooltip.remove();
            currentTooltip = null;
        }
    }
});document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('modelSelect');
    const commonFields = document.getElementById('commonFields');
    const predictBtn = document.getElementById('predictBtn');
    const form = document.getElementById('predictionForm');
    const result = document.getElementById('result');
    const resultText = document.getElementById('resultText');

    // Model-specific field groups
    const fieldGroups = {
        'model1': document.getElementById('model1Fields'),
        'model2': document.getElementById('model2Fields'),
        'model3': document.getElementById('model3Fields'),
        'model4': document.getElementById('model4Fields')
    };

    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        
        // Hide all field groups
        Object.values(fieldGroups).forEach(group => {
            group.style.display = 'none';
        });
        
        // Hide result
        result.style.display = 'none';
        
        if (selectedModel) {
            // Show common fields
            commonFields.style.display = 'block';
            
            // Show model-specific fields
            if (fieldGroups[selectedModel]) {
                fieldGroups[selectedModel].style.display = 'block';
            }
            
            // Show predict button
            predictBtn.style.display = 'block';
        } else {
            // Hide all fields and button
            commonFields.style.display = 'none';
            predictBtn.style.display = 'none';
        }
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedModel = modelSelect.value;
        if (!selectedModel) return;
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        // Convert form data to object with proper types
        for (let [key, value] of formData.entries()) {
            if (key === 'region' || key === 'title_sentiment') {
                data[key] = value.toLowerCase();
            } else if (key !== 'model') {
                data[key] = parseFloat(value) || 0;
            }
        }
        
        // Handle model-specific view_count field names
        if (selectedModel === 'model1' && data.view_count === undefined) {
            const viewCountField = document.getElementById('viewCount1');
            data.view_count = parseFloat(viewCountField.value) || 0;
        } else if (selectedModel === 'model2' && data.view_count === undefined) {
            const viewCountField = document.getElementById('viewCount2');
            data.view_count = parseFloat(viewCountField.value) || 0;
        } else if (selectedModel === 'model3' && data.view_count === undefined) {
            const viewCountField = document.getElementById('viewCount3');
            data.view_count = parseFloat(viewCountField.value) || 0;
        } else if (selectedModel === 'model4' && data.view_count === undefined) {
            const viewCountField = document.getElementById('viewCount4');
            data.view_count = parseFloat(viewCountField.value) || 0;
        }
        
        // Handle comment_count for models that need it
        if ((selectedModel === 'model2' || selectedModel === 'model3' || selectedModel === 'model4') && data.comment_count === undefined) {
            const commentField = document.getElementById(`commentCount${selectedModel.slice(-1)}`);
            data.comment_count = parseFloat(commentField.value) || 0;
        }
        
        // Handle like_view_ratio_percent for models that need it
        if ((selectedModel === 'model1' || selectedModel === 'model2') && data.like_view_ratio_percent === undefined) {
            const ratioField = document.getElementById(`likeViewRatio${selectedModel.slice(-1)}`);
            data.like_view_ratio_percent = parseFloat(ratioField.value) || 0;
        }
        
        // Make prediction request
        fetch(`/predict?model=${selectedModel}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultText.textContent = `Error: ${data.error}`;
            } else {
                resultText.textContent = data.prediction;
            }
            result.style.display = 'block';
        })
        .catch(error => {
            resultText.textContent = `Error: ${error.message}`;
            result.style.display = 'block';
        });
    });
});