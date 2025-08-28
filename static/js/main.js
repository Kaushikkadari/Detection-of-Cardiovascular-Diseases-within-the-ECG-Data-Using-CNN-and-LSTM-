/**
 * CardioDetect - ECG Cardiovascular Disease Detection
 * Main JavaScript File
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload preview
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                const fileInfo = document.createElement('div');
                fileInfo.className = 'alert alert-info mt-2';
                fileInfo.innerHTML = `<i class="fas fa-file-medical me-2"></i>Selected file: <strong>${fileName}</strong>`;
                
                // Remove any existing file info
                const existingInfo = document.querySelector('.file-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                fileInfo.classList.add('file-info');
                fileInput.parentNode.appendChild(fileInfo);
            }
        });
    }

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            } else {
                // Show loading indicator if form is valid
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
                    submitBtn.disabled = true;
                    
                    // Store original text for restoration if needed
                    submitBtn.dataset.originalText = originalText;
                }
            }
            
            form.classList.add('was-validated');
        }, false);
    });

    // ECG Plot Zooming
    const ecgPlot = document.querySelector('.ecg-plot-container img');
    if (ecgPlot) {
        ecgPlot.addEventListener('click', function() {
            this.classList.toggle('zoomed');
            if (this.classList.contains('zoomed')) {
                this.style.cursor = 'zoom-out';
                this.style.maxWidth = '150%';
                this.style.transition = 'max-width 0.3s ease';
            } else {
                this.style.cursor = 'zoom-in';
                this.style.maxWidth = '100%';
            }
        });
        
        // Add initial cursor style
        ecgPlot.style.cursor = 'zoom-in';
    }

    // Model training progress simulation
    const trainButton = document.querySelector('#train button[type="submit"]');
    if (trainButton) {
        trainButton.addEventListener('click', function(e) {
            // This is just a simulation for the demo
            // In a real app, this would be handled by a WebSocket or AJAX
            if (document.querySelector('#train form').checkValidity()) {
                e.preventDefault();
                
                // Create progress container if it doesn't exist
                let progressContainer = document.querySelector('.training-progress');
                if (!progressContainer) {
                    progressContainer = document.createElement('div');
                    progressContainer.className = 'training-progress mt-4';
                    document.querySelector('#train').appendChild(progressContainer);
                }
                
                // Show training progress
                progressContainer.innerHTML = `
                    <div class="card">
                        <div class="card-header bg-light">
                            <h5 class="card-title h6 mb-0">Training Progress</h5>
                        </div>
                        <div class="card-body">
                            <div class="progress mb-3">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" style="width: 0%" 
                                     aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                            </div>
                            <div class="training-log small">
                                <div class="text-muted">Initializing training environment...</div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Simulate training progress
                simulateTrainingProgress();
            }
        });
    }

    // Function to simulate training progress
    function simulateTrainingProgress() {
        const progressBar = document.querySelector('.training-progress .progress-bar');
        const trainingLog = document.querySelector('.training-log');
        let progress = 0;
        
        const epochs = parseInt(document.querySelector('#epochs').value) || 50;
        const interval = 100; // Update every 100ms
        const totalTime = 10000; // Total simulation time: 10 seconds
        const incrementPerStep = 100 / (totalTime / interval);
        
        const logMessages = [
            'Loading and preprocessing training data...',
            'Building CNN-LSTM model architecture...',
            'Compiling model with Adam optimizer...',
            'Starting training process...'
        ];
        
        // Add initial log messages
        let logIndex = 0;
        const logInterval = setInterval(() => {
            if (logIndex < logMessages.length) {
                trainingLog.innerHTML += `<div class="text-muted">${logMessages[logIndex]}</div>`;
                trainingLog.scrollTop = trainingLog.scrollHeight;
                logIndex++;
            } else {
                clearInterval(logInterval);
                
                // Start epoch updates
                let currentEpoch = 1;
                const epochInterval = setInterval(() => {
                    if (currentEpoch <= epochs && progress < 95) {
                        const loss = (1 - (currentEpoch / epochs)) * 0.5;
                        const accuracy = 0.5 + (currentEpoch / epochs) * 0.45;
                        
                        trainingLog.innerHTML += `<div>Epoch ${currentEpoch}/${epochs} - loss: ${loss.toFixed(4)} - accuracy: ${accuracy.toFixed(4)}</div>`;
                        trainingLog.scrollTop = trainingLog.scrollHeight;
                        currentEpoch++;
                    } else {
                        clearInterval(epochInterval);
                    }
                }, totalTime / epochs);
            }
        }, 500);
        
        // Update progress bar
        const progressInterval = setInterval(() => {
            progress += incrementPerStep;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
                
                // Training complete
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${Math.round(progress)}%`;
                progressBar.classList.remove('progress-bar-animated');
                
                trainingLog.innerHTML += `
                    <div class="text-success fw-bold">Training complete!</div>
                    <div>Model saved as cnn_lstm_model.h5</div>
                    <div>Validation accuracy: 94.2%</div>
                `;
                trainingLog.scrollTop = trainingLog.scrollHeight;
                
                // Reset button
                const trainButton = document.querySelector('#train button[type="submit"]');
                trainButton.innerHTML = trainButton.dataset.originalText;
                trainButton.disabled = false;
                
                // Add success message
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-success mt-3';
                alertDiv.innerHTML = '<i class="fas fa-check-circle me-2"></i>Model training completed successfully!';
                document.querySelector('.training-progress').appendChild(alertDiv);
            } else {
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${Math.round(progress)}%`;
            }
        }, interval);
    }
}); 