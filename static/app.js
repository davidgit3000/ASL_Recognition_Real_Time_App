// Global variables
let camera = null;
let hands = null;
let isRunning = false;
let showLandmarks = false;
let sentence = [];
let lastPrediction = null;
let lastSpeakTime = 0;
let lastAddedSign = null;
let lastAddTime = 0;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let currentModelType = "random_forest"; // Current model: 'random_forest' or 'transformer'

// DOM elements
const videoElement = document.getElementById("videoElement");
const canvasElement = document.getElementById("canvasElement");
const canvasCtx = canvasElement.getContext("2d");
const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const toggleLandmarksButton = document.getElementById("toggleLandmarks");
const clearSentenceButton = document.getElementById("clearSentence");
const speakSentenceButton = document.getElementById("speakSentence");
const statusText = document.getElementById("statusText");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const sentenceDisplay = document.getElementById("sentenceDisplay");
const enableAudioCheckbox = document.getElementById("enableAudio");
const autoSpeakCheckbox = document.getElementById("autoSpeak");
const knownSignsElement = document.getElementById("knownSigns");
const fpsCounter = document.getElementById("fpsCounter");
const modelTypeElement = document.getElementById("modelType");
const bufferStatusElement = document.getElementById("bufferStatus");
const toggleModelBtn = document.getElementById("toggleModelBtn");
const resetBufferBtn = document.getElementById("resetBufferBtn");

// Initialize MediaPipe Hands
function initializeHands() {
  hands = new Hands({
    locateFile: (file) => {
      // https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.min.js
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
    },
  });

  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults(onResults);
}

// Process results from MediaPipe
async function onResults(results) {
  // Update FPS
  frameCount++;
  const now = Date.now();
  if (now - lastFpsUpdate > 1000) {
    fpsCounter.textContent = frameCount;
    frameCount = 0;
    lastFpsUpdate = now;
  }

  // Clear canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Draw video frame
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );

  // Draw landmarks if enabled
  if (showLandmarks && results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 2,
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: "#FF0000",
        lineWidth: 1,
        radius: 3,
      });
    }
  }

  canvasCtx.restore();

  // Send to backend for prediction
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    await getPrediction(results.multiHandLandmarks);
  } else {
    updatePrediction("No hand detected", 0);
  }
}

// Get prediction from backend
async function getPrediction(landmarks) {
  try {
    if (!landmarks || landmarks.length === 0) {
      updatePrediction("No hand detected", 0);
      return;
    }

    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ landmarks: landmarks }),
    });

    const data = await response.json();

    if (data.success) {
      updatePrediction(data.prediction, data.confidence, data.buffer_status);

      // Auto-add to sentence if high confidence (green)
      if (
        data.confidence >= 80 &&
        data.prediction !== "No hand detected" &&
        data.prediction !== "Unknown sign"
        // data.prediction !== lastAddedSign
      ) {
        const now = Date.now();
        // Add to sentence with 1.5 second cooldown to avoid duplicates
        if (now - lastAddTime > 2000) {
          sentence.push(data.prediction);
          updateSentenceDisplay();
          lastAddedSign = data.prediction;
          lastAddTime = now;

          // Auto-speak if enabled
          if (autoSpeakCheckbox.checked && enableAudioCheckbox.checked) {
            speakFast(data.prediction);
          }
        }
      }

      lastPrediction = data.prediction;
    }
  } catch (error) {
    console.error("Prediction error:", error);
  }
}

// Update prediction display
function updatePrediction(prediction, confidence, bufferStatus) {
  predictionText.textContent = prediction;
  confidenceText.textContent = `Confidence: ${confidence.toFixed(1)}%`;

  // Update buffer status for Transformer
  if (bufferStatus && bufferStatusElement) {
    bufferStatusElement.textContent = bufferStatus;
    bufferStatusElement.parentElement.classList.remove("hidden");
  } else if (bufferStatusElement) {
    bufferStatusElement.parentElement.classList.add("hidden");
  }

  // Update color based on confidence
  const predictionBox = document.getElementById("predictionBox");
  predictionText.className = "text-3xl font-bold mb-2 ";

  if (prediction === "No hand detected" || prediction === "Unknown sign") {
    predictionText.className += "confidence-unknown";
  } else if (confidence >= 80) {
    predictionText.className += "confidence-high";
  } else if (confidence >= 60) {
    predictionText.className += "confidence-medium";
  } else {
    predictionText.className += "confidence-low";
  }
}

// Start camera
async function startCamera() {
  try {
    statusText.textContent = "Starting camera...";

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
    });

    videoElement.srcObject = stream;

    // Wait for video to be ready
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        resolve();
      };
    });

    // Set canvas size to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    // Always reinitialize MediaPipe to fix restart issue
    initializeHands();

    // Start camera
    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (isRunning) {
          await hands.send({ image: videoElement });
        }
      },
      width: 1280,
      height: 720,
    });

    await camera.start();
    isRunning = true;

    statusText.textContent = "Camera active";
    startButton.disabled = true;
    stopButton.disabled = false;
  } catch (error) {
    console.error("Camera error:", error);
    statusText.textContent = "Camera error: " + error.message;
    alert("Could not access camera. Please check permissions.");
  }
}

// Stop camera
function stopCamera() {
  isRunning = false;

  if (camera) {
    camera.stop();
    camera = null;
  }

  if (videoElement.srcObject) {
    videoElement.srcObject.getTracks().forEach((track) => track.stop());
    videoElement.srcObject = null;
  }

  // Close MediaPipe Hands to free resources
  if (hands) {
    hands.close();
    hands = null;
  }

  // Clear canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  statusText.textContent = "Camera stopped";
  startButton.disabled = false;
  stopButton.disabled = true;
}

// Toggle landmarks
function toggleLandmarks() {
  showLandmarks = !showLandmarks;
  document.getElementById("landmarksText").textContent = showLandmarks
    ? "Hide Landmarks"
    : "Show Landmarks";
}

// Update sentence display
function updateSentenceDisplay() {
  if (sentence.length === 0) {
    sentenceDisplay.innerHTML =
      '<em class="text-gray-400">Your sentence will appear here...</em>';
    speakSentenceButton.disabled = true;
  } else {
    // Group consecutive single letters together, keep multi-letter words separate
    let displayParts = [];
    let currentLetters = [];

    for (let i = 0; i < sentence.length; i++) {
      const item = sentence[i];

      // Check if it's a single letter (A-Z, case insensitive)
      if (item.length === 1 && /^[a-zA-Z]$/.test(item)) {
        currentLetters.push(item.toLowerCase());
      } else {
        // It's a word, not a single letter
        if (currentLetters.length > 0) {
          displayParts.push(currentLetters.join(""));
          currentLetters = [];
        }
        displayParts.push(item);
      }
    }

    // Add any remaining letters
    if (currentLetters.length > 0) {
      displayParts.push(currentLetters.join(""));
    }

    const sentenceText = displayParts.join(" ");
    sentenceDisplay.innerHTML = `<span class="text-gray-800 font-medium">${sentenceText}</span>`;
    speakSentenceButton.disabled = false;
    console.log("Sentence updated:", sentenceText);
  }
}

// Clear sentence
function clearSentence() {
  sentence = [];
  updateSentenceDisplay();
}

// Fast speech for auto-speak (optimized for speed)
function speakFast(text) {
  if (!enableAudioCheckbox.checked) return;

  // Cancel any ongoing speech for instant response
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text.toLowerCase());
  utterance.rate = 1.2; // Faster rate for quick feedback
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  utterance.lang = "en-US";

  // Use cached voice for instant playback
  const voices = window.speechSynthesis.getVoices();
  if (voices.length > 0) {
    utterance.voice = voices[0];
  }

  // Speak immediately
  window.speechSynthesis.speak(utterance);
}

// Regular speech for sentence playback
function speak(text) {
  if (!enableAudioCheckbox.checked) return;

  // Cancel any ongoing speech
  window.speechSynthesis.cancel();

  // Convert to lowercase to prevent "capitalize" being spoken
  const utterance = new SpeechSynthesisUtterance(text.toLowerCase());
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  utterance.lang = "en-US";

  const voices = window.speechSynthesis.getVoices();
  if (voices.length > 0) {
    utterance.voice = voices[0];
  }

  window.speechSynthesis.speak(utterance);
}

// Speak sentence
function speakSentence() {
  if (sentence.length > 0) {
    // Group consecutive single letters together, keep multi-letter words separate
    let displayParts = [];
    let currentLetters = [];

    for (let i = 0; i < sentence.length; i++) {
      const item = sentence[i];

      // Check if it's a single letter (A-Z, case insensitive)
      if (item.length === 1 && /^[a-zA-Z]$/.test(item)) {
        currentLetters.push(item);
      } else {
        // It's a word, not a single letter
        if (currentLetters.length > 0) {
          displayParts.push(currentLetters.join(""));
          currentLetters = [];
        }
        displayParts.push(item);
      }
    }

    // Add any remaining letters
    if (currentLetters.length > 0) {
      displayParts.push(currentLetters.join(""));
    }

    speak(displayParts.join(" "));
  }
}

// Load model info
async function loadModelInfo() {
  try {
    const response = await fetch("/model_info");
    const data = await response.json();

    if (data.success) {
      knownSignsElement.textContent = data.num_signs;
      currentModelType = data.model_type;
      if (modelTypeElement) {
        modelTypeElement.textContent =
          data.model_type === "transformer" ? "Transformer" : "Random Forest";
      }

      // Show/hide buffer status based on model type
      if (bufferStatusElement) {
        if (data.model_type === "transformer") {
          bufferStatusElement.parentElement.classList.remove("hidden");
        } else {
          bufferStatusElement.parentElement.classList.add("hidden");
        }
      }
    }
  } catch (error) {
    console.error("Error loading model info:", error);
    knownSignsElement.textContent = "Error";
  }
}

// Toggle between models
async function toggleModel() {
  try {
    const newModelType =
      currentModelType === "random_forest" ? "transformer" : "random_forest";

    const response = await fetch("/toggle_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: newModelType }),
    });

    const data = await response.json();

    if (data.success) {
      currentModelType = data.model_type;
      if (modelTypeElement) {
        modelTypeElement.textContent =
          data.model_type === "transformer" ? "Transformer" : "Random Forest";
      }

      // Show/hide buffer status
      if (bufferStatusElement) {
        if (data.model_type === "transformer") {
          bufferStatusElement.parentElement.classList.remove("hidden");
        } else {
          bufferStatusElement.parentElement.classList.add("hidden");
        }
      }

      // Update button text
      if (toggleModelBtn) {
        toggleModelBtn.textContent =
          data.model_type === "transformer"
            ? "🔄 Switch to Random Forest"
            : "🔄 Switch to Transformer";
      }

      console.log(data.message);
    } else {
      alert("Error switching model: " + data.error);
    }
  } catch (error) {
    console.error("Error toggling model:", error);
    alert("Error switching model");
  }
}

// Reset buffer (Transformer only)
async function resetBuffer() {
  try {
    const response = await fetch("/reset_buffer", { method: "POST" });
    const data = await response.json();

    if (data.success) {
      console.log("Buffer reset");
      if (bufferStatusElement) {
        bufferStatusElement.textContent = "0/30";
      }
    }
  } catch (error) {
    console.error("Error resetting buffer:", error);
  }
}

// Preload voices for faster speech
function preloadVoices() {
  // Trigger voice loading
  window.speechSynthesis.getVoices();

  // Some browsers need this event
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }
}

// Event listeners
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
toggleLandmarksButton.addEventListener("click", toggleLandmarks);
clearSentenceButton.addEventListener("click", clearSentence);
speakSentenceButton.addEventListener("click", speakSentence);
if (toggleModelBtn) toggleModelBtn.addEventListener("click", toggleModel);
if (resetBufferBtn) resetBufferBtn.addEventListener("click", resetBuffer);

// Initialize on page load
window.addEventListener("load", () => {
  loadModelInfo();
  preloadVoices();
  statusText.textContent = "Ready to start";
});

// ============================================
// ADD NEW SIGN MODAL FUNCTIONALITY
// ============================================

// Modal elements
const addSignModal = document.getElementById("addSignModal");
const addNewSignBtn = document.getElementById("addNewSignBtn");
const closeModal = document.getElementById("closeModal");
const closeSuccessBtn = document.getElementById("closeSuccessBtn");

// Step content elements
const step1Content = document.getElementById("step1Content");
const step2Content = document.getElementById("step2Content");
const step3Content = document.getElementById("step3Content");
const step4Content = document.getElementById("step4Content");

// Step indicators
const step1Indicator = document.getElementById("step1Indicator");
const step2Indicator = document.getElementById("step2Indicator");
const step3Indicator = document.getElementById("step3Indicator");

// Form inputs
const signNameInput = document.getElementById("signNameInput");
const numImagesInput = document.getElementById("numImagesInput");
const startCollectionBtn = document.getElementById("startCollectionBtn");

// Collection elements
const collectionVideo = document.getElementById("collectionVideo");
const collectionCanvas = document.getElementById("collectionCanvas");
const collectionStatus = document.getElementById("collectionStatus");
const imageCounter = document.getElementById("imageCounter");
const imageTotal = document.getElementById("imageTotal");
const cancelCollectionBtn = document.getElementById("cancelCollectionBtn");
const finishCollectionBtn = document.getElementById("finishCollectionBtn");

// Training elements
const trainingStatus = document.getElementById("trainingStatus");
const trainingProgress = document.getElementById("trainingProgress");
const trainingProgressText = document.getElementById("trainingProgressText");

// Collection state
let collectionStream = null;
let collectedImages = [];
let targetImageCount = 100;
let isCollecting = false;
let currentSignName = "";

// Open modal
addNewSignBtn.addEventListener("click", () => {
  addSignModal.classList.remove("hidden");
  addSignModal.classList.add("flex");
  resetModal();

  // Show warning if Transformer is active
  const transformerWarning = document.getElementById("transformerWarning");
  if (transformerWarning) {
    if (currentModelType === "transformer") {
      transformerWarning.classList.remove("hidden");
    } else {
      transformerWarning.classList.add("hidden");
    }
  }
});

// Close modal
closeModal.addEventListener("click", closeModalHandler);
closeSuccessBtn.addEventListener("click", closeModalHandler);

function closeModalHandler() {
  addSignModal.classList.add("hidden");
  addSignModal.classList.remove("flex");
  if (collectionStream) {
    collectionStream.getTracks().forEach((track) => track.stop());
    collectionStream = null;
  }
  resetModal();
}

// Reset modal to step 1
function resetModal() {
  showStep(1);
  signNameInput.value = "";
  numImagesInput.value = "100";
  collectedImages = [];
  isCollecting = false;
  currentSignName = "";

  // Reset collection UI
  collectionStatus.textContent = "Press SPACE to start capturing";
  imageCounter.textContent = "0";
  imageTotal.textContent = "100";
  finishCollectionBtn.disabled = true;

  // Remove spacebar listener if it exists
  document.removeEventListener("keydown", handleSpacePress);
}

// Show specific step
function showStep(step) {
  // Hide all steps
  step1Content.classList.add("hidden");
  step2Content.classList.add("hidden");
  step3Content.classList.add("hidden");
  step4Content.classList.add("hidden");

  // Reset indicators
  step1Indicator.className =
    "w-10 h-10 mx-auto rounded-full bg-gray-300 text-white flex items-center justify-center font-bold";
  step2Indicator.className =
    "w-10 h-10 mx-auto rounded-full bg-gray-300 text-white flex items-center justify-center font-bold";
  step3Indicator.className =
    "w-10 h-10 mx-auto rounded-full bg-gray-300 text-white flex items-center justify-center font-bold";

  // Show current step
  if (step === 1) {
    step1Content.classList.remove("hidden");
    step1Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-blue-600 text-white flex items-center justify-center font-bold";
  } else if (step === 2) {
    step2Content.classList.remove("hidden");
    step1Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
    step2Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-blue-600 text-white flex items-center justify-center font-bold";
  } else if (step === 3) {
    step3Content.classList.remove("hidden");
    step1Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
    step2Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
    step3Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-blue-600 text-white flex items-center justify-center font-bold";
  } else if (step === 4) {
    step4Content.classList.remove("hidden");
    step1Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
    step2Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
    step3Indicator.className =
      "w-10 h-10 mx-auto rounded-full bg-green-600 text-white flex items-center justify-center font-bold";
  }
}

// Start collection
startCollectionBtn.addEventListener("click", async () => {
  const signName = signNameInput.value.trim().toLowerCase();
  const numImages = parseInt(numImagesInput.value);

  if (!signName) {
    alert("Please enter a sign name");
    return;
  }

  if (numImages < 50 || numImages > 500) {
    alert("Number of images must be between 50 and 500");
    return;
  }

  currentSignName = signName;
  targetImageCount = numImages;
  imageTotal.textContent = numImages;
  imageCounter.textContent = "0";
  collectedImages = [];

  // Start camera for collection
  try {
    collectionStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    collectionVideo.srcObject = collectionStream;
    showStep(2);

    // Listen for spacebar
    document.addEventListener("keydown", handleSpacePress);
  } catch (error) {
    alert("Could not access camera: " + error.message);
  }
});

// Handle spacebar press
function handleSpacePress(e) {
  if (
    e.code === "Space" &&
    !isCollecting &&
    collectedImages.length < targetImageCount
  ) {
    e.preventDefault();
    startCapturing();
  }
}

// Start capturing images
function startCapturing() {
  isCollecting = true;
  collectionStatus.textContent = "Capturing...";
  finishCollectionBtn.disabled = true;

  const captureInterval = setInterval(() => {
    if (collectedImages.length >= targetImageCount) {
      clearInterval(captureInterval);
      isCollecting = false;
      collectionStatus.textContent = "Collection complete!";
      finishCollectionBtn.disabled = false;
      document.removeEventListener("keydown", handleSpacePress);
      return;
    }

    // Capture frame
    const canvas = document.createElement("canvas");
    canvas.width = collectionVideo.videoWidth;
    canvas.height = collectionVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(collectionVideo, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL("image/jpeg", 0.9);
    collectedImages.push(imageData);
    imageCounter.textContent = collectedImages.length;
  }, 100); // Capture every 100ms
}

// Cancel collection
cancelCollectionBtn.addEventListener("click", () => {
  if (collectionStream) {
    collectionStream.getTracks().forEach((track) => track.stop());
    collectionStream = null;
  }
  document.removeEventListener("keydown", handleSpacePress);
  showStep(1);
});

// Finish collection and train
finishCollectionBtn.addEventListener("click", async () => {
  if (collectionStream) {
    collectionStream.getTracks().forEach((track) => track.stop());
    collectionStream = null;
  }
  document.removeEventListener("keydown", handleSpacePress);

  // Check if Transformer model is active
  if (currentModelType === "transformer") {
    // Show instructions for Transformer
    showTransformerInstructions(currentSignName);
    return;
  }

  showStep(3);

  try {
    // Send images to backend (Random Forest only)
    trainingStatus.textContent = "Uploading images...";
    trainingProgress.style.width = "10%";
    trainingProgressText.textContent = "10%";

    const response = await fetch("/add_new_sign", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sign_name: currentSignName,
        model_type: currentModelType,
        images: collectedImages,
      }),
    });

    const data = await response.json();

    if (data.success) {
      trainingProgress.style.width = "100%";
      trainingProgressText.textContent = "100%";
      trainingStatus.textContent = "Training complete!";

      setTimeout(() => {
        showStep(4);
        // Reload model info
        loadModelInfo();
      }, 1000);
    } else {
      alert("Error: " + data.error);
      showStep(1);
    }
  } catch (error) {
    alert("Error training model: " + error.message);
    showStep(1);
  }
});

// Show Transformer instructions
function showTransformerInstructions(signName) {
  const instructions = `
    <div class="bg-blue-50 border-2 border-blue-300 rounded-lg p-6">
      <h3 class="text-xl font-bold text-blue-800 mb-4">
        🤖 Transformer Model - CLI Required
      </h3>
      <p class="text-gray-700 mb-4">
        Adding signs to the Transformer model requires collecting <strong>sequences</strong> (not static images).
        Please use the command-line tools:
      </p>
      
      <div class="bg-gray-800 text-green-400 p-4 rounded-lg font-mono text-sm mb-4">
        <p class="mb-2"># Step 1: Collect sequences for "${signName}"</p>
        <p class="mb-2">python scripts/add_new_signs.py ${signName} --sequences 15</p>
        <p class="mb-2"># Step 2: Retrain the model</p>
        <p class="mb-2">python scripts/train_transformer_pytorch.py</p>
        <p># Step 3: Restart the Flask app</p>
      </div>
      
      <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4">
        <p class="text-sm text-yellow-800">
          <strong>💡 Tip:</strong> Use 15-20 sequences for dynamic signs (hello, how are you) 
          and 10 sequences for static signs (A, B, C).
        </p>
      </div>
      
      <button
        onclick="closeModalHandler()"
        class="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-semibold"
      >
        Got it! I'll use the CLI
      </button>
    </div>
  `;

  step3Content.innerHTML = instructions;
  showStep(3);
}
