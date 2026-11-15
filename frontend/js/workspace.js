const textInput = document.getElementById("textInput");
const suggestionList = document.getElementById("suggestionList");
const predictionStatus = document.getElementById("predictionStatus");

const wordCount = document.getElementById("wordCount");
const contextCount = document.getElementById("contextCount");
const tokenDisplay = document.getElementById("tokenDisplay");

const clearButton = document.getElementById("clearButton");
const topKSelect = document.getElementById("topKSelect");

let debounceTimer = null;
let latestRequestId = 0;


function updateWordCount() {

    const text = textInput.value.trim();

    const words = text
        ? text.split(/\s+/).length
        : 0;

    wordCount.textContent = `${words} words`;
}


function showLoading() {

    suggestionList.innerHTML = `
        <div class="empty-state">
            Generating predictions...
        </div>
    `;

    predictionStatus.textContent = "Processing your context...";
}


function showError(message) {

    suggestionList.innerHTML = `
        <div class="empty-state error-state">
            ${escapeHTML(message)}
        </div>
    `;

    predictionStatus.textContent = "Prediction failed.";
}


function escapeHTML(value) {

    return String(value).replace(/[&<>"']/g, function (character) {
        const replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        };

        return replacements[character];
    });
}


function renderPredictions(data) {

    const predictions = data.predictions || [];

    if (predictions.length === 0) {

        suggestionList.innerHTML = `
            <div class="empty-state">
                No valid word suggestions were returned.
            </div>
        `;

        return;
    }

    suggestionList.innerHTML = "";

    predictions.forEach((prediction, index) => {

        const item = document.createElement("button");

        item.type = "button";
        item.className = "suggestion-item";

        item.innerHTML = `
            <span class="suggestion-rank">${index + 1}</span>

            <span class="suggestion-word">
                ${escapeHTML(prediction.word)}
            </span>

            <span class="suggestion-probability">
                ${Number(prediction.percentage).toFixed(2)}%
            </span>
        `;

        item.addEventListener("click", function () {
            acceptSuggestion(prediction.word);
        });

        suggestionList.appendChild(item);

    });

}


function updateTokenDisplay(tokens) {

    tokenDisplay.innerHTML = "";

    if (!tokens || tokens.length === 0) {
        tokenDisplay.textContent = "No tokens yet.";
        contextCount.textContent = "Context: 0 / 10 tokens";
        return;
    }

    const actualTokens = tokens.slice(-10);

    actualTokens.forEach(token => {

        const span = document.createElement("span");

        span.className = "token-chip";
        span.textContent = token;

        tokenDisplay.appendChild(span);

    });

    contextCount.textContent =
        `Context: ${actualTokens.length} / 10 tokens`;
}


function acceptSuggestion(word) {

    const currentText = textInput.value;

    const separator =
        currentText.length > 0 &&
        !/\s$/.test(currentText)
            ? " "
            : "";

    textInput.value = currentText + separator + word + " ";

    updateWordCount();

    textInput.focus();

    requestPrediction();
}


async function requestPrediction() {

    const text = textInput.value.trim();

    updateWordCount();

    if (!text) {

        suggestionList.innerHTML = `
            <div class="empty-state">
                Start typing to receive predictions.
            </div>
        `;

        predictionStatus.textContent = "Waiting for input.";

        updateTokenDisplay([]);

        return;
    }

    const requestId = ++latestRequestId;

    showLoading();

    try {

        const data = await getPrediction(
            text,
            Number(topKSelect.value)
        );

        // Ignore older responses if the user has typed more text.
        if (requestId !== latestRequestId) {
            return;
        }

        renderPredictions(data);

        updateTokenDisplay(data.context_tokens);

        predictionStatus.textContent =
            `Prediction completed using ${data.model}.`;

    } catch (error) {

        if (requestId !== latestRequestId) {
            return;
        }

        showError(error.message);

    }

}


function schedulePrediction() {

    clearTimeout(debounceTimer);

    debounceTimer = setTimeout(() => {
        requestPrediction();
    }, 500);

}


textInput.addEventListener("input", schedulePrediction);


topKSelect.addEventListener("change", function () {

    if (textInput.value.trim()) {
        requestPrediction();
    }

});


clearButton.addEventListener("click", function () {

    textInput.value = "";

    updateWordCount();
    updateTokenDisplay([]);

    suggestionList.innerHTML = `
        <div class="empty-state">
            Start typing to receive predictions.
        </div>
    `;

    predictionStatus.textContent = "Waiting for input.";

    textInput.focus();

});


checkBackendHealth()
    .then(data => {

        if (!data.model_files_available) {

            predictionStatus.textContent =
                "Model files are missing. Check backend/models.";

        } else {

            predictionStatus.textContent =
                "Backend is connected. Start typing.";

        }

    })
    .catch(() => {

        predictionStatus.textContent =
            "Cannot connect to the backend.";

    });