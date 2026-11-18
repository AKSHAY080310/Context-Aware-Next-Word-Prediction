
const textInput = document.getElementById("textInput");
const suggestionList = document.getElementById("suggestionList");
const predictionStatus = document.getElementById("predictionStatus");
const wordCount = document.getElementById("wordCount");
const contextCount = document.getElementById("contextCount");
const tokenDisplay = document.getElementById("tokenDisplay");
const clearButton = document.getElementById("clearButton");
const topKSelect = document.getElementById("topKSelect");
const modelSelect = document.getElementById("modelSelect");
const modelAvailability = document.getElementById("modelAvailability");
const modelNote = document.getElementById("modelNote");

const MODEL_LABELS = {
    lstm: "Custom LSTM",
    transformer: "Custom Transformer V2",
    gpt2: "GPT-2",
    distilgpt2: "DistilGPT-2"
};

let debounceTimer = null;
let latestRequestId = 0;


function escapeHTML(value) {

    return String(value).replace(
        /[&<>"']/g,
        c => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        })[c]
    );
}


function updateWordCount() {

    const value = textInput.value.trim();

    wordCount.textContent =
        `${value ? value.split(/\s+/).length : 0} words`;
}


function showLoading() {

    suggestionList.innerHTML =
        '<div class="empty-state">Generating predictions...</div>';

    predictionStatus.textContent =
        `Processing with ${MODEL_LABELS[modelSelect.value]}...`;
}


function showError(message) {

    suggestionList.innerHTML =
        `<div class="empty-state error-state">${escapeHTML(message)}</div>`;

    predictionStatus.textContent = "Prediction failed.";
}


function renderPredictions(data) {

    suggestionList.innerHTML = "";

    if (!data.predictions || !data.predictions.length) {

        suggestionList.innerHTML =
            '<div class="empty-state">No suggestions were returned.</div>';

        return;
    }

    data.predictions.forEach((prediction, index) => {

        const button = document.createElement("button");

        button.type = "button";
        button.className = "suggestion-item";

        button.innerHTML = `
            <span class="suggestion-rank">${index + 1}</span>
            <span class="suggestion-word">${escapeHTML(prediction.word)}</span>
            <span class="suggestion-probability">
                ${Number(prediction.percentage).toFixed(2)}%
            </span>
        `;

        button.addEventListener("click", () => {
            acceptSuggestion(prediction.word);
        });

        suggestionList.appendChild(button);
    });
}


function updateTokenDisplay(tokens, data) {

    tokenDisplay.innerHTML = "";

    if (!tokens || !tokens.length) {

        tokenDisplay.textContent = "No tokens yet.";

        contextCount.textContent = "Context: 0 tokens";

        return;
    }

    tokens.slice(-16).forEach(token => {

        const chip = document.createElement("span");

        chip.className = "token-chip";
        chip.textContent = token;

        tokenDisplay.appendChild(chip);
    });

    contextCount.textContent =
        `Context: ${tokens.length} tokens`;

    document.getElementById("contextWindow").textContent =
        data.context_size
            ? `${data.context_size} tokens`
            : `${tokens.length} tokens`;
}


function acceptSuggestion(token) {

    const current = textInput.value;

    const isHF = [
        "gpt2",
        "distilgpt2"
    ].includes(modelSelect.value);

    // GPT-family decoded tokens can include leading whitespace.
    // Preserve that whitespace when inserting a suggestion.

    textInput.value = isHF
        ? current + token
        : current + (/\s$/.test(current) ? "" : " ") + token + " ";

    updateWordCount();

    textInput.focus();

    requestPrediction();
}


async function requestPrediction() {

    const text = textInput.value.trim();

    updateWordCount();

    if (!text) {

        suggestionList.innerHTML =
            '<div class="empty-state">Start typing to receive predictions.</div>';

        predictionStatus.textContent = "Waiting for input.";

        tokenDisplay.textContent = "No tokens yet.";

        contextCount.textContent = "Context: 0 tokens";

        return;
    }

    const requestId = ++latestRequestId;

    showLoading();

    try {

        const data = await getPrediction(
            text,
            modelSelect.value,
            Number(topKSelect.value)
        );

        if (requestId !== latestRequestId) {
            return;
        }

        renderPredictions(data);

        updateTokenDisplay(
            data.context_tokens,
            data
        );

        document.getElementById("activeModel").textContent =
            data.model;

        document.getElementById("modelDescription").textContent =
            data.prediction_unit === "token"
                ? "Pretrained causal language model"
                : "Custom-trained next-word model";

        document.getElementById("suggestionTitle").textContent =
            data.prediction_unit === "token"
                ? "Next-token suggestions"
                : "Next-word suggestions";

        modelNote.textContent =
            data.note || "Custom vocabulary-based word prediction.";

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

    debounceTimer = setTimeout(
        requestPrediction,
        550
    );
}


// Automatically predict while the user types.

textInput.addEventListener(
    "input",
    schedulePrediction
);


// Update the number of suggestions.

topKSelect.addEventListener(
    "change",
    () => {
        if (textInput.value.trim()) {
            requestPrediction();
        }
    }
);


// Change the selected model.

modelSelect.addEventListener(
    "change",
    () => {

        document.getElementById("activeModel").textContent =
            MODEL_LABELS[modelSelect.value];

        document.getElementById("contextWindow").textContent =
            modelSelect.value === "lstm"
                ? "10 tokens"
                : modelSelect.value === "transformer"
                    ? "32 tokens"
                    : "Up to 1024 tokens";

        if (textInput.value.trim()) {
            requestPrediction();
        }
    }
);


// Clear the editor.

clearButton.addEventListener(
    "click",
    () => {

        textInput.value = "";

        updateWordCount();

        updateTokenDisplay([]);

        suggestionList.innerHTML =
            '<div class="empty-state">Start typing to receive predictions.</div>';

        predictionStatus.textContent = "Waiting for input.";

        textInput.focus();
    }
);


// Check whether the backend is running.

checkBackendHealth()
    .then(data => {

        const entries = Object.entries(
            data.models || {}
        );

        modelAvailability.textContent = entries
            .map(([key, info]) => {
                return `${info.label}: ${
                    info.available ? "files present" : "files missing"
                }${info.loaded ? " (loaded)" : ""}`;
            })
            .join(" | ");

        if (!data.models?.lstm?.available) {

            predictionStatus.textContent =
                "LSTM files missing. Check backend/models.";

        } else {

            predictionStatus.textContent =
                "Backend connected. Choose a model and start typing.";
        }
    })
    .catch(() => {

        modelAvailability.textContent =
            "Backend unavailable.";

        predictionStatus.textContent =
            "Cannot connect to the backend.";
    });
