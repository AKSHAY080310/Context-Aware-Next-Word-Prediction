
const MODEL_METRICS = {
    lstm: {
        name: "LSTM V3",
        loss: 6.3537,
        top1: 19.55,
        top5: 36.19,
        top10: 42.68
    },

    transformer: {
        name: "Transformer V2",
        loss: 4.8137,
        top1: 23.73,
        top5: 45.41,
        top10: 53.83
    },

    gpt2: {
        name: "GPT-2",
        loss: 3.9978,
        top1: 29.9015,
        top5: 53.4472,
        top10: 61.5908
    },

    distilgpt2: {
        name: "DistilGPT-2",
        loss: 4.2552,
        top1: 27.3931,
        top5: 49.8252,
        top10: 58.1229
    }
};


function formatPercent(value) {
    return `${value.toFixed(2)}%`;
}


function formatLoss(value) {
    return value.toFixed(4);
}


const tbody = document.getElementById("comparisonBody");

tbody.innerHTML = "";


Object.entries(MODEL_METRICS).forEach(([key, model]) => {

    const row = document.createElement("tr");
    row.dataset.model = key;

    const values = [
        model.name,
        formatLoss(model.loss),
        formatPercent(model.top1),
        formatPercent(model.top5),
        formatPercent(model.top10)
    ];

    values.forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
    });

    tbody.appendChild(row);
});


function updateSummary() {

    const key = document.getElementById(
        "metricModelSelect"
    ).value;

    const model = MODEL_METRICS[key];

    document.getElementById("metricLoss").textContent =
        formatLoss(model.loss);

    document.getElementById("metricTop1").textContent =
        formatPercent(model.top1);

    document.getElementById("metricTop5").textContent =
        formatPercent(model.top5);

    document.getElementById("metricTop10").textContent =
        formatPercent(model.top10);

    document.querySelectorAll(
        ".selected-model-name"
    ).forEach(element => {
        element.textContent = model.name;
    });

    tbody.querySelectorAll("tr").forEach(row => {
        row.classList.toggle(
            "highlight-row",
            row.dataset.model === key
        );
    });
}


document.getElementById(
    "metricModelSelect"
).addEventListener("change", updateSummary);


updateSummary();
