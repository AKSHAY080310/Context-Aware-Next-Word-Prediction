
const API_BASE_URL = "/api";


async function getPrediction(
    text,
    model = "lstm",
    topK = 5
) {

    const response = await fetch(
        `${API_BASE_URL}/predict`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text,
                model: model,
                top_k: Number(topK)
            })
        }
    );

    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            data.error || "Prediction request failed."
        );
    }

    return data;
}


async function checkBackendHealth() {

    const response = await fetch(
        `${API_BASE_URL}/health`
    );

    if (!response.ok) {
        throw new Error("Backend is unavailable.");
    }

    return response.json();
}
