const $ = (id) => document.getElementById(id);

const now = new Date();
now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
$("started_at").value = now.toISOString().slice(0, 16);

fetch("/health")
  .then((response) => response.json())
  .then((data) => {
    const element = $("health");
    element.textContent = data.status === "ok" ? `${data.model_name} · online` : "model niet getraind";
    element.classList.toggle("ok", data.status === "ok");
  })
  .catch(() => { $("health").textContent = "API offline"; });

$("trip-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = $("result");
  result.hidden = false;
  result.className = "result";
  result.innerHTML = "Berekenen…";
  const payload = {
    started_at: $("started_at").value,
    rideable_type: $("rideable_type").value,
    member_casual: $("member_casual").value,
    start_station_id: $("start_station_id").value,
    start_lat: Number($("start_lat").value),
    start_lng: Number($("start_lng").value),
  };
  try {
    const response = await fetch("/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Voorspelling mislukt");
    result.innerHTML = `<strong>${data.predicted_duration_minutes} min</strong><small>${data.message} Hold-out MAE: ${data.holdout_mae_minutes ?? "n.v.t."} min.</small>`;
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
  }
});

