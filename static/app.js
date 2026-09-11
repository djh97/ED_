const payloadEditor = document.getElementById("payload-editor");
const loadSampleButton = document.getElementById("load-sample");
const runButton = document.getElementById("run-evaluation");
const apiStatus = document.getElementById("api-status");
const summaryCard = document.getElementById("summary-card");
const recommendationsList = document.getElementById("recommendations-list");
const followUpList = document.getElementById("follow-up-list");
const toolOutputList = document.getElementById("tool-output-list");
const agentTraceList = document.getElementById("agent-trace-list");

function prettyJson(data) {
  return JSON.stringify(data, null, 2);
}

function renderStatusClass(state) {
  return `status-${state}`;
}

function recommendationPriorityClass(priority) {
  return `priority-${priority}`;
}

function setSummaryEmpty(message, isError = false) {
  summaryCard.className = `summary-card${isError ? " error" : " empty"}`;
  summaryCard.innerHTML = `<p>${message}</p>`;
}

function renderRecommendations(recommendations) {
  if (!recommendations.length) {
    recommendationsList.className = "stack empty-state";
    recommendationsList.textContent = "No recommendations returned.";
    return;
  }

  recommendationsList.className = "stack";
  recommendationsList.innerHTML = recommendations.map((item) => `
    <article class="card">
      <div class="card-head">
        <div class="card-title">${item.action}</div>
        <span class="status-pill ${recommendationPriorityClass(item.priority)}">${item.priority}</span>
      </div>
      <div class="subtle">${item.target_id ? `Target: <span class="mono">${item.target_id}</span>` : "Operational recommendation"}</div>
      <p>${item.reason}</p>
    </article>
  `).join("");
}

function renderToolOutputs(toolOutputs) {
  const cards = [
    {
      title: "Flow Prediction Tool",
      rationale: toolOutputs.flow_prediction.rationale,
      metrics: {
        "Congestion score": toolOutputs.flow_prediction.congestion_score,
        "Predicted wait": `${toolOutputs.flow_prediction.predicted_wait_minutes} min`,
        "Bottleneck": toolOutputs.flow_prediction.bottleneck_level
      }
    },
    {
      title: "Patient Risk Tool",
      rationale: toolOutputs.patient_risk.rationale,
      metrics: {
        "Highest-risk patient": toolOutputs.patient_risk.highest_risk_patient_id || "none",
        "Flagged patients": toolOutputs.patient_risk.flagged_patients.length,
        "Top risk level": toolOutputs.patient_risk.flagged_patients[0]?.risk_level || "none"
      }
    },
    {
      title: "Staffing Availability Tool",
      rationale: toolOutputs.staffing.rationale,
      metrics: {
        "Pressure score": toolOutputs.staffing.staffing_pressure_score,
        "Staffing level": toolOutputs.staffing.staffing_level,
        "Hourly capacity": toolOutputs.staffing.estimated_hourly_capacity
      }
    },
    {
      title: "Bed Management Tool",
      rationale: toolOutputs.bed_management.rationale,
      metrics: {
        "Occupancy rate": toolOutputs.bed_management.occupancy_rate,
        "Free beds now": toolOutputs.bed_management.available_beds_now,
        "Action window": toolOutputs.bed_management.action_window
      }
    }
  ];

  toolOutputList.className = "stack";
  toolOutputList.innerHTML = cards.map((card) => `
    <article class="card">
      <div class="card-head">
        <div class="card-title">${card.title}</div>
      </div>
      <p>${card.rationale}</p>
      <div class="metric-grid">
        ${Object.entries(card.metrics).map(([label, value]) => `
          <div class="metric">
            <span>${label}</span>
            <strong>${value}</strong>
          </div>
        `).join("")}
      </div>
    </article>
  `).join("");
}

function renderFollowUpPlan(followUpPlan = []) {
  if (!followUpPlan.length) {
    followUpList.className = "stack empty-state";
    followUpList.textContent = "No follow-up tasks returned.";
    return;
  }

  followUpList.className = "stack";
  followUpList.innerHTML = followUpPlan.map((item) => `
    <article class="card">
      <div class="card-head">
        <div class="card-title">${item.task_id}: ${item.linked_action}</div>
        <span class="status-pill priority-medium">${item.status}</span>
      </div>
      <div class="subtle">Owner: <span class="mono">${item.owner}</span> - Due in ${item.due_minutes} min</div>
      <p>${item.escalation_rule}</p>
    </article>
  `).join("");
}

function renderAgentTrace(agentTrace = []) {
  if (!agentTrace.length) {
    agentTraceList.className = "stack empty-state";
    agentTraceList.textContent = "No agent trace returned.";
    return;
  }

  agentTraceList.className = "stack";
  agentTraceList.innerHTML = agentTrace.map((item) => `
    <article class="card">
      <div class="card-head">
        <div class="card-title">${item.agent}</div>
      </div>
      <p>${item.step}</p>
      <div class="subtle">${(item.evidence || []).join(" - ")}</div>
    </article>
  `).join("");
}

function renderResponse(data) {
  summaryCard.className = "summary-card";
  summaryCard.innerHTML = `
    <div class="status-pill ${renderStatusClass(data.system_state)}">${data.system_state}</div>
    <p><strong>${data.action_brief || "No action brief returned."}</strong></p>
    <p>${data.summary}</p>
  `;
  renderRecommendations(data.recommendations);
  renderFollowUpPlan(data.follow_up_plan);
  renderToolOutputs(data.tool_outputs);
  renderAgentTrace(data.agent_trace);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    apiStatus.textContent = data.status === "ok" ? "API Ready" : "Unknown";
  } catch (error) {
    apiStatus.textContent = "API Offline";
  }
}

async function loadSample() {
  const response = await fetch("/sample-case");
  const data = await response.json();
  payloadEditor.value = prettyJson(data);
}

async function runEvaluation() {
  let parsed;
  try {
    parsed = JSON.parse(payloadEditor.value);
  } catch (error) {
    setSummaryEmpty("The JSON payload is invalid. Please fix the formatting and try again.", true);
    return;
  }

  summaryCard.className = "summary-card";
  summaryCard.innerHTML = "<p>Running the orchestration agent...</p>";

  try {
    const response = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed)
    });

    if (!response.ok) {
      const errorText = await response.text();
      setSummaryEmpty(`Evaluation failed: ${errorText}`, true);
      return;
    }

    const data = await response.json();
    renderResponse(data);
  } catch (error) {
    setSummaryEmpty("Could not reach the API. Make sure the local server is running.", true);
  }
}

loadSampleButton.addEventListener("click", loadSample);
runButton.addEventListener("click", runEvaluation);

checkHealth();
loadSample();
