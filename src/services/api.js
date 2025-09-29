// src/services/api.js
const API_BASE = "http://127.0.0.1:8000"; // Flask backend base URL

// 🚦 Fetch all traffic violations
export async function fetchViolations() {
  const res = await fetch(`${API_BASE}/violations`);
  if (!res.ok) throw new Error("Failed to fetch violations");
  return res.json();
}

// 📊 Fetch audit report data
export async function fetchAudit() {
  const res = await fetch(`${API_BASE}/audit`);
  if (!res.ok) throw new Error("Failed to fetch audit report");
  return res.json();
}
