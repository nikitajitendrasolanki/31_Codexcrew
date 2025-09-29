import React, { useEffect, useState } from "react";
import { fetchAudit } from "../services/api";
import { Card, CardContent } from "../components/Card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function AuditReport() {
  const [audit, setAudit] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadAudit = async (isFirst = false) => {
    try {
      if (!isFirst) setRefreshing(true);
      const data = await fetchAudit();
      setAudit(data);
    } catch (err) {
      setError("Failed to load audit report");
    } finally {
      if (isFirst) setInitialLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadAudit(true);
    const interval = setInterval(() => loadAudit(false), 10000);
    return () => clearInterval(interval);
  }, []);

  // ✅ Direct download PDF
  const downloadPDF = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/audit/pdf");
      if (!response.ok) throw new Error("Failed to fetch PDF");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "audit_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("PDF download error:", err);
      alert("Failed to download PDF");
    }
  };

  if (initialLoading) return <p className="text-center text-gray-300">Loading audit report...</p>;
  if (error) return <p className="text-red-500 text-center">{error}</p>;
  if (!audit) return null;

  return (
    <div className="p-6 space-y-6 text-gray-200 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">📊 Audit Report</h2>
        <div className="flex gap-3 items-center">
          {refreshing && <span className="text-sm text-gray-400">🔄 Updating...</span>}
          <button
            onClick={downloadPDF}
            className="px-3 py-2 bg-primary rounded-lg shadow hover:bg-primary/80 transition"
          >
            ⬇ Export PDF
          </button>
        </div>
      </div>

      {/* Model Info + Robustness */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="bg-gray-900 text-gray-200">
          <CardContent className="p-4">
            <h3 className="text-lg font-semibold text-primary">🧠 Model Info</h3>
            <p><b>Model:</b> {audit.model}</p>
            <p><b>Date:</b> {new Date(audit.date).toLocaleString()}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 text-gray-200">
          <CardContent className="p-4">
            <h3 className="text-lg font-semibold text-primary">🛡️ Robustness</h3>
            <pre className="text-sm bg-gray-800 text-gray-300 p-2 rounded max-h-40 overflow-y-auto">
              {audit.adv_summary}
            </pre>
          </CardContent>
        </Card>
      </div>

      {/* Metrics */}
      <Card className="bg-gray-900 text-gray-200">
        <CardContent className="p-4">
          <h3 className="text-lg font-semibold mb-2 text-primary">📈 Per-class Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={audit.metrics}>
              <XAxis dataKey="class" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip contentStyle={{ backgroundColor: "#1f2937", color: "#fff" }} />
              <Legend />
              <Bar dataKey="precision" fill="#4ade80" />
              <Bar dataKey="recall" fill="#60a5fa" />
              <Bar dataKey="f1" fill="#facc15" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Violations */}
      <Card className="bg-gray-900 text-gray-200">
        <CardContent className="p-4">
          <h3 className="text-lg font-semibold mb-2 text-primary">🚦 Violations Log</h3>
          {audit.violations.length === 0 ? (
            <p className="text-gray-400">No violations recorded ✅</p>
          ) : (
            <table className="w-full border-collapse border border-gray-700">
              <thead>
                <tr className="bg-gray-800 text-gray-200">
                  <th className="border p-2">Type</th>
                  <th className="border p-2">Vehicle</th>
                  <th className="border p-2">Time</th>
                  <th className="border p-2">Reason</th>
                  <th className="border p-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {audit.violations.map((v, i) => (
                  <tr key={i} className="text-center hover:bg-gray-800">
                    <td className="border p-2">{v.violation_type || "Unknown"}</td>
                    <td className="border p-2">{v.vehicle_no || "N/A"}</td>
                    <td className="border p-2">{v.timestamp || "N/A"}</td>
                    <td className="border p-2">{v.reason || "-"}</td>
                    <td className="border p-2">{v.conf || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      {/* Examples */}
      <Card className="bg-gray-900 text-gray-200">
        <CardContent className="p-4">
          <h3 className="text-lg font-semibold mb-2 text-primary">🖼️ Explainability Samples</h3>
          {audit.examples.length === 0 ? (
            <p className="text-gray-400">No examples available</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {audit.examples.map((ex, i) => (
                <div key={i} className="text-center">
                  <p className="font-medium">{ex.title}</p>
                  <img
                    src={`http://127.0.0.1:8000/${ex.img}`}
                    alt={ex.title}
                    className="rounded shadow-md mx-auto"
                  />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
