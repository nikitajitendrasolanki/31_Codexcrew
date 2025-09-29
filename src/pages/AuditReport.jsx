import React, { useEffect, useState } from "react";
import { fetchAudit } from "../services/api";
import { Card, CardContent } from "../components/Card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

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
      setError("⚠️ Failed to load audit report");
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

  if (initialLoading)
    return (
      <p className="text-center text-gray-400 animate-pulse">
        ⏳ Loading audit report...
      </p>
    );
  if (error) return <p className="text-red-400 text-center">{error}</p>;
  if (!audit) return null;

  return (
    <div className="p-6 space-y-8 bg-gradient-to-br from-gray-950 via-gray-900 to-black min-h-screen text-gray-200">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-gray-800 pb-4">
        <h2 className="text-3xl font-extrabold text-primary drop-shadow">
          📊 Audit Report
        </h2>
        <div className="flex gap-4 items-center">
          {refreshing && (
            <span className="text-sm text-gray-400 animate-pulse">
              🔄 Updating...
            </span>
          )}
          <button
            onClick={downloadPDF}
            className="px-4 py-2 bg-gradient-to-r from-primary to-blue-500 rounded-xl shadow-lg hover:opacity-90 transition-all duration-200"
          >
            ⬇ Export PDF
          </button>
        </div>
      </div>

      {/* Model Info + Robustness */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-gray-900/80 backdrop-blur-xl border border-gray-700 shadow-lg rounded-2xl">
          <CardContent className="p-5 space-y-2">
            <h3 className="text-lg font-semibold text-primary">🧠 Model Info</h3>
            <p>
              <b>Model:</b> {audit.model}
            </p>
            <p>
              <b>Date:</b> {new Date(audit.date).toLocaleString()}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/80 backdrop-blur-xl border border-gray-700 shadow-lg rounded-2xl">
          <CardContent className="p-5 space-y-2">
            <h3 className="text-lg font-semibold text-primary">🛡️ Robustness</h3>
            <pre className="text-sm bg-gray-800 text-gray-300 p-3 rounded-lg max-h-40 overflow-y-auto">
              {audit.adv_summary}
            </pre>
          </CardContent>
        </Card>
      </div>

      {/* Metrics */}
      <Card className="bg-gray-900/80 backdrop-blur-xl border border-gray-700 shadow-lg rounded-2xl">
        <CardContent className="p-5">
          <h3 className="text-lg font-semibold mb-3 text-primary">
            📈 Per-class Metrics
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={audit.metrics}>
              <XAxis dataKey="class" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip contentStyle={{ backgroundColor: "#111827", color: "#fff" }} />
              <Legend />
              <Bar dataKey="precision" fill="#34d399" radius={[6, 6, 0, 0]} />
              <Bar dataKey="recall" fill="#60a5fa" radius={[6, 6, 0, 0]} />
              <Bar dataKey="f1" fill="#fbbf24" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Violations */}
      <Card className="bg-gray-900/80 backdrop-blur-xl border border-gray-700 shadow-lg rounded-2xl">
        <CardContent className="p-5">
          <h3 className="text-lg font-semibold mb-3 text-primary">
            🚦 Violations Log
          </h3>
          {audit.violations.length === 0 ? (
            <p className="text-gray-400">✅ No violations recorded</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-700 text-sm">
                <thead>
                  <tr className="bg-gray-800/60 text-gray-300">
                    <th className="border p-2">Type</th>
                    <th className="border p-2">Vehicle</th>
                    <th className="border p-2">Time</th>
                    <th className="border p-2">Reason</th>
                    <th className="border p-2">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.violations.map((v, i) => (
                    <tr
                      key={i}
                      className="text-center hover:bg-gray-800/60 transition"
                    >
                      <td className="border p-2">{v.violation_type || "Unknown"}</td>
                      <td className="border p-2">{v.vehicle_no || "N/A"}</td>
                      <td className="border p-2">{v.timestamp || "N/A"}</td>
                      <td className="border p-2">{v.reason || "-"}</td>
                      <td className="border p-2">{v.conf || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Examples */}
      <Card className="bg-gray-900/80 backdrop-blur-xl border border-gray-700 shadow-lg rounded-2xl">
        <CardContent className="p-5">
          <h3 className="text-lg font-semibold mb-3 text-primary">
            🖼️ Explainability Samples
          </h3>
          {audit.examples.length === 0 ? (
            <p className="text-gray-400">No examples available</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {audit.examples.map((ex, i) => (
                <div
                  key={i}
                  className="text-center bg-gray-800/50 p-3 rounded-lg shadow"
                >
                  <p className="font-medium mb-2">{ex.title}</p>
                  <img
                    src={`http://127.0.0.1:8000/${ex.img}`}
                    alt={ex.title}
                    className="rounded-lg shadow-lg mx-auto max-h-64 object-contain"
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
