import React, { useEffect, useState } from "react";
import { fetchViolations } from "../services/api";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer
} from "recharts";

export default function Dashboard() {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // ----------------- Load Function -----------------
  const loadData = async (isFirst = false) => {
    try {
      if (!isFirst) setRefreshing(true);
      const data = await fetchViolations();
      setViolations(data || []);
    } catch (err) {
      setError("❌ Failed to load violations");
    } finally {
      if (isFirst) setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData(true); // first load
    const interval = setInterval(() => loadData(false), 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading) return <p className="text-center">⏳ Loading violations...</p>;
  if (error) return <p className="text-red-500 text-center">{error}</p>;

  // ----------------- Data Transform -----------------
  const totalViolations = violations.length;
  const violationTypes = {};
  violations.forEach((v) => {
    violationTypes[v.type] = (violationTypes[v.type] || 0) + 1;
  });

  const barData = Object.entries(violationTypes).map(([type, count]) => ({
    type,
    count,
  }));

  // Group by timestamp for line chart
  const timeData = violations.reduce((acc, v) => {
    const t = v.timestamp?.slice(0, 16); // minute precision
    if (!acc[t]) acc[t] = 0;
    acc[t]++;
    return acc;
  }, {});
  const lineData = Object.entries(timeData).map(([time, count]) => ({
    time,
    count,
  }));

  const latestViolation = violations[0] || null;

  // ----------------- UI -----------------
  return (
    <div className="p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">🚦 Traffic Violations Dashboard</h2>
        {refreshing && <p className="text-sm text-gray-500">🔄 Updating...</p>}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="card p-4 text-center">
          <h3 className="text-lg font-semibold">Total Violations</h3>
          <p className="text-3xl font-bold text-red-500">{totalViolations}</p>
        </div>
        <div className="card p-4 text-center">
          <h3 className="text-lg font-semibold">Violation Types</h3>
          <p className="text-3xl font-bold text-blue-500">{Object.keys(violationTypes).length}</p>
        </div>
        <div className="card p-4 text-center">
          <h3 className="text-lg font-semibold">Latest</h3>
          <p className="text-md text-gray-600">
            {latestViolation ? `${latestViolation.type} @ ${latestViolation.timestamp}` : "N/A"}
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Bar Chart */}
        <div className="card p-4">
          <h3 className="text-xl font-semibold mb-4">Violation Types</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Line Chart */}
        <div className="card p-4">
          <h3 className="text-xl font-semibold mb-4">Violations Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Table */}
      <div className="card p-4">
        <h3 className="text-xl font-semibold mb-4">Violation Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300 text-sm">
            <thead>
              <tr className="bg-gray-100 text-left">
                <th className="border border-gray-300 p-2">ID</th>
                <th className="border border-gray-300 p-2">Type</th>
                <th className="border border-gray-300 p-2">Vehicle</th>
                <th className="border border-gray-300 p-2">Reason</th>
                <th className="border border-gray-300 p-2">Confidence</th>
                <th className="border border-gray-300 p-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {violations.map((v, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="border border-gray-300 p-2">{v.id || i + 1}</td>
                  <td className="border border-gray-300 p-2">{v.type || "Unknown"}</td>
                  <td className="border border-gray-300 p-2">{v.vehicle_no || "N/A"}</td>
                  <td className="border border-gray-300 p-2">{v.reason || "N/A"}</td>
                  <td className="border border-gray-300 p-2">{(v.confidence * 100).toFixed(1)}%</td>
                  <td className="border border-gray-300 p-2">{v.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
