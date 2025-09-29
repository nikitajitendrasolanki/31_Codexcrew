import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import AuditReport from "./pages/AuditReport";
import LiveDemo from "./pages/LiveDemo";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingWrapper />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/report" element={<AuditReport />} />
        <Route path="/live" element={<LiveDemo />} />
      </Routes>
    </Router>
  );
}

// 👇 navigate wrapper
function LandingWrapper() {
  const navigate = (path) => {
    window.location.href = path;
  };
  return <Landing navigate={navigate} />;
}
