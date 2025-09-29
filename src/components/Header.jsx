import React from "react";

export default function Header({ route, navigate }) {
  return (
    <header className="sticky top-0 z-40 backdrop-blur bg-surface-900/70 border-b border-white/5">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-primary" />
          <span className="font-display font-semibold">Ruby Vision</span>
        </div>
        <nav className="hidden md:flex items-center gap-2">
          <button onClick={() => navigate("/live")} className={`btn-ghost ${route === "/live" ? "bg-primary/10" : ""}`}>Live Demo</button>
          <button onClick={() => navigate("/dashboard")} className={`btn-ghost ${route === "/dashboard" ? "bg-primary/10" : ""}`}>Dashboard</button>
          <button onClick={() => navigate("/report")} className={`btn-ghost ${route === "/report" ? "bg-primary/10" : ""}`}>Audit Report</button>
        </nav>
        <div className="md:hidden">
          <button onClick={() => navigate("/")} className="btn-primary">Home</button>
        </div>
      </div>
    </header>
  );
}


