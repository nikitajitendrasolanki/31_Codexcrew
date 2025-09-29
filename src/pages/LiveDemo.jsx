import React from "react";

export default function LiveDemo() {
  return (
    <div className="min-h-screen bg-tech p-6">
      <h2 className="text-2xl font-bold mb-6 text-white">
        🚦 Live Traffic Violation Detection
      </h2>

      {/* Streamlit UI Embed */}
      <div className="w-full h-[85vh] rounded-xl overflow-hidden border border-white/10 shadow-lg">
        <iframe
          src="http://localhost:8501/"
          title="Traffic Violation Detection"
          className="w-full h-full"
        />
      </div>
    </div>
  );
}
