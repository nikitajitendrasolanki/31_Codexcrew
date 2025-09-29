import React from "react";
import { motion } from "framer-motion";

export default function Landing({ navigate }) {
  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center px-6 py-16 text-center overflow-hidden">
      
      {/* 🔥 Animated Gradient Background */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-gray-800"
        animate={{
          backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
        style={{
          backgroundSize: "400% 400%",
          zIndex: -2,
        }}
      />

      {/* 🔴 Moving Particles */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-primary rounded-full opacity-40"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            animate={{
              y: [null, -50],
              opacity: [0.2, 1, 0.2],
            }}
            transition={{
              duration: Math.random() * 10 + 5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      {/* 🔹 Main content */}
      <div className="max-w-4xl relative z-10">
        <div className="eyebrow mb-3 text-primary">Why Choose Us</div>
        <h1 className="heading-xl mb-4 text-white drop-shadow-lg">
          AI-Powered <span className="text-primary">Traffic</span> Violation Detection
        </h1>
        <p className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto mb-10">
          Automatic detection of traffic violations with explainability and fairness
          for a safer, smarter city.
        </p>
        <div className="divider-red mb-10 mx-auto w-32" />
      </div>

      {/* 🔥 Glassmorphism CTA Card */}
      <motion.div
        className="relative z-10 backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-6 shadow-2xl"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-3xl">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/live")}
            className="btn-primary w-full"
          >
            🚦 Live Demo
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/dashboard")}
            className="btn-ghost w-full"
          >
            📊 Dashboard
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/report")}
            className="btn-ghost w-full"
          >
            📑 Audit Report
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
