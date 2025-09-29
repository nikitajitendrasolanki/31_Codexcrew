/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#0f1115",  // darkest background
          800: "#1a1c20",  // dashboard bg
          700: "#2a2d32",  // table bg
        },
        primary: {
          DEFAULT: "#ef4444", // red-500
          700: "#dc2626"
        }
      },
      fontFamily: {
        display: ["Rajdhani", "sans-serif"],
      },
      boxShadow: {
        "glow-red": "0 0 15px rgba(239,68,68,0.6)",
      },
    },
  },
  plugins: [],
}
