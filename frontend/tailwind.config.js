/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', "system-ui", "sans-serif"],
        sans: ['"Outfit"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: {
          950: "#05070c",
          900: "#0a0e18",
          800: "#12182a",
          700: "#1b2438",
        },
        neon: {
          cyan: "#5ce1e6",
          magenta: "#ff4d8d",
          gold: "#ffd166",
          rose: "#ff4d6d",
        },
      },
      boxShadow: {
        glow: "0 0 80px rgba(92, 225, 230, 0.12)",
        board: "0 30px 80px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.06)",
      },
    },
  },
  plugins: [],
};
