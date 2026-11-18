/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        theatre: {
          950: '#070a12',
          900: '#0b1120',
          800: '#131e36',
          700: '#1e2e50',
          cyan: '#06b6d4',
          amber: '#f59e0b',
          crimson: '#f43f5e',
        },
      },
      fontFamily: {
        devanagari: ['"Noto Sans Devanagari"', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
