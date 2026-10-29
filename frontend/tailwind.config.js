/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#071426', // Deep navy background
          900: '#101F35', // Card navy background
          850: '#152642', // Card hover / elevated
          800: '#1E2E48', // Card borders / dividers
          700: '#2A3F60',
        },
        teal: {
          400: '#2DD4BF',
          500: '#14B8A6',
          600: '#0D9488',
        },
        accent: {
          teal: '#14B8A6',
          blue: '#60A5FA',
          purple: '#A78BFA',
        },
        coolwhite: '#DCE7F5',
        muted: '#8DA4BE',
        lightmuted: '#64748B',
        brand: {
          50: '#f0fdf4',
          100: '#dcfce7',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
        },
      },
    },
  },
  plugins: [],
}
