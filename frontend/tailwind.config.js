/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Warm terracotta accent (was a cool teal in the dark theme).
        teal: {
          300: '#dc9468',
          400: '#c1683f',
          500: '#a8532e',
        },
        // Warm cream/tan surfaces (was navy panels in the dark theme).
        ink: {
          900: '#f6efe8',
          800: '#ffffff',
          700: '#f5ece2',
          600: '#e8dccd',
          500: '#d8c2a8',
        },
        // Inverted + warmed so headings (100) read darkest on the light ground
        // and faint labels (700+) read lightest — same rank order the dark
        // theme relied on, just flipped for a light background.
        slate: {
          100: '#2b2320',
          200: '#3a2f2a',
          300: '#4d4139',
          400: '#7a655c',
          500: '#9c8a80',
          600: '#b3a49a',
          700: '#c9bdb2',
          800: '#d8cec2',
          900: '#e5ddd2',
        },
        red: { 400: '#c0392b', 500: '#a8321f' },
        emerald: { 400: '#2f8a5c', 500: '#256e49' },
        amber: { 300: '#c98a2c', 400: '#b3761c' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        display: ['"Fraunces"', 'Georgia', 'serif'],
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-teal': 'pulse-teal 2s ease-in-out infinite',
        'fade-in': 'fade-in 0.5s ease-out',
        'drift': 'drift 8s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-teal': {
          '0%, 100%': { opacity: '0.6', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.05)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'drift': {
          '0%': { transform: 'translate(0px, 0px) rotate(0deg)' },
          '100%': { transform: 'translate(12px, -8px) rotate(3deg)' },
        },
      },
      boxShadow: {
        teal: '0 10px 28px rgba(193, 104, 63, 0.22)',
      },
      backgroundImage: {
        'dot-grid': "radial-gradient(rgba(193,104,63,0.14) 1px, transparent 1px)",
        'radial-teal': "radial-gradient(ellipse 80% 55% at 50% -10%, rgba(193,104,63,0.08), transparent)",
      },
      backgroundSize: {
        'dot-grid': '28px 28px',
      },
    },
  },
  plugins: [],
}
