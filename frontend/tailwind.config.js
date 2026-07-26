/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        teal: {
          300: '#5eead4',
          400: '#00d4aa',
          500: '#00b896',
          600: '#00806b',
        },
        ink: {
          950: '#070b14',
          900: '#0d1322',
          800: '#111e2e',
          700: '#162030',
          600: '#1e2d3d',
          500: '#253548',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-teal': 'pulse-teal 2s ease-in-out infinite',
        'scan-line': 'scan-line 2.4s linear infinite',
        'fade-in': 'fade-in 0.5s ease-out',
        'slide-up': 'slide-up 0.45s ease-out',
        'drift': 'drift 8s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-teal': {
          '0%, 100%': { opacity: '0.6', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.05)' },
        },
        'scan-line': {
          '0%': { transform: 'translateY(0%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(18px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'drift': {
          '0%': { transform: 'translate(0px, 0px) rotate(0deg)' },
          '100%': { transform: 'translate(12px, -8px) rotate(3deg)' },
        },
      },
      boxShadow: {
        teal: '0 0 28px rgba(0, 212, 170, 0.15)',
        'teal-sm': '0 0 12px rgba(0, 212, 170, 0.1)',
        card: '0 4px 32px rgba(0,0,0,0.45)',
      },
      backgroundImage: {
        'dot-grid': "radial-gradient(rgba(0,212,170,0.18) 1px, transparent 1px)",
        'radial-teal': "radial-gradient(ellipse 80% 55% at 50% -10%, rgba(0,212,170,0.07), transparent)",
      },
      backgroundSize: {
        'dot-grid': '28px 28px',
      },
    },
  },
  plugins: [],
}
