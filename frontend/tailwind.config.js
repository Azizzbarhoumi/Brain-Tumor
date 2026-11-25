/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Consolas', 'monospace'],
      },
      colors: {
        // Near-black blue surfaces
        ink: {
          950: '#05080d',
          900: '#0a0f17',
          850: '#0e1520',
          800: '#131b28',
          700: '#1c2636',
          600: '#2a3648',
        },
        // The single accent colour
        accent: {
          200: '#b5e6ff',
          300: '#7dd5ff',
          DEFAULT: '#3bc4ff',
          500: '#3bc4ff',
          600: '#139ee0',
          700: '#0b77ad',
        },
        // Semantic verdict colours (used only for the verdict and the mask)
        lesion: '#ff4d5e',
        clear: '#34d399',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-400px 0' },
          '100%': { backgroundPosition: '400px 0' },
        },
        scan: {
          '0%': { top: '0%' },
          '100%': { top: '100%' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
      },
      animation: {
        'fade-up': 'fade-up 420ms cubic-bezier(0.2, 0.7, 0.2, 1) both',
        shimmer: 'shimmer 1.4s linear infinite',
        scan: 'scan 1.8s ease-in-out infinite alternate',
        'pulse-soft': 'pulse-soft 1.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
