module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0052CC',
          light: '#E6F0FF'
        },
        secondary: '#00B8D4',
        critical: '#DC2626',
        success: '#059669',
        warning: '#D97706',
        slate: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A'
        }
      },
      borderRadius: {
        '2xl': '1.25rem',
        '3xl': '1.5rem'
      },
      boxShadow: {
        'glow': '0 0 20px rgba(99,102,241,0.3)',
        'glass': '0 8px 32px 0 rgba(0,0,0,0.15)',
        'card': '0 8px 32px rgba(0,0,0,0.2)',
        'card-hover': '0 12px 40px rgba(0,0,0,0.25)'
      },
      animation: {
        'gradient': 'gradient 8s linear infinite',
        'pulse-slow': 'pulse 3s infinite'
      },
      keyframes: {
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        }
      }
    }
  },
  plugins: []
}
