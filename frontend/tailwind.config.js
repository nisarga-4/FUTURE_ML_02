/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#16134A',
        violet: '#6C4DFF',
        'deep-violet': '#4E35D3',
        lavender: '#A994FF',
        ice: '#5FE7F2',
        success: '#29C788',
        danger: '#FF4F78',
      },
      fontFamily: {
        sans: ['Inter', 'Manrope', 'Segoe UI', 'sans-serif'],
      },
      boxShadow: {
        glass: '0 24px 70px rgba(70, 53, 172, .16)',
      },
    },
  },
  plugins: [],
}
