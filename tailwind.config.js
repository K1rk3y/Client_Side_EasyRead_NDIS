/** @type {import('tailwindcss').Config} */

module.exports = {
  content: [
    './easyreadweb/myapp/templates/**/*.html', // 匹配所有子目录
    './easyreadweb/myapp/static/**/*.js',
    './templates/**/*.html', // 如果有全局模板目录
    './app/templates/**/*.html', // 如果有其它 app
    // 其它你实际用到的模板路径
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}