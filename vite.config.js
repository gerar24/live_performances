import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Auto-detect base for GitHub Pages when building in GitHub Actions
// - User/Org site: base = '/'
// - Project site: base = '/<repo>/'
const isCI = process.env.GITHUB_ACTIONS === 'true'
const repo = process.env.GITHUB_REPOSITORY ? process.env.GITHUB_REPOSITORY.split('/')[1] : ''
const base = isCI && repo ? `/${repo}/` : '/'

export default defineConfig({
  plugins: [react()],
  base,
})


