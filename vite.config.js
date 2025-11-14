import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Auto-detect base for GitHub Pages when building in GitHub Actions
// - User/Org site (repo ends with .github.io): base = '/'
// - Project site: base = '/<repo>/'
const isCI = process.env.GITHUB_ACTIONS === 'true'
const repoFull = process.env.GITHUB_REPOSITORY || ''
const repo = repoFull.includes('/') ? repoFull.split('/')[1] : repoFull
const isUserSite = repo.endsWith('.github.io')
const base = isCI && repo ? (isUserSite ? '/' : `/${repo}/`) : '/'

export default defineConfig({
  plugins: [react()],
  base,
})


