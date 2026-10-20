// @ts-nocheck

import * as fs from 'fs'
import * as path from 'path'
import { execFileSync } from 'child_process'

import * as tar from 'tar'
import * as pty from 'node-pty'
import log from 'electron-log'

import {
  getConfig,
  setConfig,
  getInstallDir,
  portInUse,
  downloadFileWithProgress
} from './index'

import { getModelsDir } from './huggingface'
import { ServiceLock, isProcessAlive } from './service-lock'

// ─── State ──────────────────────────────────────────────

let ptyProcess: pty.IPty | null = null
let pid: number | null = null
let url: string | null = null
let status: string | null = null // null | setting-up | starting | started | stopped | failed
let logBuffer: string[] = []

const lock = new ServiceLock('llamacpp')
let binaryPath: string | null = null

// ─── Public Getters ─────────────────────────────────────

export const getLlamaCppInfo = () => {
  // Lazily discover a cached binary on cold boot so the UI never falsely
  // reports "not installed" when the files are actually on disk.
  if (!binaryPath) {
    const cacheBase = path.join(getInstallDir(), 'llama.cpp')
    try {
      if (fs.existsSync(cacheBase)) {
        const dirs = fs.readdirSync(cacheBase, { withFileTypes: true })
          .filter((d) => d.isDirectory())
        for (const d of dirs) {
          const found = findBinary(path.join(cacheBase, d.name))
          if (found) {
            binaryPath = found
            break
          }
        }
      }
    } catch {
      // Ignore — best-effort discovery
    }
  }

  // Extract version tag from binaryPath — the tag is the directory name
  // directly under the llama.cpp cache dir, e.g. …/llama.cpp/<tag>/bin/llama-server
  let version: string | null = null
  if (binaryPath) {
    const cacheBase = path.join(getInstallDir(), 'llama.cpp')
    const relative = path.relative(cacheBase, binaryPath)
    const tag = relative.split(path.sep)[0]
    if (tag) version = tag
  }
  return { url, status, pid, binaryPath, version }
}

export const getLlamaCppPty = (): pty.IPty | null => ptyProcess
export const getLlamaCppLog = (): string[] => logBuffer

// ─── Asset Resolution ───────────────────────────────────

interface ReleaseAsset {
  name: string
  browser_download_url: string
}

/**
 * Detect the best GPU variant for the current platform.
 * Returns the variant string (e.g. 'cuda-12.4', 'vulkan', 'rocm', 'cpu').
 */
const detectBestVariant = (): string => {
  const platform = process.platform

  // macOS: Metal is baked into the macOS binary; no variant choice needed.
  if (platform === 'darwin') return 'cpu'

  // Check for NVIDIA GPU (CUDA)
  try {
    execFileSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
      timeout: 5000,
      stdio: 'pipe'
    })
    // NVIDIA GPU detected
    if (platform === 'win32') return 'cuda-12.4'
    if (platform === 'linux') return 'cuda-12.8'
  } catch {
    // nvidia-smi not available or no NVIDIA GPU
  }

  // Check for Vulkan support
  try {
    execFileSync('vulkaninfo', ['--summary'], { timeout: 5000, stdio: 'pipe' })
    return 'vulkan'
  } catch {
    // Vulkan not available
  }

  // Linux: check for ROCm (AMD GPU)
  if (platform === 'linux') {
    try {
      if (fs.existsSync('/opt/rocm') || fs.existsSync('/usr/lib/rocm')) {
        return 'rocm'
      }
    } catch {
      // ROCm not available
    }
  }

  return 'cpu'
}

/**
 * Resolve the variant — if 'auto' or empty, detect the best one.
 */
const resolveVariant = (variant: string | undefined): string => {
  if (!variant || variant === 'auto') {
    const detected = detectBestVariant()
    log.info(`Auto-detected variant: ${detected}`)
    return detected
  }
  return variant
}

/**
 * Determine the correct release asset name for this platform/arch/variant.
 */
const getAssetPattern = (tag: string, variant: string): { pattern: string; isZip: boolean } => {
  const platform = process.platform
  const arch = process.arch

  if (platform === 'darwin') {
    const archStr = arch === 'arm64' ? 'arm64' : 'x64'
    return { pattern: `llama-${tag}-bin-macos-${archStr}.tar.gz`, isZip: false }
  }

  if (platform === 'linux') {
    const variantMap: Record<string, string> = {
      cpu: `llama-${tag}-bin-ubuntu-x64.tar.gz`,
      vulkan: `llama-${tag}-bin-ubuntu-vulkan-x64.tar.gz`,
      'cuda-12.8': `llama-${tag}-bin-ubuntu-cuda-12.8-x64.tar.gz`,
      'cuda-13.4': `llama-${tag}-bin-ubuntu-cuda-13.4-x64.tar.gz`,
      rocm: `llama-${tag}-bin-ubuntu-rocm-10.0-x64.tar.gz`
    }
    const name = variantMap[variant] ?? variantMap.cpu
    return { pattern: name, isZip: false }
  }

  if (platform === 'win32') {
    const archStr = arch === 'arm64' ? 'arm64' : 'x64'
    const variantMap: Record<string, string> = {
      cpu: `llama-${tag}-bin-win-cpu-${archStr}.zip`,
      'cuda-12.4': `llama-${tag}-bin-win-cuda-12.4-x64.zip`,
      'cuda-13.1': `llama-${tag}-bin-win-cuda-13.1-x64.zip`,
      'cuda-13.3': `llama-${tag}-bin-win-cuda-13.3-x64.zip`,
      'cuda-13.4': `llama-${tag}-bin-win-cuda-13.4-x64.zip`,
      vulkan: `llama-${tag}-bin-win-vulkan-x64.zip`
    }
    const name = variantMap[variant] ?? variantMap.cpu
    return { pattern: name, isZip: true }
  }

  return { pattern: `llama-${tag}-bin-ubuntu-x64.tar.gz`, isZip: false }
}

const getCudartPattern = (variant: string): string | null => {
  if (process.platform !== 'win32') return null
  if (variant === 'cuda-12.4') return 'cudart-llama-bin-win-cuda-12.4-x64.zip'
  if (variant === 'cuda-13.1') return 'cudart-llama-bin-win-cuda-13.1-x64.zip'
  if (variant === 'cuda-13.3') return 'cudart-llama-bin-win-cuda-13.3-x64.zip'
  if (variant === 'cuda-13.4') return 'cudart-llama-bin-win-cuda-13.4-x64.zip'
  return null
}

const FALLBACK_LLAMA_TAG = 'b11750'

const fetchLlamaReleaseData = async (version: string): Promise<any> => {
  const headers = {
    Accept: 'application/vnd.github.v3+json',
    'User-Agent': 'OpenWebUI-Desktop'
  }

  try {
    if (version && version !== 'latest') {
      const response = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${version}`, {
        headers,
        signal: AbortSignal.timeout(10000)
      })
      if (response.ok) return await response.json()
    } else {
      const response = await fetch('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10', {
        headers,
        signal: AbortSignal.timeout(10000)
      })
      if (response.ok) {
        const releases = await response.json()
        const buildRelease = releases.find((r: any) => r.tag_name?.startsWith('b') && Array.isArray(r.assets) && r.assets.length > 5)
        if (buildRelease) return buildRelease
      }
      const latestResp = await fetch('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest', {
        headers,
        signal: AbortSignal.timeout(8000)
      })
      if (latestResp.ok) {
        const data = await latestResp.json()
        if (Array.isArray(data.assets) && data.assets.length > 5) return data
      }
    }
  } catch (err) {
    log.warn('[llama.cpp] GitHub API request failed:', err)
  }

  const tag = (version && version !== 'latest') ? version : FALLBACK_LLAMA_TAG
  log.info(`[llama.cpp] Using fallback tag: ${tag}`)
  return {
    tag_name: tag,
    assets: []
  }
}

/**
 * Find the llama-server binary inside the extracted directory.
 */
const findBinary = (dir: string): string | null => {
  const exeName = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'

  const candidates = [
    path.join(dir, exeName),
    path.join(dir, 'bin', exeName),
    path.join(dir, 'build', 'bin', exeName)
  ]

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate
  }

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const nested = path.join(dir, entry.name, exeName)
        if (fs.existsSync(nested)) return nested
        const nestedBin = path.join(dir, entry.name, 'bin', exeName)
        if (fs.existsSync(nestedBin)) return nestedBin
      }
    }
  } catch {}

  return null
}

// ─── Setup (Download & Extract) ─────────────────────────

export const setupLlamaCpp = async (
  onStatus?: (status: string) => void
): Promise<string> => {
  const config = await getConfig()
  const llamaConfig = config.llamaCpp ?? {}
  const version = llamaConfig.version || 'latest'
  const variant = resolveVariant(llamaConfig.variant)

  const cacheBase = path.join(getInstallDir(), 'llama.cpp')
  if (!fs.existsSync(cacheBase)) {
    fs.mkdirSync(cacheBase, { recursive: true })
  }

  // ── Check for existing cached binary before any network request ──
  // This allows llama.cpp to start offline when previously installed.
  if (version !== 'latest') {
    // Pinned version — check its specific directory
    const pinnedDir = path.join(cacheBase, version)
    const pinnedBinary = fs.existsSync(pinnedDir) ? findBinary(pinnedDir) : null
    if (pinnedBinary) {
      log.info(`Using cached llama-server binary (pinned ${version}): ${pinnedBinary}`)
      binaryPath = pinnedBinary
      onStatus?.('Ready')
      return pinnedBinary
    }
  } else {
    // 'latest' — scan all cached version directories for a usable binary
    try {
      const cachedVersions = fs.readdirSync(cacheBase, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)

      for (const cachedTag of cachedVersions) {
        const cachedBinary = findBinary(path.join(cacheBase, cachedTag))
        if (cachedBinary) {
          log.info(`Found cached llama-server binary (${cachedTag}): ${cachedBinary}`)
          // Still try to fetch release info to see if there's a newer version,
          // but if the network is unavailable, use the cached binary.
          binaryPath = cachedBinary
          break
        }
      }
    } catch {
      // Cache directory scan failed — proceed to network fetch
    }
  }

  onStatus?.('Fetching release info…')
  let releaseData: any
  try {
    releaseData = await fetchLlamaReleaseData(version)
  } catch (error) {
    if (binaryPath) {
      log.info('Network unavailable, using cached llama-server binary:', binaryPath)
      onStatus?.('Ready (offline)')
      return binaryPath
    }
    releaseData = { tag_name: FALLBACK_LLAMA_TAG, assets: [] }
  }

  const tag = releaseData.tag_name || FALLBACK_LLAMA_TAG
  log.info(`llama.cpp release tag: ${tag}`)

  const versionDir = path.join(cacheBase, tag)
  if (!fs.existsSync(versionDir)) {
    fs.mkdirSync(versionDir, { recursive: true })
  }

  const existingBinary = findBinary(versionDir)
  if (existingBinary) {
    log.info(`llama-server binary already exists: ${existingBinary}`)
    binaryPath = existingBinary
    return existingBinary
  }

  const { pattern, isZip } = getAssetPattern(tag, variant)
  const asset = (releaseData.assets as ReleaseAsset[])?.find((a) => a.name === pattern)
  const downloadUrl = asset?.browser_download_url || `https://github.com/ggml-org/llama.cpp/releases/download/${tag}/${pattern}`

  log.info(`Downloading asset: ${pattern} from ${downloadUrl}`)
  onStatus?.(`Downloading ${pattern}…`)

  const downloadPath = path.join(versionDir, pattern)
  if (!fs.existsSync(downloadPath)) {
    await downloadFileWithProgress(downloadUrl, downloadPath, (progress) => {
      onStatus?.(`Downloading… ${progress.toFixed(0)}%`)
    })
  }

  onStatus?.('Extracting…')
  log.info(`Extracting ${downloadPath} to ${versionDir}`)

  if (isZip) {
    try {
      if (process.platform === 'win32') {
        execFileSync('powershell', [
          '-Command',
          `Expand-Archive -Path "${downloadPath}" -DestinationPath "${versionDir}" -Force`
        ])
      } else {
        execFileSync('unzip', ['-o', downloadPath, '-d', versionDir])
      }
    } catch (error) {
      throw new Error(`Failed to extract zip: ${error?.message ?? error}`)
    }
  } else {
    await tar.x({ cwd: versionDir, file: downloadPath })
  }

  try {
    fs.unlinkSync(downloadPath)
  } catch {}

  // If Windows CUDA variant, also download & extract matching cudart runtime DLLs
  const cudartName = getCudartPattern(variant)
  if (cudartName) {
    const cudartAsset = (releaseData.assets as ReleaseAsset[])?.find((a) => a.name === cudartName)
    const cudartUrl = cudartAsset?.browser_download_url || `https://github.com/ggml-org/llama.cpp/releases/download/${tag}/${cudartName}`
    const cudartPath = path.join(versionDir, cudartName)
    try {
      log.info(`Downloading CUDA runtime: ${cudartName}`)
      onStatus?.(`Downloading CUDA runtime…`)
      await downloadFileWithProgress(cudartUrl, cudartPath, (p) => {
        onStatus?.(`Downloading CUDA runtime… ${p.toFixed(0)}%`)
      })
      onStatus?.('Extracting CUDA runtime…')
      execFileSync('powershell', [
        '-Command',
        `Expand-Archive -Path "${cudartPath}" -DestinationPath "${versionDir}" -Force`
      ])
      try { fs.unlinkSync(cudartPath) } catch {}
    } catch (cErr) {
      log.warn(`Optional CUDA runtime download/extraction failed:`, cErr)
    }
  }

  if (process.platform !== 'win32') {
    const binary = findBinary(versionDir)
    if (binary) {
      try {
        fs.chmodSync(binary, 0o755)
      } catch {}
    }
  }

  const resultBinary = findBinary(versionDir)
  if (!resultBinary) {
    throw new Error(`llama-server binary not found after extraction in ${versionDir}`)
  }

  log.info(`llama-server binary ready: ${resultBinary}`)
  binaryPath = resultBinary
  onStatus?.('Ready')
  return resultBinary
}

export const checkLlamaCppUpdate = async (): Promise<{ currentVersion: string | null; latestVersion: string | null; updateAvailable: boolean }> => {
  const currentInfo = getLlamaCppInfo()

  try {
    const releaseData = await fetchLlamaReleaseData('latest')
    const latestVersion = releaseData.tag_name
    const currentVersion = currentInfo.version
    
    if (!currentVersion) {
      return { currentVersion: null, latestVersion, updateAvailable: true }
    }
    
    return { 
      currentVersion, 
      latestVersion, 
      updateAvailable: currentVersion !== latestVersion 
    }
  } catch (error) {
    log.error('Failed to check for llama.cpp updates:', error)
    return { 
      currentVersion: currentInfo.version, 
      latestVersion: null, 
      updateAvailable: false 
    }
  }
}

export const updateLlamaCpp = async (
  onStatus?: (status: string) => void
): Promise<{ url?: string; status?: string; pid?: number; binaryPath?: string; version?: string | null }> => {
  onStatus?.('Checking for updates…')
  let releaseTag: string
  try {
    const releaseData = await fetchLlamaReleaseData('latest')
    releaseTag = releaseData.tag_name
  } catch (error) {
    throw new Error(
      `Cannot update llama.cpp: unable to reach GitHub. ` +
      `Please check your internet connection. (${error?.message ?? error})`
    )
  }

  await stopLlamaCpp()
  
  const currentInfo = getLlamaCppInfo()
  if (currentInfo.version) {
    const cacheDir = path.join(getInstallDir(), 'llama.cpp', currentInfo.version)
    if (fs.existsSync(cacheDir)) {
      onStatus?.('Removing old version…')
      try {
        fs.rmSync(cacheDir, { recursive: true, force: true })
      } catch (err) {
        log.error(`Failed to remove old llama.cpp cache at ${cacheDir}:`, err)
      }
    }
  }
  
  const config = await getConfig()
  await setConfig({ llamaCpp: { ...config.llamaCpp, version: 'latest' } })
  onStatus?.('Downloading update…')
  await setupLlamaCpp(onStatus)
  
  return getLlamaCppInfo()
}

// ─── Lifecycle ──────────────────────────────────────────

export const startLlamaCpp = async (
  onStatus?: (status: string) => void
): Promise<{ url: string; pid: number }> => {
  if (!lock.acquire()) {
    return { url, pid }
  }

  await stopLlamaCpp()

  status = 'setting-up'
  onStatus?.('Setting up llama.cpp…')

  const binary = await setupLlamaCpp(onStatus)

  status = 'starting'
  onStatus?.('Starting llama-server…')

  const config = await getConfig()
  const llamaConfig = config.llamaCpp ?? {}
  const host = '127.0.0.1'

  let desiredPort = llamaConfig.port || 18881
  let availablePort = desiredPort
  while (await portInUse(availablePort, host)) {
    availablePort++
    if (availablePort > desiredPort + 100) {
      throw new Error('No available port found for llama-server')
    }
  }

  const extraArgs = llamaConfig.extraArgs ?? []
  const modelsDir = getModelsDir()
  const commandArgs = ['--host', host, '--port', availablePort.toString(), '--models-dir', modelsDir, ...extraArgs]

  log.info('Starting llama-server:', binary, commandArgs.join(' '))

  let spawned: pty.IPty
  try {
    spawned = pty.spawn(binary, commandArgs, {
      name: 'xterm-256color',
      cols: 200,
      rows: 50,
      env: {
        ...process.env,
        ...(config.envVars ?? {})
      }
    })
  } catch (error) {
    status = 'failed'
    throw new Error(`Failed to spawn llama-server: ${error?.message ?? error}`)
  }

  const spawnedPid = spawned.pid
  logBuffer = []
  ptyProcess = spawned
  pid = spawnedPid

  spawned.onData((data: string) => {
    logBuffer.push(data)
    log.info(`[llamacpp:${spawnedPid}] ${data.replace(/[\r\n]+/g, ' ').trim()}`)
  })

  spawned.onExit(({ exitCode, signal }) => {
    log.info(`[llamacpp:${spawnedPid}] Exited code=${exitCode} signal=${signal}`)
    const exitMsg = `\r\n[Process exited with code ${exitCode}${signal ? ` signal ${signal}` : ''}]\r\n`
    logBuffer.push(exitMsg)
    ptyProcess = null
    pid = null
    url = null
    status = 'stopped'
  })

  const serverUrl = `http://${host}:${availablePort}`
  const maxAttempts = 30
  let ready = false

  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 1000))
    try {
      const resp = await fetch(`${serverUrl}/health`, { signal: AbortSignal.timeout(2000) })
      if (resp.ok) {
        const body = await resp.json()
        if (body.status === 'ok' || body.status === 'no slot available') {
          ready = true
          break
        }
      }
    } catch {
      // Not ready yet
    }
  }

  if (!ready) {
    log.warn('llama-server did not report healthy within 30s, continuing anyway')
  }

  url = serverUrl
  status = 'started'
  log.info(`llama-server started — PID: ${spawnedPid}, URL: ${serverUrl}`)

  return { url: serverUrl, pid: spawnedPid }
}

export const stopLlamaCpp = async (): Promise<void> => {
  if (ptyProcess) {
    try {
      ptyProcess.kill()
    } catch (e) {
      log.warn('Failed to kill llama-server PTY:', e)
    }
    await new Promise((r) => setTimeout(r, 2000))
    if (pid) {
      try {
        process.kill(pid, 0)
        process.kill(pid, 'SIGKILL')
      } catch {
        // already dead
      }
    }
  }
  ptyProcess = null
  pid = null
  url = null
  status = null
  logBuffer = []
  lock.release()
}

/**
 * Validate whether the tracked llama.cpp process is still alive.
 * Used for crash recovery on app startup.
 */
export const validateLlamaCppProcess = (): boolean => {
  if (!pid) return false
  if (isProcessAlive(pid)) return true
  // Stale PID — clean up
  pid = null
  status = null
  lock.release()
  return false
}

/**
 * Uninstall llama.cpp — stop the server and remove all downloaded binaries.
 */
export const uninstallLlamaCpp = async (): Promise<void> => {
  await stopLlamaCpp()

  const cacheBase = path.join(getInstallDir(), 'llama.cpp')
  if (fs.existsSync(cacheBase)) {
    fs.rmSync(cacheBase, { recursive: true, force: true })
    log.info('Removed llama.cpp directory:', cacheBase)
  }

  binaryPath = null
}
