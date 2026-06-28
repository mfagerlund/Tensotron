<#
.SYNOPSIS
  Run the Tensotron test suite at low CPU priority so it doesn't hog the machine.

.DESCRIPTION
  Launches `dotnet test` and forces its (and the spawned testhost's) process
  priority down. ILGPU's CPU-accelerator worker threads live inside testhost, so
  lowering the process priority class lowers those threads too.

.PARAMETER Priority
  ProcessPriorityClass to use. Default BelowNormal. Use Idle for the absolute lowest.

.PARAMETER Filter
  Optional xUnit --filter expression. If given, it is used verbatim and overrides -Showcase.

.PARAMETER Showcase
  Run ONLY the full-strength Category=Showcase convergence tests (pole-cart PPO, MNIST CNN).
  These are slow (minutes+; intended for a GPU) and are excluded from a normal run.
  By default (no -Filter, no -Showcase) the showcase convergence tests are skipped; the fast
  always-on ShowcaseSmokeTests still run.
#>
param(
    [ValidateSet("Idle", "BelowNormal", "Normal")]
    [string]$Priority = "BelowNormal",
    [string]$Filter = "",
    [switch]$Showcase
)

$ErrorActionPreference = "Stop"

# Clear ONLY stray test hosts left over from a previous run of THIS repo (an interrupted run
# can leave a testhost holding file/output-dir locks). Scoped by command line to the repo path
# so we never kill unrelated .NET work (build servers, other solutions' test runs, IDEs) on the
# machine — a blanket `Get-Process dotnet | Stop-Process` would be destructive on a CI agent.
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @('testhost.exe', 'testhost.x86.exe', 'vstest.console.exe') -and
        $_.CommandLine -and $_.CommandLine -like "*$repoRoot*"
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$pc = [System.Diagnostics.ProcessPriorityClass]::$Priority
$argList = "test `"$PSScriptRoot\..\Tensotron.sln`" --verbosity quiet --nologo"
if ($Filter) { $argList += " --filter `"$Filter`"" }
elseif ($Showcase) { $argList += " --filter `"Category=Showcase`"" }
else { $argList += " --filter `"Category!=Showcase`"" }

$proc = Start-Process dotnet -ArgumentList $argList -PassThru -NoNewWindow
try { $proc.PriorityClass = $pc } catch {}

# testhost spawns a moment after dotnet; force its priority down as well (children
# normally inherit, but enforce it explicitly to win any startup race).
Start-Sleep -Seconds 2
Get-Process testhost, vstest.console -ErrorAction SilentlyContinue | ForEach-Object {
    try { $_.PriorityClass = $pc } catch {}
}

$proc.WaitForExit()
Write-Host "Test process exited with code $($proc.ExitCode) (priority: $Priority)"
exit $proc.ExitCode
