<#
  uart_rt_oneshot_boot.ps1 -- boot an ALTERNATIVE kernel image ONCE on the Z-Turn
  through the U-Boot prompt on the serial console (COM6), capturing the full log.

  - Never runs saveenv and never edits U-Boot variables: the command typed is only
    `fatload mmc 0 <addr> <image> && bootm <addr>`. Any later reboot / power cycle
    boots the default image.ub again.
  - With -NoInterrupt it does not touch autoboot at all (normal default boot, log only).
  - Stops autoboot by sending a key once "Hit any key to stop autoboot" appears.

  Example:
    powershell -File tools\uart_rt_oneshot_boot.ps1 -Image image_rt.ub `
       -OutFile boot_rt.log -MetaFile boot_rt.meta.json `
       -RebootCmd "ssh -i %USERPROFILE%\.ssh\id_zturn -o BatchMode=yes root@192.168.2.99 \"sync; reboot\""
#>
param(
    [string]$Port = 'COM6',
    [int]$Baud = 115200,
    [string]$Image = 'image_rt.ub',
    [string]$LoadAddr = '0x10000000',
    [Parameter(Mandatory = $true)][string]$OutFile,
    [Parameter(Mandatory = $true)][string]$MetaFile,
    [Parameter(Mandatory = $true)][string]$RebootCmd,
    [int]$TimeoutSeconds = 300,
    [string]$DonePattern = 'login:',
    [switch]$NoInterrupt
)
if ($NoInterrupt) { $Image = '' }
$ErrorActionPreference = 'Stop'

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
$sp.Handshake = [System.IO.Ports.Handshake]::None
$sp.ReadTimeout = 500
$sp.DtrEnable = $true
$sp.RtsEnable = $true

$bootCmd = if ($Image) { "fatload mmc 0 $LoadAddr $Image && bootm $LoadAddr" } else { '' }
$meta = [ordered]@{
    port = $Port; baud = $Baud; image = $Image; boot_cmd = $bootCmd
    saveenv_used = $false; reboot_cmd = $RebootCmd
    opened_utc = $null; trigger_utc = $null; trigger_exit_code = $null
    autoboot_prompt_utc = $null; uboot_prompt_utc = $null; boot_cmd_sent_utc = $null
    starting_kernel_utc = $null; done_utc = $null; stopped_reason = 'timeout'; total_chars = 0
}

$sw = New-Object System.IO.StreamWriter($OutFile, $false, (New-Object System.Text.UTF8Encoding($false)))
$acc = New-Object System.Text.StringBuilder
function Now { (Get-Date).ToUniversalTime().ToString('o') }
function Pump {
    try { $c = $sp.ReadExisting() } catch { $c = '' }
    if ($c) { $sw.Write($c); $sw.Flush(); [void]$acc.Append($c); $script:meta.total_chars += $c.Length; return $true }
    return $false
}

$sp.Open(); $meta.opened_utc = Now
$end = (Get-Date).AddSeconds(1); while ((Get-Date) -lt $end) { if (-not (Pump)) { Start-Sleep -Milliseconds 50 } }

$meta.trigger_utc = Now
# ssh prints "connection closed by remote host" on stderr when the board reboots;
# that is expected and must not abort the capture.
$ErrorActionPreference = 'Continue'
try { & cmd.exe /c "$RebootCmd 2>nul" | Out-Null } catch { }
$meta.trigger_exit_code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
[void]$acc.Clear()

$state = if ($Image) { 'wait_autoboot' } else { 'wait_done' }
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $got = Pump
    $text = $acc.ToString()
    if ($state -eq 'wait_autoboot' -and $text -match 'Hit any key to stop autoboot') {
        $sp.Write(' '); $meta.autoboot_prompt_utc = Now; $state = 'wait_prompt'
        [void]$acc.Clear(); continue
    }
    if ($state -eq 'wait_prompt' -and $text -match '[A-Za-z\-]*>\s*$') {
        $meta.uboot_prompt_utc = Now
        Start-Sleep -Milliseconds 300; [void](Pump)
        $sp.Write("$bootCmd`r"); $meta.boot_cmd_sent_utc = Now; $state = 'wait_kernel'
        [void]$acc.Clear(); continue
    }
    if ($state -eq 'wait_kernel' -and $text -match 'Starting kernel') {
        $meta.starting_kernel_utc = Now; $state = 'wait_done'; continue
    }
    if ($state -eq 'wait_done' -and $text -match $DonePattern) {
        $meta.done_utc = Now; $meta.stopped_reason = 'done_pattern'
        $e2 = (Get-Date).AddSeconds(3); while ((Get-Date) -lt $e2) { if (-not (Pump)) { Start-Sleep -Milliseconds 50 } }
        break
    }
    if (-not $got) { Start-Sleep -Milliseconds 30 }
}
if ($meta.stopped_reason -eq 'timeout') { $meta.stopped_reason = "timeout_in_state_$state" }
$sw.Close(); $sp.Close()
$meta | ConvertTo-Json | Out-File -FilePath $MetaFile -Encoding utf8
Write-Output "DONE reason=$($meta.stopped_reason) chars=$($meta.total_chars)"
