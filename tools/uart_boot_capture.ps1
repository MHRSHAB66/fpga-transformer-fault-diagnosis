<#
  uart_boot_capture.ps1 -- capture the Z-Turn (XC7Z020 / PYNQ 2.4) serial console
  over USB-UART (CP210x) while optionally triggering a single reboot over SSH.

  Read-only by default: without -RebootCmd it just opens the port and records
  whatever the console emits (useful to validate the capture path at a prompt).

  All console bytes are written verbatim to -OutFile (UTF-8, no re-encoding of the
  framing). Start/stop timestamps and the trigger action go to -MetaFile as JSON.

  Example (validate, no reboot):
    powershell -File uart_boot_capture.ps1 -Port COM6 -OutFile boot.log `
        -MetaFile boot.meta.json -DurationSeconds 5 -SendOnOpen "`n"

  Example (single reboot + capture to login prompt):
    powershell -File uart_boot_capture.ps1 -Port COM6 -OutFile boot.log `
        -MetaFile boot.meta.json -DurationSeconds 180 -StopPattern "login:" `
        -RebootCmd "ssh -i $HOME\.ssh\id_zturn -o StrictHostKeyChecking=no root@192.168.2.99 'sync; reboot'"
#>
param(
    [string]$Port = 'COM6',
    [int]$Baud = 115200,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [Parameter(Mandatory = $true)][string]$MetaFile,
    [int]$DurationSeconds = 180,
    [string]$StopPattern = '',          # regex; capture stops early when matched
    [int]$PreTriggerSeconds = 2,        # read this long before firing -RebootCmd
    [string]$SendOnOpen = '',           # literal bytes to write right after open
    [string]$RebootCmd = ''             # shell command to trigger reboot (optional)
)

$ErrorActionPreference = 'Stop'

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
$sp.Handshake = [System.IO.Ports.Handshake]::None
$sp.ReadTimeout = 500
$sp.NewLine = "`n"
$sp.DtrEnable = $true
$sp.RtsEnable = $true

$meta = [ordered]@{
    port              = $Port
    baud              = $Baud
    data_bits         = 8
    parity            = 'none'
    stop_bits         = 1
    flow_control      = 'none'
    out_file          = $OutFile
    duration_seconds  = $DurationSeconds
    stop_pattern      = $StopPattern
    reboot_cmd        = $RebootCmd
    opened_utc        = $null
    trigger_sent_utc  = $null
    trigger_exit_code = $null
    first_byte_utc    = $null
    stop_match_utc    = $null
    closed_utc        = $null
    stopped_reason    = $null
    total_bytes       = 0
}

$sp.Open()
$meta.opened_utc = (Get-Date).ToUniversalTime().ToString('o')

$sw = New-Object System.IO.StreamWriter($OutFile, $false, (New-Object System.Text.UTF8Encoding($false)))
$acc = New-Object System.Text.StringBuilder
$total = 0
$firstByte = $false

function Read-Available {
    try { $chunk = $sp.ReadExisting() } catch { $chunk = '' }
    if ($chunk -and $chunk.Length -gt 0) {
        $sw.Write($chunk); $sw.Flush()
        [void]$acc.Append($chunk)
        $script:total += $chunk.Length
        if (-not $script:firstByte) {
            $meta.first_byte_utc = (Get-Date).ToUniversalTime().ToString('o')
            $script:firstByte = $true
        }
        return $true
    }
    return $false
}

if ($SendOnOpen) { $sp.Write($SendOnOpen) }

# Pre-trigger window: capture anything already on the wire.
$preEnd = (Get-Date).AddSeconds($PreTriggerSeconds)
while ((Get-Date) -lt $preEnd) { if (-not (Read-Available)) { Start-Sleep -Milliseconds 80 } }

# Fire the (single) reboot trigger if requested.
if ($RebootCmd) {
    $meta.trigger_sent_utc = (Get-Date).ToUniversalTime().ToString('o')
    try {
        & cmd.exe /c $RebootCmd 2>&1 | Out-Null
        $meta.trigger_exit_code = $LASTEXITCODE
    } catch {
        $meta.trigger_exit_code = -1
    }
    # Only match the end-of-boot stop pattern against post-reboot output, so a
    # pre-reboot shell prompt already on the wire cannot end the capture early.
    [void]$acc.Clear()
}

# Main capture window.
$deadline = (Get-Date).AddSeconds($DurationSeconds)
$meta.stopped_reason = 'timeout'
while ((Get-Date) -lt $deadline) {
    $got = Read-Available
    if ($StopPattern -and ($acc.ToString() -match $StopPattern)) {
        $meta.stop_match_utc = (Get-Date).ToUniversalTime().ToString('o')
        $meta.stopped_reason = 'stop_pattern'
        # drain a little more so the matched prompt line is fully captured
        $drainEnd = (Get-Date).AddSeconds(2)
        while ((Get-Date) -lt $drainEnd) { if (-not (Read-Available)) { Start-Sleep -Milliseconds 80 } }
        break
    }
    if (-not $got) { Start-Sleep -Milliseconds 80 }
}

$sw.Close()
$sp.Close()
$meta.closed_utc = (Get-Date).ToUniversalTime().ToString('o')
$meta.total_bytes = $total
$meta | ConvertTo-Json | Out-File -FilePath $MetaFile -Encoding utf8
Write-Output "DONE reason=$($meta.stopped_reason) bytes=$total out=$OutFile"
