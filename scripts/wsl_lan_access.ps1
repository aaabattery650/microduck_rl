# microduck LAN access — run in ADMIN PowerShell.
# Forwards the WSL web UIs to your LAN so any device (phone, laptop) can open
# them, and opens the firewall for those ports. Safe to re-run any time.
#
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_lan_access.ps1
#
# Forwards: 8765 (train dashboard), 8766 (reward lab), 8767 (lab 3D),
#           8080 (policy viewer). Rerun after a PC reboot (WSL's IP changes).

$ErrorActionPreference = "Stop"

$wslIp = (wsl -d Ubuntu-24.04 hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) { throw "Could not read the WSL IP - is Ubuntu-24.04 running?" }
Write-Host "WSL IP: $wslIp"

$ports = 8765, 8766, 8767, 8080
foreach ($p in $ports) {
    netsh interface portproxy delete v4tov4 listenport=$p listenaddress=0.0.0.0 2>$null
    netsh interface portproxy add v4tov4 listenport=$p listenaddress=0.0.0.0 `
        connectport=$p connectaddress=$wslIp | Out-Null
    netsh advfirewall firewall delete rule name="microduck $p" 2>$null
    netsh advfirewall firewall add rule name="microduck $p" dir=in action=allow `
        protocol=TCP localport=$p | Out-Null
}

$lan = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and
                   $_.IPAddress -notlike "172.1[6-9].*" -and $_.IPAddress -notlike "172.2.*" -and
                   $_.IPAddress -notlike "172.3[0-1].*" } |
    Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "Open from ANY device on your network (phone/laptop browser):"
foreach ($p in $ports) { Write-Host "  http://${lan}:$p" }
Write-Host ""
Write-Host "(8765 = training dashboard, 8766 = reward lab, 8767 = lab 3D, 8080 = policy viewer)"
Write-Host "Rerun this script after every PC reboot - the WSL IP changes."
