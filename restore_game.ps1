param([string]$Keys = "{ENTER}")
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@
$p = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue
if ($p -and $p.MainWindowHandle -ne 0) {
    [Win32]::ShowWindowAsync($p.MainWindowHandle, 9) | Out-Null   # SW_RESTORE
    Start-Sleep -Milliseconds 1500
    [Win32]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 1000
    [System.Windows.Forms.SendKeys]::SendWait($Keys)
}
Start-Sleep -Milliseconds 3000
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save("C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client\screen.png")
