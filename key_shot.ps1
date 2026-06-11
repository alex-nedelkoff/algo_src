param([string]$Keys = "{ENTER}")
Add-Type -AssemblyName System.Windows.Forms,System.Drawing,Microsoft.VisualBasic
$p = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue
if ($p) { [Microsoft.VisualBasic.Interaction]::AppActivate($p.Id) | Out-Null; Start-Sleep -Milliseconds 600 }
[System.Windows.Forms.SendKeys]::SendWait($Keys)
Start-Sleep -Milliseconds 2500
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save("C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client\screen.png")
