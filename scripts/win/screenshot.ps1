# Capture the primary screen of the interactive desktop session to a PNG.
# Must run as an /it (interactive) scheduled task -- a plain ssh spawn is headless and captures black.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save("C:\Users\alexj\Documents\aigp-waypoints\screen.png")
$g.Dispose(); $bmp.Dispose()
