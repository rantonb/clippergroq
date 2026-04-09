$body = @{
    contents = @(
        @{
            parts = @(
                @{ text = "Halo, apakah kunci ini aktif?" }
            )
        }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyBB9VOqyS0AD29mpXTQ4llIngQYxhMeKhg" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body