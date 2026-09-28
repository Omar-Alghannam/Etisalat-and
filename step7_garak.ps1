Write-Host '============================================================'
Write-Host 'Step 7 - Garak LLM Red-Teaming (Fast 1-2 Min Scan)'
Write-Host '============================================================'
Write-Host 'Testing against server running on http://localhost:8000/api/chat'
Write-Host 'Probing for: DAN Jailbreaks, Developer Mode Bypasses, & Base64 Injections...'
Write-Host '============================================================'

python -m garak --target_type rest -G garak_config.json --probes dan.Dan_11_0,dan.ChatGPT_Developer_Mode_v2,encoding.InjectBase64 --generations 1 --report_prefix garak_report

