# Keeps one LLM feedback run going until it finishes, independent of any
# Claude Code / VS Code session. Resumes every 30 minutes after a quota stop.
#   powershell -File scripts\run_llm_loop.ps1 -Model openai/gpt-oss-120b -Log results\v2\llm_feedback_log_gptoss120b_v2.txt
#   powershell -File scripts\run_llm_loop.ps1 -Model qwen/qwen3.8-27b -Log results\v2\llm_feedback_log_qwen_extra.txt -Extra noisy_vote2,noisy_verify
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$Extra = ""
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$tag = $Model.Split("/")[-1]
$runs = "results\v2\llm_feedback_runs_${tag}_v2.jsonl"
if ($Extra) { $env:LLM_EXTRA_CONDITIONS = $Extra }

function Finished {
    if ($Extra) {
        # follow-up conditions: done once every failing problem has all of them
        $failing = python -c "import json; print(sum(not json.loads(l)['visible_pass'] for l in open('results/v2/llm_feedback_first_${tag}_v2.jsonl', encoding='utf-8')))"
        $need = [int]$failing * $Extra.Split(",").Count
        $n = (Select-String -Path $runs -Pattern ('"condition": "(' + ($Extra -replace ",", "|") + ')"')).Count
        return $n -ge $need
    }
    # PowerShell 5.1 appends with *>> in UTF-16, so the log mixes encodings;
    # strip NUL bytes before searching or "ALL DONE" is never found.
    $text = [IO.File]::ReadAllText((Resolve-Path $Log)) -replace "`0", ""
    return $text.Contains("ALL DONE")
}

for ($i = 0; $i -lt 300; $i++) {
    if (Finished) { Add-Content $Log "LOOP: finished"; break }
    python -m experiments.llm_feedback_experiment $Model *>> $Log
    if (Finished) { Add-Content $Log "LOOP: finished"; break }
    Start-Sleep -Seconds 1800
}
