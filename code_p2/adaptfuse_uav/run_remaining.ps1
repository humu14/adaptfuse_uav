<#
  run_remaining.ps1
  Runs all 12 remaining experiments (1 incomplete + 11 not started).
  Skips only when BOTH checkpoint AND test_results.json exist (fully done).
  Logs each run to outputs\logs\<experiment_name>_run.log.

  Usage (from project root):
      .\run_remaining.ps1
  Dry-run (5 batches per epoch to validate):
      .\run_remaining.ps1 -DryRunBatches 5
#>

param(
    [int]$DryRunBatches = 0
)

$ErrorActionPreference = "Continue"
$Root          = Split-Path -Parent $MyInvocation.MyCommand.Path
$CheckpointDir = Join-Path $Root "outputs\checkpoints"
$LogDir        = Join-Path $Root "outputs\logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# All 12 remaining experiments
# EXP-22 is listed first: checkpoint exists but test_results.json is missing
$Experiments = @(
    @{ Config="configs/adaptfuse_v1_spatial_attn.yaml";         Name="adaptfuse_v1_spatial_attn"      },
    @{ Config="configs/fusion_rgb_thermal.yaml";                 Name="fusion_rgb_thermal"             },
    @{ Config="configs/fusion_rgb_audio.yaml";                   Name="fusion_rgb_audio"               },
    @{ Config="configs/fusion_thermal_audio.yaml";               Name="fusion_thermal_audio"           },
    @{ Config="configs/adaptfuse_v1_efficientnet_rgb.yaml";      Name="adaptfuse_v1_efficientnet_rgb"  },
    @{ Config="configs/adaptfuse_v1_efficientnet_thermal.yaml";  Name="adaptfuse_v1_efficientnet_thermal" },
    @{ Config="configs/adaptfuse_v1_mobilevit.yaml";             Name="adaptfuse_v1_mobilevit"         },
    @{ Config="configs/ablation_no_rue.yaml";                    Name="ablation_no_rue"                },
    @{ Config="configs/ablation_no_attention.yaml";              Name="ablation_no_attention"          },
    @{ Config="configs/ablation_no_gru.yaml";                    Name="ablation_no_gru"                },
    @{ Config="configs/ablation_no_nuisance.yaml";               Name="ablation_no_nuisance"           },
    @{ Config="configs/ablation_no_quality_tokens.yaml";         Name="ablation_no_quality_tokens"     }
)

$Total   = $Experiments.Count
$Current = 0
$Sep     = "=" * 75

Write-Host $Sep
Write-Host "  AdapFuse-UAV: Running $Total REMAINING Experiments"
Write-Host $Sep

foreach ($Exp in $Experiments) {
    $Current++
    $Name        = $Exp.Name
    $Config      = $Exp.Config
    $Best        = Join-Path $CheckpointDir "${Name}_best.pth"
    $TestResults = Join-Path $LogDir "${Name}_test_results.json"
    $Log         = Join-Path $LogDir "${Name}_run.log"

    Write-Host ""
    Write-Host "[$Current/$Total] $Name"

    # Skip only when fully done (checkpoint + test_results both present)
    if ((Test-Path $Best) -and (Test-Path $TestResults)) {
        Write-Host "  [SKIP] Already complete."
        continue
    }

    if (Test-Path $Best) {
        Write-Host "  [RESUME] Checkpoint found, test_results missing -- will auto-resume and re-evaluate."
    } else {
        Write-Host "  [NEW] Starting fresh."
    }

    Write-Host "  Config : $Config"
    Write-Host "  Log    : $Log"
    Write-Host "  Time   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $Cmd = "python training/train.py --config $Config"
    if ($DryRunBatches -gt 0) {
        $Cmd += " --dry_run_batches $DryRunBatches"
    }
    Write-Host "  CMD    : $Cmd"

    $Start = Get-Date
    try {
        Invoke-Expression "$Cmd 2>&1" | Tee-Object -FilePath $Log
        $Elapsed = (Get-Date) - $Start
        Write-Host "  [DONE] Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))"
    } catch {
        Write-Warning "  [ERROR] $Name failed -- check $Log"
        Write-Warning "  Continuing to next experiment..."
    }
}

Write-Host ""
Write-Host $Sep
Write-Host "  ALL REMAINING EXPERIMENTS FINISHED"
Write-Host $Sep
