<#
  run_reval.ps1
  Re-evaluates all 14 already-trained experiments to get mAP@0.5 scores.
  Each experiment auto-resumes from its _best.pth checkpoint and runs only
  the final test evaluation pass (no training epochs repeated).
  Skips any experiment that already has map50 in its test_results.json.

  Usage (from project root):
      .\run_reval.ps1
#>

$ErrorActionPreference = "Continue"
$Root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir  = Join-Path $Root "outputs\logs"
$Sep     = "=" * 75

# All 14 already-trained experiments in logical order
$Experiments = @(
    # --- Single-Modal Baselines ---
    @{ Config="configs/baseline_rgb.yaml";                   Name="baseline_rgb"                   },
    @{ Config="configs/baseline_thermal.yaml";               Name="baseline_thermal"               },
    @{ Config="configs/baseline_audio.yaml";                 Name="baseline_audio"                 },

    # --- Fusion Comparisons ---
    @{ Config="configs/early_fusion.yaml";                   Name="early_fusion"                   },
    @{ Config="configs/late_fusion.yaml";                    Name="late_fusion"                    },
    @{ Config="configs/intermediate_fixed.yaml";             Name="intermediate_fixed"             },

    # --- Core AdapFuse Models ---
    @{ Config="configs/adaptfuse_v1.yaml";                   Name="adaptfuse_v1"                   },
    @{ Config="configs/adaptfuse_v1_domain_prompts.yaml";    Name="adaptfuse_v1_domain_prompts"    },
    @{ Config="configs/adaptfuse_v1_gdblock.yaml";           Name="adaptfuse_v1_gdblock"           },
    @{ Config="configs/adaptfuse_v1_panns.yaml";             Name="adaptfuse_v1_panns"             },
    @{ Config="configs/adaptfuse_v1_spatial_attn.yaml";      Name="adaptfuse_v1_spatial_attn"      },
    @{ Config="configs/uni_adaptfuse.yaml";                  Name="uni_adaptfuse"                  },

    # --- Knowledge Distillation ---
    @{ Config="configs/adaptfuse_v2_kd.yaml";                Name="adaptfuse_v2_kd"               },
    @{ Config="configs/adaptfuse_v2_kd_panns.yaml";          Name="adaptfuse_v2_kd_panns"         }
)

$Total   = $Experiments.Count
$Current = 0

Write-Host $Sep
Write-Host "  AdapFuse-UAV: Re-Evaluating $Total Trained Experiments for mAP@0.5"
Write-Host $Sep

foreach ($Exp in $Experiments) {
    $Current++
    $Name        = $Exp.Name
    $Config      = $Exp.Config
    $TestResults = Join-Path $LogDir "${Name}_test_results.json"
    $Log         = Join-Path $LogDir "${Name}_reval.log"

    Write-Host ""
    Write-Host "[$Current/$Total] $Name"

    # Check if map50 already present in existing test_results
    if (Test-Path $TestResults) {
        $content = Get-Content $TestResults -Raw
        if ($content -match '"map50"') {
            Write-Host "  [SKIP] map50 already in test_results.json"
            continue
        }
        Write-Host "  [REVAL] test_results.json exists but no map50 -- re-evaluating..."
    } else {
        Write-Host "  [REVAL] No test_results.json -- running full eval..."
    }

    Write-Host "  Config : $Config"
    Write-Host "  Log    : $Log"
    Write-Host "  Time   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $Cmd = "python training/train.py --config $Config"
    Write-Host "  CMD    : $Cmd"

    $Start = Get-Date
    try {
        Invoke-Expression "$Cmd 2>&1" | Tee-Object -FilePath $Log
        $Elapsed = (Get-Date) - $Start
        Write-Host "  [DONE] Elapsed: $($Elapsed.ToString('hh\:mm\:ss'))"
    } catch {
        Write-Warning "  [ERROR] $Name failed -- check $Log"
    }
}

Write-Host ""
Write-Host $Sep
Write-Host "  ALL RE-EVALUATIONS COMPLETE -- Check outputs/logs/*_test_results.json"
Write-Host $Sep
