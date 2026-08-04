# Download OPUS moses bitexts for FR→(sw/ln/kg/lu)
# Usage:
#   .\dataset\download.ps1
#   .\dataset\download.ps1 -IncludeNllbSw

param(
  [switch]$IncludeNllbSw,
  [switch]$SkipUnzip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Ensure-Dir([string]$Path) {
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Download-File([string]$Url, [string]$Dest) {
  if ((Test-Path $Dest) -and (Get-Item $Dest).Length -gt 0) {
    Write-Host "SKIP  $(Split-Path $Dest -Leaf)"
    return
  }
  Ensure-Dir (Split-Path $Dest -Parent)
  Write-Host "GET   $(Split-Path $Dest -Leaf)"
  & curl.exe -L --fail --retry 3 --silent --show-error -o $Dest $Url
  if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
}

$downloads = @(
  @{ Pair = "fr-ln"; File = "NLLB-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-ln"; File = "MultiCCAligned-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-ln"; File = "tico-19-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-tico-19/v2020-10-28/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-ln"; File = "wikimedia-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-ln"; File = "Tatoeba-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-ln"; File = "XLEnt-fr-ln.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/fr-ln.txt.zip" },
  @{ Pair = "fr-kg"; File = "NLLB-fr-kg.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/fr-kg.txt.zip" },
  @{ Pair = "fr-kg"; File = "GNOME-fr-kg.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/fr-kg.txt.zip" },
  @{ Pair = "fr-kg"; File = "wikimedia-fr-kg.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/fr-kg.txt.zip" },
  @{ Pair = "fr-lu"; File = "NLLB-fr-lua.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/fr-lua.txt.zip" },
  @{ Pair = "fr-sw"; File = "Tatoeba-fr-swc.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/fr-swc.txt.zip" },
  @{ Pair = "fr-sw"; File = "GlobalVoices-fr-sw.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/fr-sw.txt.zip" },
  @{ Pair = "fr-sw"; File = "TED2020-fr-sw.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/fr-sw.txt.zip" },
  @{ Pair = "fr-sw"; File = "tico-19-fr-sw.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-tico-19/v2020-10-28/moses/fr-sw.txt.zip" },
  @{ Pair = "fr-sw"; File = "wikimedia-fr-sw.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/fr-sw.txt.zip" }
)

if ($IncludeNllbSw) {
  $downloads += @{ Pair = "fr-sw"; File = "NLLB-fr-sw.txt.zip"; Url = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/fr-sw.txt.zip" }
}

foreach ($d in $downloads) {
  $dest = Join-Path $Root "$($d.Pair)\raw\$($d.File)"
  Download-File $d.Url $dest
}

if (-not $SkipUnzip) {
  foreach ($pair in @("fr-sw", "fr-ln", "fr-kg", "fr-lu")) {
    $raw = Join-Path $Root "$pair\raw"
    $moses = Join-Path $Root "$pair\moses"
    Ensure-Dir $moses
    Get-ChildItem $raw -Filter *.zip -ErrorAction SilentlyContinue | ForEach-Object {
      $out = Join-Path $moses $_.BaseName
      Ensure-Dir $out
      Write-Host "UNZIP $($_.Name)"
      Expand-Archive -Force -Path $_.FullName -DestinationPath $out
    }
  }
}

Write-Host "Done."
