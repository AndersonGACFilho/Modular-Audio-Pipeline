[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$ProjectPath = "D:\VideoTranscripter",

    [ValidateNotNullOrEmpty()]
    [string]$ModulePath = "app",

    [ValidateRange(1, 102400)]
    [int]$MaxFileSizeKB = 1024,

    [ValidateRange(0, 10240)]
    [int]$MaxTotalSizeMB = 10
)

$consoleUtf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $consoleUtf8
[Console]::OutputEncoding = $consoleUtf8
$OutputEncoding = $consoleUtf8

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ignoreFolders = @(
    "node_modules",
    "dist",
    "build",
    ".angular",
    ".next",
    ".turbo",
    "vendor",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "assets",
    "coverage",
    "htmlcov",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache"
)

$allowedExtensions = @(
    ".py",
    ".ts",
    ".html",
    ".scss",
    ".css",
    ".sql",
    ".ps1",
    ".sh",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".ini",
    ".cfg"
)

$allowedFileNames = @(
    "app.dockerfile",
    "Makefile",
    "Pipfile",
    "requirements.txt",
    "docker-compose.yml",
    "docker-compose.yaml",
    "tsconfig.json",
    "package.json",
    "poetry.toml",
    "pyproject.toml",
    "alembic.ini",
    "pytest.ini",
    ".editorconfig"
)

$ignoredFilePatterns = @(
    "context_project*.txt",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "*.map",
    "*.min.css",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    ".env",
    ".env.*",
    "credentials*.json",
    "service-account*.json",
    "secrets*.json"
)

$script:Utf8WithBom = New-Object System.Text.UTF8Encoding($true)
$script:Utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

function New-CaseInsensitiveSet {
    param(
        [Parameter(Mandatory)]
        [string[]]$Values
    )

    $set = New-Object "System.Collections.Generic.HashSet[string]" (
    [System.StringComparer]::OrdinalIgnoreCase
    )

    foreach ($value in $Values) {
        [void]$set.Add($value)
    }

    return ,$set
}

function Test-MatchesAnyPattern {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string[]]$Patterns
    )

    foreach ($pattern in $Patterns) {
        if ($Name -like $pattern) {
            return $true
        }
    }

    return $false
}

function ConvertTo-SafeFileName {
    param(
        [Parameter(Mandatory)]
        [string]$Name
    )

    $safeName = $Name

    foreach ($invalidCharacter in [System.IO.Path]::GetInvalidFileNameChars()) {
        $safeName = $safeName -replace [Regex]::Escape(
                $invalidCharacter.ToString()
        ), "_"
    }

    $safeName = $safeName.Trim()

    if ([string]::IsNullOrWhiteSpace($safeName)) {
        return "project"
    }

    return $safeName
}

function Get-RelativeProjectPath {
    param(
        [Parameter(Mandatory)]
        [string]$FullPath
    )

    $relativePath = $FullPath.Substring(
            $script:NormalizedRootPath.Length
    )

    return $relativePath.TrimStart(
            [char[]]"\/"
    ).Replace("\", "/")
}

function Get-Utf8ByteCount {
    param(
        [AllowEmptyString()]
        [string]$Text
    )

    if ($null -eq $Text) {
        return [long]0
    }

    return [long]$script:Utf8WithoutBom.GetByteCount($Text)
}

function New-FileBlock {
    param(
        [Parameter(Mandatory)]
        [string]$RelativePath,

        [AllowEmptyString()]
        [string]$Content
    )

    $builder = New-Object System.Text.StringBuilder

    [void]$builder.AppendLine(("=" * 100))
    [void]$builder.AppendLine("FILE: $RelativePath")
    [void]$builder.AppendLine(("=" * 100))

    if ([string]::IsNullOrWhiteSpace($Content)) {
        [void]$builder.AppendLine("[EMPTY FILE]")
    }
    else {
        [void]$builder.AppendLine($Content)
    }

    [void]$builder.AppendLine()
    [void]$builder.AppendLine("--- END OF FILE: $RelativePath ---")
    [void]$builder.AppendLine()

    return $builder.ToString()
}

function New-PartText {
    param(
        [Parameter(Mandatory)]
        [int]$PartNumber,

        [Parameter(Mandatory)]
        [int]$PartCount,

        [Parameter(Mandatory)]
        [System.Collections.IList]$Records,

        [Parameter(Mandatory)]
        [string]$OutputFileName,

        [Parameter(Mandatory)]
        [bool]$IsLastPart
    )

    $builder = New-Object System.Text.StringBuilder
    $sourceBytesInPart = [long]0

    foreach ($record in $Records) {
        $sourceBytesInPart += [long]$record.SourceBytes
    }

    [void]$builder.AppendLine("# PROJECT CONTEXT")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
    [void]$builder.AppendLine("Project root: $script:NormalizedProjectPath")
    [void]$builder.AppendLine("Collected directory: $script:NormalizedRootPath")
    [void]$builder.AppendLine("Module: $script:LastFolderName")
    [void]$builder.AppendLine("Output file: $OutputFileName")

    if ($PartCount -gt 1) {
        [void]$builder.AppendLine("Part: $PartNumber of $PartCount")
    }

    [void]$builder.AppendLine("Files in this part: $($Records.Count)")
    [void]$builder.AppendLine(
            "Source file size in this part: $([Math]::Round($sourceBytesInPart / 1MB, 2)) MB"
    )

    if ($script:MaxPartBytes -eq [long]::MaxValue) {
        [void]$builder.AppendLine("Limit per part: unlimited")
    }
    else {
        [void]$builder.AppendLine("Limit per part: $script:ConfiguredMaxTotalSizeMB MB")
    }

    [void]$builder.AppendLine()

    foreach ($record in $Records) {
        [void]$builder.Append($record.Block)
    }

    [void]$builder.AppendLine(("=" * 100))
    [void]$builder.AppendLine("PART SUMMARY")
    [void]$builder.AppendLine(("=" * 100))
    [void]$builder.AppendLine("Part: $PartNumber of $PartCount")
    [void]$builder.AppendLine("Files in this part: $($Records.Count)")

    if ($IsLastPart) {
        [void]$builder.AppendLine()
        [void]$builder.AppendLine(("=" * 100))
        [void]$builder.AppendLine("OVERALL GENERATION SUMMARY")
        [void]$builder.AppendLine(("=" * 100))
        [void]$builder.AppendLine("Files found: $($script:Stats['ScannedFiles'])")
        [void]$builder.AppendLine("Files included: $($script:Stats['IncludedFiles'])")
        [void]$builder.AppendLine("Parts generated: $PartCount")
        [void]$builder.AppendLine("Ignored directories: $($script:Stats['IgnoredDirectories'])")
        [void]$builder.AppendLine("Ignored by extension: $($script:Stats['IgnoredByExtension'])")
        [void]$builder.AppendLine("Ignored by file name pattern: $($script:Stats['IgnoredByPattern'])")
        [void]$builder.AppendLine("Ignored by individual size: $($script:Stats['IgnoredByFileSize'])")
        [void]$builder.AppendLine("Ignored symbolic links: $($script:Stats['IgnoredSymbolicLinks'])")
        [void]$builder.AppendLine("Enumeration errors: $($script:Stats['EnumerationErrors'])")
        [void]$builder.AppendLine("Read errors: $($script:Stats['ReadErrors'])")

        if ($script:EnumerationErrors.Count -gt 0) {
            [void]$builder.AppendLine()
            [void]$builder.AppendLine("ENUMERATION ERRORS:")

            foreach ($errorItem in $script:EnumerationErrors) {
                [void]$builder.AppendLine("- $errorItem")
            }
        }

        if ($script:ReadErrors.Count -gt 0) {
            [void]$builder.AppendLine()
            [void]$builder.AppendLine("READ ERRORS:")

            foreach ($errorItem in $script:ReadErrors) {
                [void]$builder.AppendLine("- $errorItem")
            }
        }
    }

    return $builder.ToString()
}

$expandedProjectPath = [Environment]::ExpandEnvironmentVariables(
        $ProjectPath.Trim().Trim('"')
)

if (-not [System.IO.Path]::IsPathRooted($expandedProjectPath)) {
    $expandedProjectPath = Join-Path `
        -Path $PSScriptRoot `
        -ChildPath $expandedProjectPath
}

if (-not (Test-Path -LiteralPath $expandedProjectPath -PathType Container)) {
    throw "The project root does not exist: $expandedProjectPath"
}

$script:NormalizedProjectPath = (
Get-Item -LiteralPath $expandedProjectPath
).FullName.TrimEnd([char[]]"\/")

$cleanModulePath = [Environment]::ExpandEnvironmentVariables(
        $ModulePath.Trim().Trim('"')
)

if ([System.IO.Path]::IsPathRooted($cleanModulePath)) {
    throw "ModulePath must be relative to ProjectPath: $cleanModulePath"
}

$cleanModulePath = $cleanModulePath.TrimStart([char[]]"\/")

$combinedRootPath = Join-Path `
    -Path $script:NormalizedProjectPath `
    -ChildPath $cleanModulePath

if (-not (Test-Path -LiteralPath $combinedRootPath -PathType Container)) {
    throw (
    "The directory specified in ModulePath does not exist within ProjectPath." +
            [Environment]::NewLine +
            "ProjectPath: $script:NormalizedProjectPath" +
            [Environment]::NewLine +
            "ModulePath: $cleanModulePath" +
            [Environment]::NewLine +
            "Final path: $combinedRootPath"
    )
}

$rootItem = Get-Item -LiteralPath $combinedRootPath

$script:NormalizedRootPath = $rootItem.FullName.TrimEnd(
        [char[]]"\/"
)

$script:LastFolderName = Split-Path `
    -Path $script:NormalizedRootPath `
    -Leaf

$safeFolderName = ConvertTo-SafeFileName `
    -Name $script:LastFolderName

$outputBaseName = "context_project_$safeFolderName"
$outputDirectory = $script:NormalizedProjectPath

Get-ChildItem `
    -LiteralPath $outputDirectory `
    -File `
    -Force `
    -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "${outputBaseName}*.txt"
        } |
        Remove-Item -Force -ErrorAction SilentlyContinue

$ignoredDirectorySet = New-CaseInsensitiveSet `
    -Values $ignoreFolders

$allowedExtensionSet = New-CaseInsensitiveSet `
    -Values $allowedExtensions

$allowedFileNameSet = New-CaseInsensitiveSet `
    -Values $allowedFileNames

$maxFileSizeBytes = [long]$MaxFileSizeKB * 1KB
$script:ConfiguredMaxTotalSizeMB = $MaxTotalSizeMB

if ($MaxTotalSizeMB -gt 0) {
    $script:MaxPartBytes = [long]$MaxTotalSizeMB * 1MB
}
else {
    $script:MaxPartBytes = [long]::MaxValue
}

$reservedMetadataBytes = [long](64KB)

if (
$script:MaxPartBytes -ne [long]::MaxValue `
        -and $script:MaxPartBytes -le $reservedMetadataBytes
) {
    throw "MaxTotalSizeMB must be greater than 0.0625 MB."
}

if ($script:MaxPartBytes -eq [long]::MaxValue) {
    $targetPayloadBytes = [long]::MaxValue
}
else {
    $targetPayloadBytes = $script:MaxPartBytes - $reservedMetadataBytes
}

$script:Stats = [ordered]@{
    ScannedFiles         = 0
    IncludedFiles        = 0
    IgnoredDirectories   = 0
    IgnoredByExtension   = 0
    IgnoredByPattern     = 0
    IgnoredByFileSize    = 0
    IgnoredSymbolicLinks = 0
    EnumerationErrors    = 0
    ReadErrors           = 0
}

$script:EnumerationErrors = New-Object "System.Collections.Generic.List[string]"
$script:ReadErrors = New-Object "System.Collections.Generic.List[string]"
$candidateFiles = New-Object "System.Collections.Generic.List[System.IO.FileInfo]"

$directoryStack = New-Object "System.Collections.Generic.Stack[System.IO.DirectoryInfo]"
$directoryStack.Push([System.IO.DirectoryInfo]$rootItem)

while ($directoryStack.Count -gt 0) {
    $currentDirectory = $directoryStack.Pop()

    try {
        $items = Get-ChildItem `
            -LiteralPath $currentDirectory.FullName `
            -Force `
            -ErrorAction Stop
    }
    catch {
        $script:Stats["EnumerationErrors"]++
        $script:EnumerationErrors.Add(
                "$($currentDirectory.FullName): $($_.Exception.Message)"
        )
        continue
    }

    foreach ($item in $items) {
        if ($item.PSIsContainer) {
            if ($ignoredDirectorySet.Contains($item.Name)) {
                $script:Stats["IgnoredDirectories"]++
                continue
            }

            if (
            ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) `
                    -ne 0
            ) {
                $script:Stats["IgnoredSymbolicLinks"]++
                continue
            }

            $directoryStack.Push([System.IO.DirectoryInfo]$item)
            continue
        }

        $script:Stats["ScannedFiles"]++
        $file = [System.IO.FileInfo]$item

        if (
        Test-MatchesAnyPattern `
                -Name $file.Name `
                -Patterns $ignoredFilePatterns
        ) {
            $script:Stats["IgnoredByPattern"]++
            continue
        }

        $isAllowedExtension = $allowedExtensionSet.Contains(
                $file.Extension
        )

        $isAllowedFileName = $allowedFileNameSet.Contains(
                $file.Name
        )

        if (-not ($isAllowedExtension -or $isAllowedFileName)) {
            $script:Stats["IgnoredByExtension"]++
            continue
        }

        if ($file.Length -gt $maxFileSizeBytes) {
            $script:Stats["IgnoredByFileSize"]++
            continue
        }

        $candidateFiles.Add($file)
    }
}

$sortedFiles = @(
$candidateFiles |
        Sort-Object {
            Get-RelativeProjectPath -FullPath $_.FullName
        }
)

$fileRecords = New-Object System.Collections.ArrayList
$totalCandidateFiles = $sortedFiles.Count
$currentFileNumber = 0

foreach ($file in $sortedFiles) {
    $currentFileNumber++

    if ($totalCandidateFiles -gt 0) {
        $percentage = [Math]::Floor(
                ($currentFileNumber / $totalCandidateFiles) * 100
        )

        Write-Progress `
            -Activity "Reading project files" `
            -Status "$currentFileNumber of $totalCandidateFiles files" `
            -PercentComplete $percentage
    }

    $relativePath = Get-RelativeProjectPath `
        -FullPath $file.FullName

    try {
        $content = [System.IO.File]::ReadAllText(
                $file.FullName
        )

        $block = New-FileBlock `
            -RelativePath $relativePath `
            -Content $content

        $record = [PSCustomObject]@{
            FullName     = $file.FullName
            RelativePath = $relativePath
            SourceBytes  = [long]$file.Length
            BlockBytes   = [long](Get-Utf8ByteCount -Text $block)
            Block        = $block
        }

        [void]$fileRecords.Add($record)
        $script:Stats["IncludedFiles"]++
    }
    catch {
        $script:Stats["ReadErrors"]++
        $script:ReadErrors.Add(
                "${relativePath}: $($_.Exception.Message)"
        )
    }
}

Write-Progress `
    -Activity "Reading project files" `
    -Completed

$parts = New-Object System.Collections.ArrayList
$currentPart = New-Object System.Collections.ArrayList
$currentPayloadBytes = [long]0

foreach ($record in $fileRecords) {
    $wouldExceedTarget = (
    $currentPart.Count -gt 0 `
            -and $targetPayloadBytes -ne [long]::MaxValue `
            -and (
    $currentPayloadBytes + [long]$record.BlockBytes
    ) -gt $targetPayloadBytes
    )

    if ($wouldExceedTarget) {
        [void]$parts.Add($currentPart)
        $currentPart = New-Object System.Collections.ArrayList
        $currentPayloadBytes = [long]0
    }

    [void]$currentPart.Add($record)
    $currentPayloadBytes += [long]$record.BlockBytes
}

if ($currentPart.Count -gt 0) {
    [void]$parts.Add($currentPart)
}

if ($parts.Count -eq 0) {
    [void]$parts.Add(
            (New-Object System.Collections.ArrayList)
    )
}

if ($script:MaxPartBytes -ne [long]::MaxValue) {
    $changed = $true
    $safetyCounter = 0

    while ($changed) {
        $changed = $false
        $safetyCounter++

        if ($safetyCounter -gt 100000) {
            throw "Unable to stabilize the file split."
        }

        $partCount = $parts.Count

        for ($partIndex = 0; $partIndex -lt $parts.Count; $partIndex++) {
            $partNumber = $partIndex + 1
            $isLastPart = $partNumber -eq $partCount

            $temporaryOutputName = (
            "{0}_part_{1}_of_{2}.txt" -f `
                    $outputBaseName,
            $partNumber.ToString("000"),
            $partCount.ToString("000")
            )

            $temporaryText = New-PartText `
                -PartNumber $partNumber `
                -PartCount $partCount `
                -Records $parts[$partIndex] `
                -OutputFileName $temporaryOutputName `
                -IsLastPart $isLastPart

            $actualBytes = (
            Get-Utf8ByteCount -Text $temporaryText
            ) + $script:Utf8WithBom.GetPreamble().Length

            if (
            $actualBytes -gt $script:MaxPartBytes `
                    -and $parts[$partIndex].Count -gt 1
            ) {
                $lastRecordIndex = $parts[$partIndex].Count - 1
                $recordToMove = $parts[$partIndex][$lastRecordIndex]
                $parts[$partIndex].RemoveAt($lastRecordIndex)

                if (($partIndex + 1) -lt $parts.Count) {
                    $parts[$partIndex + 1].Insert(
                            0,
                            $recordToMove
                    )
                }
                else {
                    $newPart = New-Object System.Collections.ArrayList
                    [void]$newPart.Add($recordToMove)
                    [void]$parts.Add($newPart)
                }

                $changed = $true
                break
            }
        }
    }
}

$outputFiles = New-Object System.Collections.ArrayList
$oversizedParts = New-Object System.Collections.ArrayList
$finalPartCount = $parts.Count

for ($partIndex = 0; $partIndex -lt $finalPartCount; $partIndex++) {
    $partNumber = $partIndex + 1
    $isLastPart = $partNumber -eq $finalPartCount

    if ($finalPartCount -eq 1) {
        $outputFileName = "${outputBaseName}.txt"
    }
    else {
        $outputFileName = (
        "{0}_part_{1}_of_{2}.txt" -f `
                $outputBaseName,
        $partNumber.ToString("000"),
        $finalPartCount.ToString("000")
        )
    }

    $outputPath = Join-Path `
        -Path $outputDirectory `
        -ChildPath $outputFileName

    $partText = New-PartText `
        -PartNumber $partNumber `
        -PartCount $finalPartCount `
        -Records $parts[$partIndex] `
        -OutputFileName $outputFileName `
        -IsLastPart $isLastPart

    [System.IO.File]::WriteAllText(
            $outputPath,
            $partText,
            $script:Utf8WithBom
    )

    $outputInfo = Get-Item -LiteralPath $outputPath

    $outputRecord = [PSCustomObject]@{
        PartNumber = $partNumber
        Path       = $outputPath
        Name       = $outputFileName
        SizeBytes  = [long]$outputInfo.Length
        FileCount  = $parts[$partIndex].Count
    }

    [void]$outputFiles.Add($outputRecord)

    if (
    $script:MaxPartBytes -ne [long]::MaxValue `
            -and $outputInfo.Length -gt $script:MaxPartBytes
    ) {
        [void]$oversizedParts.Add($outputRecord)
    }
}

Write-Host ""
Write-Host "Context generated successfully." -ForegroundColor Green
Write-Host "Project root: $script:NormalizedProjectPath"
Write-Host "Collected directory: $script:NormalizedRootPath"
Write-Host "Module: $script:LastFolderName"
Write-Host "Files found: $($script:Stats['ScannedFiles'])"
Write-Host "Files included: $($script:Stats['IncludedFiles'])"
Write-Host "Parts generated: $finalPartCount"

if ($MaxTotalSizeMB -gt 0) {
    Write-Host "Limit per part: $MaxTotalSizeMB MB"
}
else {
    Write-Host "Limit per part: unlimited"
}

Write-Host ""

foreach ($outputFile in $outputFiles) {
    $sizeMB = [Math]::Round(
            $outputFile.SizeBytes / 1MB,
            2
    )

    Write-Host (
    "[{0}/{1}] {2} - {3} MB - {4} file(s)" -f `
            $outputFile.PartNumber,
    $finalPartCount,
    $outputFile.Path,
    $sizeMB,
    $outputFile.FileCount
    )
}

if ($oversizedParts.Count -gt 0) {
    Write-Warning (
    "$($oversizedParts.Count) part(s) exceeded the limit because " +
            "a single source file did not fit in one part. " +
            "No files were discarded."
    )
}

if (
$script:Stats["ReadErrors"] -gt 0 `
        -or $script:Stats["EnumerationErrors"] -gt 0
) {
    Write-Warning (
    "Generation completed with read or enumeration errors. " +
            "See the summary in the last part."
    )
}

if ($script:Stats["IgnoredByFileSize"] -gt 0) {
    Write-Warning (
    "$($script:Stats['IgnoredByFileSize']) file(s) were ignored " +
            "because they exceed MaxFileSizeKB=$MaxFileSizeKB."
    )
}
