"""
audio_pipeline/integrate.py

Automated Integration Script for Audio Pipeline v2.1

This script automatically integrates all fixes and new features:
- Fixed Silero VAD
- Fixed config type hints
- Hybrid LLM post-processing
- FasterWhisper support
- Optimized diarization
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse


class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_step(message: str):
    """Print step message."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}➜ {message}{Colors.END}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def find_audio_pipeline_dir() -> Path:
    """Find audio_pipeline directory."""
    # Try current directory
    if Path("audio_pipeline").exists():
        return Path("audio_pipeline")
    
    # Try parent directory
    if Path("../audio_pipeline").exists():
        return Path("../audio_pipeline")
    
    # Try searching in common locations
    for parent in [Path.cwd(), Path.cwd().parent]:
        for item in parent.iterdir():
            if item.is_dir() and item.name == "audio_pipeline":
                return item
    
    return None


def backup_file(file_path: Path) -> bool:
    """Backup a file."""
    if not file_path.exists():
        return True
    
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    
    try:
        shutil.copy2(file_path, backup_path)
        return True
    except Exception as e:
        print_error(f"Failed to backup {file_path.name}: {e}")
        return False


def copy_integrated_file(source: Path, dest: Path) -> bool:
    """Copy integrated file."""
    try:
        shutil.copy2(source, dest)
        return True
    except Exception as e:
        print_error(f"Failed to copy {source.name}: {e}")
        return False


def check_dependencies() -> List[Tuple[str, bool]]:
    """Check if required packages are installed."""
    dependencies = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("faster_whisper", "FasterWhisper"),
        ("instructor", "Instructor"),
        ("transformers", "Transformers"),
        ("pydantic", "Pydantic"),
    ]
    
    results = []
    for package, name in dependencies:
        try:
            __import__(package)
            results.append((name, True))
        except ImportError:
            results.append((name, False))
    
    return results


def install_dependencies(missing: List[str]) -> bool:
    """Install missing dependencies."""
    import subprocess
    
    packages = {
        "FasterWhisper": "faster-whisper>=1.0.0",
        "Instructor": "instructor>=1.0.0",
        "Transformers": "transformers>=4.36.0",
        "Pydantic": "pydantic>=2.0.0",
        "PyTorch": "torch>=2.5.0",
        "TorchAudio": "torchaudio>=2.5.0",
    }
    
    install_list = [packages[dep] for dep in missing if dep in packages]
    
    if not install_list:
        return True
    
    print_step(f"Installing {len(install_list)} missing packages...")
    
    cmd = [sys.executable, "-m", "pip", "install"] + install_list
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        return False


def integrate_files(source_dir: Path, target_dir: Path, dry_run: bool = False) -> bool:
    """Integrate all files."""
    
    files_to_integrate = {
        "pipeline_integrated.py": "pipeline.py",
        "vad.py": "vad.py",
        "config.py": "config.py",
        "transcriber.py": "transcriber.py",
        "diarizer.py": "diarizer.py",
        "post_processing_hybrid.py": "post_processing_hybrid.py",
    }
    
    success = True
    
    for source_name, target_name in files_to_integrate.items():
        source_file = source_dir / source_name
        target_file = target_dir / target_name
        
        if not source_file.exists():
            print_warning(f"Source file not found: {source_name}")
            continue
        
        print_step(f"Processing {target_name}...")
        
        if not dry_run:
            # Backup original
            if target_file.exists():
                if backup_file(target_file):
                    print_success(f"Backed up to {target_name}.bak")
                else:
                    success = False
                    continue
            
            # Copy integrated file
            if copy_integrated_file(source_file, target_file):
                print_success(f"Updated {target_name}")
            else:
                success = False
        else:
            print(f"  Would update: {target_file}")
    
    return success


def verify_integration(target_dir: Path) -> bool:
    """Verify integration by trying to import."""
    print_step("Verifying integration...")
    
    # Add target directory to path
    sys.path.insert(0, str(target_dir.parent))
    
    try:
        # Try importing
        from audio_pipeline import AudioPipeline, PipelineConfig
        print_success("AudioPipeline imports OK")
        
        from audio_pipeline.post_processing_hybrid import HybridLLMPostProcessor
        print_success("HybridLLMPostProcessor imports OK")
        
        # Try creating config
        config = PipelineConfig()
        print_success("PipelineConfig creates OK")
        
        # Check new features
        assert hasattr(config, 'llm'), "Missing LLM config"
        print_success("LLM config present")
        
        assert hasattr(config.vad, 'provider'), "Missing VAD provider"
        print_success("VAD provider present")
        
        assert hasattr(config.transcription, 'backend'), "Missing transcription backend"
        print_success("Transcription backend present")
        
        print_success("All verifications passed!")
        return True
        
    except Exception as e:
        print_error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main integration function."""
    parser = argparse.ArgumentParser(
        description="Integrate Audio Pipeline v2.1 updates"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Target audio_pipeline directory (auto-detect if not specified)"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Source directory with integrated files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip integration verification"
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}Audio Pipeline v2.1 - Automated Integration{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    
    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be modified")
    
    # Find target directory
    print_step("Finding audio_pipeline directory...")
    
    if args.target_dir:
        target_dir = args.target_dir
    else:
        target_dir = find_audio_pipeline_dir()
    
    if not target_dir or not target_dir.exists():
        print_error("Could not find audio_pipeline directory!")
        print("Please specify with --target-dir")
        return 1
    
    print_success(f"Found: {target_dir}")
    
    # Check dependencies
    if not args.skip_deps:
        print_step("Checking dependencies...")
        
        dep_results = check_dependencies()
        missing = [name for name, installed in dep_results if not installed]
        
        for name, installed in dep_results:
            if installed:
                print_success(f"{name} installed")
            else:
                print_warning(f"{name} not installed")
        
        if missing and not args.dry_run:
            response = input(f"\nInstall {len(missing)} missing packages? [y/N]: ")
            if response.lower() == 'y':
                if not install_dependencies(missing):
                    print_error("Failed to install dependencies")
                    return 1
                print_success("Dependencies installed")
            else:
                print_warning("Skipping dependency installation")
    
    # Integrate files
    print_step("Integrating files...")
    
    if not integrate_files(args.source_dir, target_dir, args.dry_run):
        print_error("Integration had errors")
        return 1
    
    if args.dry_run:
        print_success("Dry run complete - no files were modified")
        return 0
    
    # Verify integration
    if not args.skip_verify:
        if not verify_integration(target_dir):
            print_error("Integration verification failed!")
            print_warning("You can restore backups (.bak files) if needed")
            return 1
    
    # Success!
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}✓ Integration Complete!{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.END}")
    
    print("\n📋 What was integrated:")
    print("  ✓ Fixed Silero VAD (torchaudio.save)")
    print("  ✓ Fixed config type hints")
    print("  ✓ Added Hybrid LLM post-processing")
    print("  ✓ Added FasterWhisper support")
    print("  ✓ Optimized diarization")
    
    print("\n📚 Next steps:")
    print("  1. Update your config.json (see config_example.json)")
    print("  2. Test with: python main.py --input your_audio.mp3")
    print("  3. Check INTEGRATION_GUIDE.md for details")
    
    print("\n💡 Tip:")
    print(f"  Backups saved as .bak files in {target_dir}")
    print("  To restore: mv file.bak file")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
