#!/usr/bin/env python3
"""
Cleanup script for NBA Betting Model codebase.

This script removes temporary files, cache directories, and old artifacts
to keep the codebase clean and organized.

Usage:
    python scripts/cleanup.py
    python scripts/cleanup.py --dry-run    # Show what would be deleted
    python scripts/cleanup.py --deep       # More aggressive cleanup
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


class CodebaseCleanup:
    """Handles cleanup of temporary files and directories."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.deleted_items = []
        
    def log_action(self, action: str, path: str):
        """Log cleanup actions."""
        if self.dry_run:
            print(f"[DRY RUN] {action}: {path}")
        else:
            print(f"{action}: {path}")
            self.deleted_items.append(path)
    
    def remove_file(self, file_path: Path):
        """Remove a single file."""
        if file_path.exists():
            self.log_action("Removing file", str(file_path))
            if not self.dry_run:
                file_path.unlink()
    
    def remove_directory(self, dir_path: Path):
        """Remove a directory and its contents."""
        if dir_path.exists():
            self.log_action("Removing directory", str(dir_path))
            if not self.dry_run:
                shutil.rmtree(dir_path)
    
    def clean_system_files(self):
        """Remove macOS and system-specific files."""
        print("🧹 Cleaning system files...")
        
        # Find and remove .DS_Store files
        for ds_store in self.project_root.rglob(".DS_Store"):
            self.remove_file(ds_store)
        
        # Remove other macOS artifacts
        for pattern in ["._*", ".AppleDouble", ".LSOverride"]:
            for file_path in self.project_root.rglob(pattern):
                self.remove_file(file_path)
    
    def clean_python_cache(self):
        """Remove Python cache files and directories."""
        print("🐍 Cleaning Python cache...")
        
        # Remove __pycache__ directories
        for cache_dir in self.project_root.rglob("__pycache__"):
            self.remove_directory(cache_dir)
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            self.remove_file(pyc_file)
        
        # Remove .pyo files
        for pyo_file in self.project_root.rglob("*.pyo"):
            self.remove_file(pyo_file)
    
    def clean_temporary_files(self):
        """Remove temporary and backup files."""
        print("📝 Cleaning temp files...")
        
        patterns = ["*~", ".#*", "*.tmp", "*.temp", "*.bak", "*.swp", "*.swo"]
        for pattern in patterns:
            for temp_file in self.project_root.rglob(pattern):
                self.remove_file(temp_file)
    
    def clean_old_logs(self, days: int = 30):
        """Remove log files older than specified days."""
        print(f"📋 Cleaning logs older than {days} days...")
        
        logs_dir = self.project_root / "logs"
        if not logs_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                self.remove_file(log_file)
    
    def clean_old_models(self, keep_latest: int = 3):
        """Keep only the latest N model files."""
        print(f"Cleaning old models (keeping latest {keep_latest})...")
        
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            return
        
        # Get all model files sorted by modification time
        model_files = [f for f in models_dir.glob("nba_model_*.joblib") 
                      if f.name != "nba_model_latest.joblib"]
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old model files
        for old_model in model_files[keep_latest:]:
            self.remove_file(old_model)
            
            # Also remove corresponding metadata file
            metadata_file = old_model.with_suffix('.yml')
            if metadata_file.exists():
                self.remove_file(metadata_file)
    
    def clean_empty_directories(self):
        """Remove empty directories."""
        print("📁 Removing empty directories...")
        
        # Walk directories bottom-up to handle nested empty dirs
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                
                # Skip important directories that should exist even if empty
                skip_dirs = {'.git', 'logs', 'models', 'notebooks', 'tests'}
                if dir_name in skip_dirs:
                    continue
                
                try:
                    if not any(dir_path.iterdir()):  # Directory is empty
                        self.remove_directory(dir_path)
                except OSError:
                    pass  # Directory may have been removed already
    
    def deep_clean(self):
        """Perform aggressive cleanup for development environments."""
        print("🔥 Performing deep clean...")
        
        # Remove data files (be careful in production!)
        data_dir = self.project_root / "data"
        if data_dir.exists():
            for subdir in ["raw", "processed"]:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    print(f"⚠️  Deep clean would remove data/{subdir}")
                    if not self.dry_run:
                        response = input(f"Remove {subdir_path}? [y/N]: ")
                        if response.lower() == 'y':
                            self.remove_directory(subdir_path)
        
        # Remove all models
        models_dir = self.project_root / "models"
        if models_dir.exists():
            print("⚠️  Deep clean would remove all models")
            if not self.dry_run:
                response = input(f"Remove all models in {models_dir}? [y/N]: ")
                if response.lower() == 'y':
                    for model_file in models_dir.glob("*"):
                        if model_file.is_file():
                            self.remove_file(model_file)
    
    def run_cleanup(self, deep: bool = False, log_retention_days: int = 30):
        """Run the complete cleanup process."""
        print("=" * 60)
        print("🧹 NBA BETTING MODEL CODEBASE CLEANUP")
        print("=" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be deleted")
            print()
        
        self.clean_system_files()
        self.clean_python_cache()
        self.clean_temporary_files()
        self.clean_old_logs(log_retention_days)
        self.clean_old_models()
        self.clean_empty_directories()
        
        if deep:
            self.deep_clean()
        
        print()
        print("=" * 60)
        if self.dry_run:
            print("✅ DRY RUN COMPLETED")
        else:
            print("✅ CLEANUP COMPLETED")
            print(f"📊 Removed {len(self.deleted_items)} items")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up NBA Betting Model codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--deep",
        action="store_true", 
        help="Perform deep cleanup (removes data and models)"
    )
    
    parser.add_argument(
        "--log-retention-days",
        type=int,
        default=30,
        help="Number of days to retain log files (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # Run cleanup
    cleaner = CodebaseCleanup(project_root, dry_run=args.dry_run)
    cleaner.run_cleanup(
        deep=args.deep,
        log_retention_days=args.log_retention_days
    )


if __name__ == "__main__":
    main()
