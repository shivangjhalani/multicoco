#!/usr/bin/env python3
"""
Utility script to find and fix torch.utils.checkpoint warnings.

This script helps identify all torch.utils.checkpoint.checkpoint calls
that are missing the use_reentrant parameter and need to be updated.
"""

import os
import re
import glob

def find_checkpoint_calls(root_dir="."):
    """Find all torch.utils.checkpoint.checkpoint calls in Python files."""
    pattern = r'torch\.utils\.checkpoint\.checkpoint\s*\('
    files_with_issues = []
    
    # Search for Python files
    for file_path in glob.glob(os.path.join(root_dir, "**/*.py"), recursive=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        # Check if use_reentrant is already specified
                        if 'use_reentrant' not in content[content.find(line):content.find(line) + 500]:
                            files_with_issues.append({
                                'file': file_path,
                                'line': i,
                                'content': line.strip()
                            })
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return files_with_issues

def print_summary(issues):
    """Print a summary of checkpoint issues found."""
    print("🔍 Torch Checkpoint Warning Fix Summary")
    print("=" * 50)
    
    if not issues:
        print("✅ No checkpoint calls missing use_reentrant parameter found!")
        return
    
    print(f"⚠️  Found {len(issues)} checkpoint calls that need fixing:\n")
    
    for issue in issues:
        print(f"📁 File: {issue['file']}")
        print(f"📍 Line {issue['line']}: {issue['content']}")
        print("🔧 Solution: Add 'use_reentrant=False' parameter")
        print("-" * 40)

def main():
    print("Scanning for torch.utils.checkpoint calls...")
    issues = find_checkpoint_calls()
    print_summary(issues)
    
    if issues:
        print("\n💡 To fix the warnings:")
        print("1. Add 'use_reentrant=False' to each checkpoint call")
        print("2. PyTorch recommends use_reentrant=False for better performance")
        print("3. This will suppress the warnings and prepare for PyTorch 2.5+")

if __name__ == "__main__":
    main() 