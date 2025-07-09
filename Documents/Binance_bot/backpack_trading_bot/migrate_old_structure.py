#!/usr/bin/env python3
"""
Migration Script for Old Bot Structure
Helps transition from the old scattered structure to the new organized system
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class BotMigrationTool:
    """Tool to migrate from old bot structure to new organized structure"""
    
    def __init__(self, source_dir: str, target_dir: str = "binance_trading_bot"):
        """
        Initialize migration tool.
        
        Args:
            source_dir: Source directory with old structure
            target_dir: Target directory for new structure
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.migration_map = self._create_migration_map()
        
    def _create_migration_map(self) -> Dict[str, str]:
        """Create mapping from old structure to new structure"""
        return {
            # Grid Trading v3.0 (best version)
            "binance-triangular-arbitrage-bot/src/v3/": "core/strategies/grid_trading/",
            "v0.3/src/v3/": "core/strategies/grid_trading/",
            
            # Arbitrage Bot
            "binance-triangular-arbitrage-bot/src/arbitrage_bot/": "core/strategies/arbitrage/",
            "Arb Bot v0.1/": "core/strategies/arbitrage/legacy/",
            
            # Delta Neutral
            "binance-triangular-arbitrage-bot/src/delta_neutral/": "core/strategies/delta_neutral/",
            "v0.3/src/delta_neutral/": "core/strategies/delta_neutral/",
            
            # Advanced Trading System
            "binance-triangular-arbitrage-bot/src/advanced_trading_system/": "core/strategies/advanced/",
            "v0.3/src/advanced_trading_system/": "core/strategies/advanced/",
            
            # Compliance and Risk
            "binance-triangular-arbitrage-bot/src/compliance/": "core/execution/compliance/",
            "v0.3/src/compliance/": "core/execution/compliance/",
            
            # Data and Analytics
            "binance-triangular-arbitrage-bot/data/": "backtesting/data/",
            "v0.3/data/": "backtesting/data/",
            
            # Backtesting
            "binance-triangular-arbitrage-bot/*backtest*.py": "backtesting/legacy/",
            "v0.3/*backtest*.py": "backtesting/legacy/",
            
            # Configuration
            "binance-triangular-arbitrage-bot/config.yaml": "config/legacy/",
            "v0.3/config.yaml": "config/legacy/",
            
            # Deployment
            "binance-triangular-arbitrage-bot/deployment/": "deployment/",
            "v0.3/deployment/": "deployment/",
            
            # Charts and Results
            "binance-triangular-arbitrage-bot/charts/": "backtesting/results/charts/",
            "v0.3/charts/": "backtesting/results/charts/",
            
            # Documentation
            "binance-triangular-arbitrage-bot/*.md": "docs/legacy/",
            "v0.3/*.md": "docs/legacy/",
            
            # Web API (if exists)
            "*/web_api/": "web_api/",
            "*/flask_api/": "web_api/",
        }
    
    def analyze_old_structure(self) -> Dict[str, Any]:
        """Analyze the old structure and provide migration recommendations"""
        analysis = {
            "total_files": 0,
            "migrated_files": 0,
            "duplicate_files": 0,
            "important_files": [],
            "recommendations": []
        }
        
        try:
            # Count files in old structure
            for root, dirs, files in os.walk(self.source_dir):
                analysis["total_files"] += len(files)
                
                # Identify important files
                for file in files:
                    file_path = Path(root) / file
                    if self._is_important_file(file_path):
                        analysis["important_files"].append(str(file_path))
            
            # Check for duplicates
            analysis["duplicate_files"] = self._count_duplicates()
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing old structure: {e}")
            return analysis
    
    def _is_important_file(self, file_path: Path) -> bool:
        """Check if file is important for migration"""
        important_patterns = [
            "*market_analyzer.py",
            "*grid_engine.py",
            "*arbitrage_bot.py",
            "*delta_neutral*.py",
            "*config*.py",
            "*backtest*.py",
            "requirements.txt",
            "*.yaml",
            "*.yml"
        ]
        
        return any(file_path.match(pattern) for pattern in important_patterns)
    
    def _count_duplicates(self) -> int:
        """Count duplicate files across versions"""
        file_names = {}
        duplicate_count = 0
        
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('.py'):
                    if file in file_names:
                        file_names[file] += 1
                    else:
                        file_names[file] = 1
        
        for file, count in file_names.items():
            if count > 1:
                duplicate_count += count - 1
        
        return duplicate_count
    
    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations"""
        recommendations = [
            "Use Grid Trading v3.0 from binance-triangular-arbitrage-bot/src/v3/ as the primary implementation",
            "Preserve the Supertrend enhancements and market regime detection",
            "Consolidate all backtesting files into a single framework",
            "Migrate Flask API to web_api/ directory",
            "Update configuration files to use the new YAML format",
            "Preserve deployment scripts and monitoring setup",
            "Archive old versions in legacy/ subdirectories",
            "Update imports to use the new module structure"
        ]
        
        return recommendations
    
    def migrate_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Migrate files from old structure to new structure.
        
        Args:
            dry_run: If True, only simulate migration without actual file operations
            
        Returns:
            Migration results
        """
        results = {
            "migrated_files": [],
            "skipped_files": [],
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            # Create target directory structure if not dry run
            if not dry_run:
                self._create_target_structure()
            
            # Process each mapping
            for source_pattern, target_path in self.migration_map.items():
                try:
                    migrated = self._process_migration_pattern(
                        source_pattern, target_path, dry_run
                    )
                    results["migrated_files"].extend(migrated)
                    
                except Exception as e:
                    error_msg = f"Error processing {source_pattern}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            results["errors"].append(str(e))
            return results
    
    def _create_target_structure(self):
        """Create the target directory structure"""
        directories = [
            "core/strategies/arbitrage",
            "core/strategies/grid_trading", 
            "core/strategies/delta_neutral",
            "core/strategies/advanced",
            "core/execution/compliance",
            "core/data/analytics",
            "core/utils",
            "config/legacy",
            "backtesting/data",
            "backtesting/results/charts",
            "backtesting/legacy",
            "deployment/docker",
            "deployment/systemd",
            "deployment/monitoring",
            "web_api",
            "tests",
            "docs/legacy",
            "logs"
        ]
        
        for directory in directories:
            (self.target_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _process_migration_pattern(self, source_pattern: str, target_path: str, dry_run: bool) -> List[str]:
        """Process a single migration pattern"""
        migrated_files = []
        
        # Handle glob patterns
        if '*' in source_pattern:
            # Find matching files
            matching_files = list(self.source_dir.glob(source_pattern))
            for file_path in matching_files:
                if file_path.is_file():
                    target_file = self.target_dir / target_path / file_path.name
                    
                    if not dry_run:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, target_file)
                    
                    migrated_files.append(f"{file_path} -> {target_file}")
        else:
            # Handle directory migration
            source_dir = self.source_dir / source_pattern
            if source_dir.exists():
                target_dir = self.target_dir / target_path
                
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all files from source to target
                    for item in source_dir.rglob('*'):
                        if item.is_file():
                            relative_path = item.relative_to(source_dir)
                            target_file = target_dir / relative_path
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target_file)
                            migrated_files.append(f"{item} -> {target_file}")
        
        return migrated_files
    
    def create_migration_report(self, analysis: Dict[str, Any], migration_results: Dict[str, Any]) -> str:
        """Create a comprehensive migration report"""
        report = f"""
# Bot Migration Report
Generated: {Path(__file__).name}

## Analysis Summary
- Total files in old structure: {analysis['total_files']}
- Important files identified: {len(analysis['important_files'])}
- Duplicate files found: {analysis['duplicate_files']}
- Files successfully migrated: {len(migration_results['migrated_files'])}
- Files skipped: {len(migration_results['skipped_files'])}
- Errors encountered: {len(migration_results['errors'])}

## Migration Recommendations
"""
        
        for i, recommendation in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += "\n## Important Files Identified\n"
        for file in analysis['important_files'][:20]:  # Show first 20
            report += f"- {file}\n"
        
        if migration_results['errors']:
            report += "\n## Errors Encountered\n"
            for error in migration_results['errors']:
                report += f"- {error}\n"
        
        report += f"""
## Next Steps
1. Review the migrated files in the new structure
2. Update import statements to use new module paths
3. Test the consolidated strategies
4. Update configuration files to use new YAML format
5. Run the new unified main.py entry point
6. Verify all functionality works as expected

## New Structure Overview
```
binance_trading_bot/
├── core/strategies/          # All trading strategies
├── core/execution/          # Order and risk management
├── core/data/              # Data management
├── core/utils/             # Utilities and config
├── config/                 # Configuration files
├── backtesting/            # Backtesting framework
├── deployment/             # Deployment scripts
├── web_api/               # API interface
└── main.py                # Unified entry point
```

Migration completed successfully! 🎉
"""
        
        return report

def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate old bot structure to new organized system')
    parser.add_argument('source', help='Source directory with old structure')
    parser.add_argument('--target', default='binance_trading_bot', help='Target directory')
    parser.add_argument('--dry-run', action='store_true', help='Simulate migration without file operations')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze old structure')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create migration tool
    migrator = BotMigrationTool(args.source, args.target)
    
    # Analyze old structure
    print("🔍 Analyzing old structure...")
    analysis = migrator.analyze_old_structure()
    
    print(f"📊 Analysis complete:")
    print(f"   Total files: {analysis['total_files']}")
    print(f"   Important files: {len(analysis['important_files'])}")
    print(f"   Duplicate files: {analysis['duplicate_files']}")
    
    if args.analyze_only:
        print("\n📋 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
        return
    
    # Perform migration
    print(f"\n🚀 Starting migration {'(DRY RUN)' if args.dry_run else ''}...")
    migration_results = migrator.migrate_files(dry_run=args.dry_run)
    
    print(f"✅ Migration complete:")
    print(f"   Files migrated: {len(migration_results['migrated_files'])}")
    print(f"   Files skipped: {len(migration_results['skipped_files'])}")
    print(f"   Errors: {len(migration_results['errors'])}")
    
    # Generate report
    report = migrator.create_migration_report(analysis, migration_results)
    
    # Save report
    report_file = Path(args.target) / "MIGRATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Migration report saved to: {report_file}")
    
    if not args.dry_run:
        print("\n🎉 Migration completed successfully!")
        print("   Please review the migrated files and update imports as needed.")
        print("   Run 'python main.py --help' to see the new unified interface.")

if __name__ == "__main__":
    main()