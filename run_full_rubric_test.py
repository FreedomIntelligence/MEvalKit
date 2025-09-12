#!/usr/bin/env python3
"""
Run full rubric scoring test on all questions
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.QA_scorer import Rubric_scorer

def main():
    """Run complete rubric scoring on HealthBench results"""
    
    print("=== Running Complete HealthBench Rubric Scoring ===")
    print("Dataset: HealthBench")
    print("User ID: test")
    print("Business ID: HealthBench_doubao-1.5-pro-32k_202509091443")
    print("This will process ALL questions with ALL rubrics...")
    print()
    
    try:
        # Initialize scorer
        scorer = Rubric_scorer("HealthBench", "test", "HealthBench_doubao-1.5-pro-32k_202509091443")
        print("✓ Rubric_scorer initialized")
        
        # Run complete scoring
        print("\n🚀 Starting complete scoring process...")
        score_results = scorer.scoring()
        
        print("\n" + "="*50)
        print("📊 FINAL SCORING RESULTS")
        print("="*50)
        print(f"Valid Ratio: {score_results['valid_ratio']:.4f}")
        print(f"Final Score: {score_results['score']:.2f}%")
        
        if 'total_points' in score_results:
            print(f"Total Points Earned: {score_results['total_points']}")
            print(f"Total Possible Points: {score_results['total_possible_points']}")
            print(f"Raw Score: {score_results['total_points']}/{score_results['total_possible_points']}")
        
        print("\n✅ Scoring completed successfully!")
        print(f"Results saved to: results/test/HealthBench_doubao-1.5-pro-32k_202509091443_score.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during scoring: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)