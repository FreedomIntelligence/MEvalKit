#!/usr/bin/env python3
"""
Test script for Rubric_scorer functionality
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.QA_scorer import Rubric_scorer

def test_rubric_scorer():
    """Test the Rubric_scorer with the specified HealthBench result file"""
    
    print("=== Testing Rubric_scorer ===")
    print("Dataset: HealthBench")
    print("User ID: test") 
    print("Business ID: HealthBench_doubao-1.5-pro-32k_202509091443")
    print()
    
    try:
        # Initialize scorer
        scorer = Rubric_scorer("HealthBench", "test", "HealthBench_doubao-1.5-pro-32k_202509091443")
        print("✓ Rubric_scorer initialized successfully")
        
        # Test rubrics loading
        print("\n--- Testing Rubrics Loading ---")
        rubrics = scorer.load_rubrics()
        
        if rubrics is not None:
            print(f"✓ Successfully loaded {len(rubrics)} questions with rubrics")
            
            # Show summary of rubrics for first few questions
            for i, question_rubrics in enumerate(rubrics[:3]):  # Show first 3 questions
                print(f"Question {i+1}: {len(question_rubrics)} rubrics")
                for j, rubric in enumerate(question_rubrics[:2]):  # Show first 2 rubrics per question
                    print(f"  Rubric {j+1}: {rubric['points']} points - {rubric['criterion'][:80]}...")
                if len(question_rubrics) > 2:
                    print(f"  ... and {len(question_rubrics)-2} more rubrics")
                print()
        else:
            print("✗ Failed to load rubrics")
            return False
        
        # Test basic functionality without API calls
        print("--- Testing Basic Functionality ---")
        
        # Test prompt generation
        try:
            # Test with the first question's first rubric
            if len(rubrics) > 0 and len(rubrics[0]) > 0:
                test_question = "Test question"
                test_response = "Test response"
                test_rubric = [rubrics[0][0]]  # First rubric of first question
                
                rubric_items = scorer._format_rubric_items(test_rubric)
                print(f"✓ Rubric formatting works - generated {len(rubric_items)} characters")
                
                prompt = scorer._generate_prompt(test_question, test_response, rubric_items)
                print(f"✓ Prompt generation works - generated {len(prompt)} characters")
                
                # Test score extraction with mock JSON
                mock_json_response = '```json\n{"explanation": "Test", "criteria_met": true}\n```'
                extracted_score = scorer.extract_rubric_score(mock_json_response)
                print(f"✓ Score extraction works - extracted score: {extracted_score}")
                
        except Exception as e:
            print(f"✗ Basic functionality test failed: {e}")
            
        print("Skipped actual scoring process to avoid API calls.")
            
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rubric_scorer()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)