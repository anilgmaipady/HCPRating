#!/usr/bin/env python3
"""
Export Utilities for RD Rating System
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExportUtils:
    """Utility class for exporting scoring results in various formats."""
    
    def __init__(self, output_dir: str = "exports"):
        """Initialize export utilities."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_to_json(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export results to JSON format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rd_scores_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "format": "json"
            },
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results)} results to {filepath}")
        return str(filepath)
    
    def export_to_csv(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export results to CSV format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rd_scores_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten the results for CSV export
        flattened_results = []
        for result in results:
            flattened_result = {
                'rd_name': result.get('rd_name', ''),
                'session_date': result.get('session_date', ''),
                'overall_score': result.get('overall_score', 0),
                'empathy_score': result.get('scores', {}).get('empathy', 0),
                'clarity_score': result.get('scores', {}).get('clarity', 0),
                'accuracy_score': result.get('scores', {}).get('accuracy', 0),
                'professionalism_score': result.get('scores', {}).get('professionalism', 0),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', ''),
                'strengths': '; '.join(result.get('strengths', [])),
                'areas_for_improvement': '; '.join(result.get('areas_for_improvement', [])),
                'timestamp': result.get('timestamp', '')
            }
            flattened_results.append(flattened_result)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(results)} results to {filepath}")
        return str(filepath)
    
    def export_to_excel(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export results to Excel format with multiple sheets."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rd_scores_{timestamp}.xlsx"
        
        filepath = self.output_dir / filename
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self._create_summary_data(results)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            detailed_results = self._flatten_results_for_excel(results)
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Scores breakdown sheet
            scores_data = self._create_scores_breakdown(results)
            scores_df = pd.DataFrame(scores_data)
            scores_df.to_excel(writer, sheet_name='Scores Breakdown', index=False)
        
        logger.info(f"Exported {len(results)} results to {filepath}")
        return str(filepath)
    
    def export_report(self, results: List[Dict[str, Any]], report_data: Dict[str, Any], 
                     filename: Optional[str] = None) -> str:
        """Export comprehensive report with results and analysis."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rd_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            "report_info": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(results),
                "report_type": "comprehensive"
            },
            "summary": report_data.get('summary', {}),
            "detailed_results": results,
            "analysis": self._generate_analysis(results)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported comprehensive report to {filepath}")
        return str(filepath)
    
    def _create_summary_data(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create summary data for Excel export."""
        if not results:
            return []
        
        # Calculate averages
        avg_scores = {
            'overall': sum(r.get('overall_score', 0) for r in results) / len(results),
            'empathy': sum(r.get('scores', {}).get('empathy', 0) for r in results) / len(results),
            'clarity': sum(r.get('scores', {}).get('clarity', 0) for r in results) / len(results),
            'accuracy': sum(r.get('scores', {}).get('accuracy', 0) for r in results) / len(results),
            'professionalism': sum(r.get('scores', {}).get('professionalism', 0) for r in results) / len(results)
        }
        
        # Find top performers
        sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
        top_performers = sorted_results[:3]
        
        summary_data = [
            {
                'metric': 'Total Evaluations',
                'value': len(results)
            },
            {
                'metric': 'Average Overall Score',
                'value': round(avg_scores['overall'], 2)
            },
            {
                'metric': 'Average Empathy Score',
                'value': round(avg_scores['empathy'], 2)
            },
            {
                'metric': 'Average Clarity Score',
                'value': round(avg_scores['clarity'], 2)
            },
            {
                'metric': 'Average Accuracy Score',
                'value': round(avg_scores['accuracy'], 2)
            },
            {
                'metric': 'Average Professionalism Score',
                'value': round(avg_scores['professionalism'], 2)
            }
        ]
        
        # Add top performers
        for i, performer in enumerate(top_performers, 1):
            summary_data.append({
                'metric': f'Top Performer #{i}',
                'value': f"{performer.get('rd_name', 'Unknown')} - {performer.get('overall_score', 0):.2f}"
            })
        
        return summary_data
    
    def _flatten_results_for_excel(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten results for Excel export."""
        flattened = []
        for result in results:
            flattened_result = {
                'RD Name': result.get('rd_name', ''),
                'Session Date': result.get('session_date', ''),
                'Overall Score': result.get('overall_score', 0),
                'Empathy': result.get('scores', {}).get('empathy', 0),
                'Clarity': result.get('scores', {}).get('clarity', 0),
                'Accuracy': result.get('scores', {}).get('accuracy', 0),
                'Professionalism': result.get('scores', {}).get('professionalism', 0),
                'Confidence': result.get('confidence', 0),
                'Reasoning': result.get('reasoning', ''),
                'Strengths': '; '.join(result.get('strengths', [])),
                'Areas for Improvement': '; '.join(result.get('areas_for_improvement', [])),
                'Timestamp': result.get('timestamp', '')
            }
            flattened.append(flattened_result)
        
        return flattened
    
    def _create_scores_breakdown(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create scores breakdown for Excel export."""
        breakdown = []
        
        for result in results:
            rd_name = result.get('rd_name', 'Unknown')
            scores = result.get('scores', {})
            
            for dimension, score in scores.items():
                breakdown.append({
                    'RD Name': rd_name,
                    'Dimension': dimension.title(),
                    'Score': score,
                    'Overall Score': result.get('overall_score', 0)
                })
        
        return breakdown
    
    def _generate_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis of the results."""
        if not results:
            return {}
        
        # Calculate statistics
        scores_by_dimension = {
            'empathy': [r.get('scores', {}).get('empathy', 0) for r in results],
            'clarity': [r.get('scores', {}).get('clarity', 0) for r in results],
            'accuracy': [r.get('scores', {}).get('accuracy', 0) for r in results],
            'professionalism': [r.get('scores', {}).get('professionalism', 0) for r in results]
        }
        
        analysis = {
            'score_distribution': {},
            'dimension_analysis': {},
            'performance_insights': []
        }
        
        # Score distribution
        overall_scores = [r.get('overall_score', 0) for r in results]
        analysis['score_distribution'] = {
            'excellent_5': sum(1 for s in overall_scores if s >= 4.5),
            'good_4': sum(1 for s in overall_scores if 3.5 <= s < 4.5),
            'average_3': sum(1 for s in overall_scores if 2.5 <= s < 3.5),
            'below_average_2': sum(1 for s in overall_scores if 1.5 <= s < 2.5),
            'poor_1': sum(1 for s in overall_scores if s < 1.5)
        }
        
        # Dimension analysis
        for dimension, scores in scores_by_dimension.items():
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            analysis['dimension_analysis'][dimension] = {
                'average': round(avg_score, 2),
                'maximum': max_score,
                'minimum': min_score,
                'range': max_score - min_score
            }
        
        # Performance insights
        if len(results) > 1:
            # Find strongest and weakest dimensions
            dimension_averages = {
                dim: analysis['dimension_analysis'][dim]['average'] 
                for dim in analysis['dimension_analysis']
            }
            
            strongest_dim = max(dimension_averages, key=dimension_averages.get)
            weakest_dim = min(dimension_averages, key=dimension_averages.get)
            
            analysis['performance_insights'] = [
                f"Strongest dimension: {strongest_dim.title()} (avg: {dimension_averages[strongest_dim]:.2f})",
                f"Weakest dimension: {weakest_dim.title()} (avg: {dimension_averages[weakest_dim]:.2f})",
                f"Overall performance: {sum(overall_scores) / len(overall_scores):.2f}/5.0"
            ]
        
        return analysis 