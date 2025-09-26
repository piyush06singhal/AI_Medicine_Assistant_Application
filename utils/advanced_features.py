"""
Advanced Features for AI Medical Assistant
Additional technical features for portfolio enhancement.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import base64
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics and visualization features."""
    
    def __init__(self):
        """Initialize advanced analytics."""
        self.logger = logging.getLogger(__name__)
    
    def create_prediction_timeline(self, history: List[Dict]) -> go.Figure:
        """Create prediction timeline visualization."""
        if not history:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Group by date and count predictions
        daily_counts = df.groupby('date').size().reset_index(name='count')
        
        # Create timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['count'],
            mode='lines+markers',
            name='Daily Predictions',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#ff7f0e')
        ))
        
        fig.update_layout(
            title="Prediction Timeline",
            xaxis_title="Date",
            yaxis_title="Number of Predictions",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_confidence_distribution(self, history: List[Dict]) -> go.Figure:
        """Create confidence score distribution chart."""
        if not history:
            return None
        
        # Extract confidence scores
        confidences = [h.get('confidence', 0) for h in history if h.get('confidence') is not None]
        
        if not confidences:
            return None
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name='Confidence Distribution',
            marker_color='#2ca02c',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            template='plotly_white'
        )
        
        return fig
    
    def create_model_performance_metrics(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        if not history:
            return {}
        
        # Extract data
        confidences = [h.get('confidence', 0) for h in history if h.get('confidence') is not None]
        processing_times = [h.get('processing_time', 0) for h in history if h.get('processing_time') is not None]
        
        metrics = {
            'total_predictions': len(history),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'median_confidence': np.median(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'high_confidence_predictions': sum(1 for c in confidences if c >= 0.7),
            'low_confidence_predictions': sum(1 for c in confidences if c < 0.4),
            'success_rate': len([h for h in history if h.get('predicted_disease') != 'Unknown']) / len(history) if history else 0
        }
        
        return metrics
    
    def create_disease_heatmap(self, history: List[Dict]) -> go.Figure:
        """Create disease prediction heatmap."""
        if not history:
            return None
        
        # Extract data
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Create pivot table
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Prediction Activity Heatmap",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            template='plotly_white'
        )
        
        return fig

class ModelComparison:
    """Model comparison and evaluation features."""
    
    def __init__(self):
        """Initialize model comparison."""
        self.logger = logging.getLogger(__name__)
    
    def compare_model_performance(self, history: List[Dict]) -> Dict[str, Any]:
        """Compare performance of different models."""
        if not history:
            return {}
        
        # Group by model type
        nlp_only = [h for h in history if h.get('model_availability', {}).get('nlp_available') and not h.get('model_availability', {}).get('cv_available')]
        cv_only = [h for h in history if h.get('model_availability', {}).get('cv_available') and not h.get('model_availability', {}).get('nlp_available')]
        both_models = [h for h in history if h.get('model_availability', {}).get('nlp_available') and h.get('model_availability', {}).get('cv_available')]
        
        comparison = {}
        
        for model_type, data in [('NLP Only', nlp_only), ('CV Only', cv_only), ('Both Models', both_models)]:
            if data:
                confidences = [h.get('confidence', 0) for h in data if h.get('confidence') is not None]
                processing_times = [h.get('processing_time', 0) for h in data if h.get('processing_time') is not None]
                
                comparison[model_type] = {
                    'count': len(data),
                    'avg_confidence': np.mean(confidences) if confidences else 0,
                    'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                    'success_rate': len([h for h in data if h.get('predicted_disease') != 'Unknown']) / len(data)
                }
        
        return comparison
    
    def create_model_comparison_chart(self, comparison: Dict[str, Any]) -> go.Figure:
        """Create model comparison chart."""
        if not comparison:
            return None
        
        models = list(comparison.keys())
        confidences = [comparison[m]['avg_confidence'] for m in models]
        processing_times = [comparison[m]['avg_processing_time'] for m in models]
        success_rates = [comparison[m]['success_rate'] for m in models]
        
        fig = go.Figure()
        
        # Add confidence bars
        fig.add_trace(go.Bar(
            name='Avg Confidence',
            x=models,
            y=confidences,
            yaxis='y',
            offsetgroup=1,
            marker_color='#1f77b4'
        ))
        
        # Add processing time bars
        fig.add_trace(go.Bar(
            name='Avg Processing Time (s)',
            x=models,
            y=processing_times,
            yaxis='y2',
            offsetgroup=2,
            marker_color='#ff7f0e'
        ))
        
        # Add success rate line
        fig.add_trace(go.Scatter(
            name='Success Rate',
            x=models,
            y=success_rates,
            yaxis='y3',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model Type",
            yaxis=dict(title="Confidence", side="left"),
            yaxis2=dict(title="Processing Time (s)", side="right", overlaying="y"),
            yaxis3=dict(title="Success Rate", side="right", overlaying="y", position=0.85),
            template='plotly_white'
        )
        
        return fig

class DataExport:
    """Data export and reporting features."""
    
    def __init__(self):
        """Initialize data export."""
        self.logger = logging.getLogger(__name__)
    
    def export_comprehensive_report(self, history: List[Dict], format: str = 'json') -> str:
        """Export comprehensive analysis report."""
        if not history:
            return None
        
        # Calculate metrics
        analytics = AdvancedAnalytics()
        metrics = analytics.create_model_performance_metrics(history)
        
        comparison = ModelComparison()
        model_comparison = comparison.compare_model_performance(history)
        
        # Create comprehensive report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_predictions': len(history),
                'report_version': '1.0'
            },
            'performance_metrics': metrics,
            'model_comparison': model_comparison,
            'recent_predictions': history[-10:] if len(history) > 10 else history,
            'disease_distribution': self._get_disease_distribution(history),
            'language_distribution': self._get_language_distribution(history),
            'temporal_analysis': self._get_temporal_analysis(history)
        }
        
        # Export based on format
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'csv':
            return self._export_to_csv(history)
        else:
            return str(report)
    
    def _get_disease_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """Get disease distribution."""
        diseases = [h.get('predicted_disease', 'Unknown') for h in history]
        return {disease: diseases.count(disease) for disease in set(diseases)}
    
    def _get_language_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """Get language distribution."""
        languages = [h.get('detected_language', 'en') for h in history]
        return {lang: languages.count(lang) for lang in set(languages)}
    
    def _get_temporal_analysis(self, history: List[Dict]) -> Dict[str, Any]:
        """Get temporal analysis."""
        if not history:
            return {}
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'first_prediction': df['timestamp'].min().isoformat(),
            'last_prediction': df['timestamp'].max().isoformat(),
            'total_days': (df['timestamp'].max() - df['timestamp'].min()).days + 1,
            'avg_predictions_per_day': len(history) / ((df['timestamp'].max() - df['timestamp'].min()).days + 1)
        }
    
    def _export_to_csv(self, history: List[Dict]) -> str:
        """Export to CSV format."""
        df = pd.DataFrame(history)
        return df.to_csv(index=False)

class RealTimeMonitoring:
    """Real-time monitoring and alerting features."""
    
    def __init__(self):
        """Initialize real-time monitoring."""
        self.logger = logging.getLogger(__name__)
    
    def get_system_health(self, history: List[Dict]) -> Dict[str, Any]:
        """Get system health metrics."""
        if not history:
            return {'status': 'healthy', 'message': 'No data available'}
        
        # Calculate recent metrics
        recent_history = [h for h in history if (datetime.now() - datetime.fromisoformat(h.get('timestamp', '2024-01-01'))).days <= 1]
        
        if not recent_history:
            return {'status': 'healthy', 'message': 'No recent activity'}
        
        # Calculate metrics
        avg_confidence = np.mean([h.get('confidence', 0) for h in recent_history])
        avg_processing_time = np.mean([h.get('processing_time', 0) for h in recent_history])
        error_rate = len([h for h in recent_history if h.get('predicted_disease') == 'Unknown']) / len(recent_history)
        
        # Determine health status
        if avg_confidence < 0.3 or error_rate > 0.5 or avg_processing_time > 10:
            status = 'warning'
            message = 'System performance degraded'
        elif avg_confidence < 0.5 or error_rate > 0.3 or avg_processing_time > 5:
            status = 'caution'
            message = 'System performance below optimal'
        else:
            status = 'healthy'
            message = 'System operating normally'
        
        return {
            'status': status,
            'message': message,
            'metrics': {
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_processing_time,
                'error_rate': error_rate,
                'recent_predictions': len(recent_history)
            }
        }
    
    def create_health_dashboard(self, health: Dict[str, Any]) -> str:
        """Create health dashboard HTML."""
        status_colors = {
            'healthy': '#2ca02c',
            'caution': '#ff7f0e',
            'warning': '#d62728'
        }
        
        color = status_colors.get(health['status'], '#666666')
        
        html = f"""
        <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h3>System Health: {health['status'].upper()}</h3>
            <p>{health['message']}</p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div>
                    <strong>Avg Confidence:</strong><br>
                    {health['metrics']['avg_confidence']:.2%}
                </div>
                <div>
                    <strong>Processing Time:</strong><br>
                    {health['metrics']['avg_processing_time']:.2f}s
                </div>
                <div>
                    <strong>Error Rate:</strong><br>
                    {health['metrics']['error_rate']:.2%}
                </div>
                <div>
                    <strong>Recent Predictions:</strong><br>
                    {health['metrics']['recent_predictions']}
                </div>
            </div>
        </div>
        """
        
        return html

# Global instances
advanced_analytics = AdvancedAnalytics()
model_comparison = ModelComparison()
data_export = DataExport()
real_time_monitoring = RealTimeMonitoring()

def create_advanced_dashboard(history: List[Dict]) -> None:
    """Create advanced analytics dashboard."""
    st.markdown("## 📊 Advanced Analytics Dashboard")
    
    # System health
    health = real_time_monitoring.get_system_health(history)
    st.markdown(real_time_monitoring.create_health_dashboard(health), unsafe_allow_html=True)
    
    # Performance metrics
    metrics = advanced_analytics.create_model_performance_metrics(history)
    if metrics:
        st.markdown("### 📈 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", metrics['total_predictions'])
        with col2:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2%}")
        with col3:
            st.metric("Success Rate", f"{metrics['success_rate']:.2%}")
        with col4:
            st.metric("Avg Processing Time", f"{metrics['avg_processing_time']:.2f}s")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction timeline
        timeline_fig = advanced_analytics.create_prediction_timeline(history)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidence_fig = advanced_analytics.create_confidence_distribution(history)
        if confidence_fig:
            st.plotly_chart(confidence_fig, use_container_width=True)
    
    # Model comparison
    comparison = model_comparison.compare_model_performance(history)
    if comparison:
        st.markdown("### 🤖 Model Performance Comparison")
        comparison_fig = model_comparison.create_model_comparison_chart(comparison)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export JSON Report"):
            report = data_export.export_comprehensive_report(history, 'json')
            st.download_button(
                label="Download JSON Report",
                data=report,
                file_name=f"medical_assistant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export CSV Data"):
            csv_data = data_export.export_comprehensive_report(history, 'csv')
            st.download_button(
                label="Download CSV Data",
                data=csv_data,
                file_name=f"medical_assistant_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Export Full Report"):
            report = data_export.export_comprehensive_report(history, 'json')
            st.download_button(
                label="Download Full Report",
                data=report,
                file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
