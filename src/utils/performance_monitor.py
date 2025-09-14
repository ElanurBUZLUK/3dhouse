"""
Performance monitoring and telemetry for Sketch2House3D.
Tracks processing times, memory usage, and quality metrics.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import threading
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True,
                 max_metrics: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
            enable_cpu_tracking: Whether to track CPU usage
            max_metrics: Maximum number of metrics to store
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        
        # System info
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': psutil.sys.version,
                'platform': psutil.sys.platform
            }
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
            return {}
    
    @contextmanager
    def track_operation(self, 
                       operation_name: str,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to track operation performance.
        
        Args:
            operation_name: Name of the operation
            metadata: Additional metadata
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_percent()
        
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_percent()
            
            duration_ms = (end_time - start_time) * 1000
            memory_usage_mb = end_memory - start_memory
            cpu_percent = (start_cpu + end_cpu) / 2
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_percent=cpu_percent,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            self._add_metric(metric)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        if not self.enable_cpu_tracking:
            return 0.0
        
        try:
            return psutil.cpu_percent()
        except Exception:
            return 0.0
    
    def _add_metric(self, metric: PerformanceMetrics):
        """Add metric to the list."""
        with self.lock:
            self.metrics.append(metric)
            
            # Keep only the most recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            if not self.metrics:
                return {
                    'total_operations': 0,
                    'success_rate': 0.0,
                    'average_duration_ms': 0.0,
                    'total_duration_ms': 0.0,
                    'operations': {}
                }
            
            # Calculate overall statistics
            total_operations = len(self.metrics)
            successful_operations = sum(1 for m in self.metrics if m.success)
            success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
            
            durations = [m.duration_ms for m in self.metrics if m.success]
            average_duration = np.mean(durations) if durations else 0.0
            total_duration = sum(durations)
            
            # Group by operation
            operations = {}
            for metric in self.metrics:
                op_name = metric.operation_name
                if op_name not in operations:
                    operations[op_name] = {
                        'count': 0,
                        'success_count': 0,
                        'total_duration_ms': 0.0,
                        'average_duration_ms': 0.0,
                        'min_duration_ms': float('inf'),
                        'max_duration_ms': 0.0,
                        'success_rate': 0.0
                    }
                
                op_stats = operations[op_name]
                op_stats['count'] += 1
                if metric.success:
                    op_stats['success_count'] += 1
                    op_stats['total_duration_ms'] += metric.duration_ms
                    op_stats['min_duration_ms'] = min(op_stats['min_duration_ms'], metric.duration_ms)
                    op_stats['max_duration_ms'] = max(op_stats['max_duration_ms'], metric.duration_ms)
            
            # Calculate averages for each operation
            for op_stats in operations.values():
                if op_stats['count'] > 0:
                    op_stats['success_rate'] = op_stats['success_count'] / op_stats['count']
                if op_stats['success_count'] > 0:
                    op_stats['average_duration_ms'] = op_stats['total_duration_ms'] / op_stats['success_count']
                if op_stats['min_duration_ms'] == float('inf'):
                    op_stats['min_duration_ms'] = 0.0
            
            return {
                'total_operations': total_operations,
                'success_rate': success_rate,
                'average_duration_ms': average_duration,
                'total_duration_ms': total_duration,
                'operations': operations,
                'system_info': self.system_info
            }
    
    def get_operation_metrics(self, operation_name: str) -> List[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        with self.lock:
            return [m for m in self.metrics if m.operation_name == operation_name]
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """Get recent metrics."""
        with self.lock:
            return self.metrics[-count:]
    
    def clear_metrics(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
    
    def export_metrics(self, file_path: str) -> bool:
        """Export metrics to JSON file."""
        try:
            with self.lock:
                metrics_data = []
                for metric in self.metrics:
                    metrics_data.append({
                        'operation_name': metric.operation_name,
                        'start_time': metric.start_time,
                        'end_time': metric.end_time,
                        'duration_ms': metric.duration_ms,
                        'memory_usage_mb': metric.memory_usage_mb,
                        'cpu_percent': metric.cpu_percent,
                        'success': metric.success,
                        'error_message': metric.error_message,
                        'metadata': metric.metadata
                    })
                
                with open(file_path, 'w') as f:
                    json.dump({
                        'system_info': self.system_info,
                        'metrics': metrics_data,
                        'export_time': datetime.now().isoformat()
                    }, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False
    
    def get_performance_report(self) -> str:
        """Get a formatted performance report."""
        summary = self.get_metrics_summary()
        
        report = f"""
Performance Report
=================
Total Operations: {summary['total_operations']}
Success Rate: {summary['success_rate']:.2%}
Average Duration: {summary['average_duration_ms']:.2f} ms
Total Duration: {summary['total_duration_ms']:.2f} ms

Operation Breakdown:
"""
        
        for op_name, op_stats in summary['operations'].items():
            report += f"""
  {op_name}:
    Count: {op_stats['count']}
    Success Rate: {op_stats['success_rate']:.2%}
    Average Duration: {op_stats['average_duration_ms']:.2f} ms
    Min Duration: {op_stats['min_duration_ms']:.2f} ms
    Max Duration: {op_stats['max_duration_ms']:.2f} ms
"""
        
        return report


class QualityMetrics:
    """Tracks quality metrics for 3D generation."""
    
    def __init__(self):
        """Initialize quality metrics tracker."""
        self.metrics = []
        self.lock = threading.Lock()
    
    def add_quality_metric(self, 
                          metric_name: str,
                          value: float,
                          threshold: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Add a quality metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            threshold: Optional threshold for pass/fail
            metadata: Additional metadata
        """
        with self.lock:
            metric = {
                'metric_name': metric_name,
                'value': value,
                'threshold': threshold,
                'passed': value >= threshold if threshold is not None else True,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.metrics.append(metric)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality metrics summary."""
        with self.lock:
            if not self.metrics:
                return {'total_metrics': 0, 'pass_rate': 0.0, 'metrics': {}}
            
            # Group by metric name
            metric_groups = {}
            for metric in self.metrics:
                name = metric['metric_name']
                if name not in metric_groups:
                    metric_groups[name] = {
                        'count': 0,
                        'passed': 0,
                        'values': [],
                        'average': 0.0,
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                
                group = metric_groups[name]
                group['count'] += 1
                if metric['passed']:
                    group['passed'] += 1
                
                value = metric['value']
                group['values'].append(value)
                group['min'] = min(group['min'], value)
                group['max'] = max(group['max'], value)
            
            # Calculate averages
            for group in metric_groups.values():
                if group['values']:
                    group['average'] = np.mean(group['values'])
                if group['min'] == float('inf'):
                    group['min'] = 0.0
                if group['max'] == float('-inf'):
                    group['max'] = 0.0
            
            # Calculate overall pass rate
            total_metrics = len(self.metrics)
            passed_metrics = sum(1 for m in self.metrics if m['passed'])
            pass_rate = passed_metrics / total_metrics if total_metrics > 0 else 0.0
            
            return {
                'total_metrics': total_metrics,
                'pass_rate': pass_rate,
                'metrics': metric_groups
            }


# Global instances
performance_monitor = PerformanceMonitor()
quality_metrics = QualityMetrics()


def track_performance(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to track function performance.
    
    Args:
        operation_name: Name of the operation
        metadata: Additional metadata
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with performance_monitor.track_operation(operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    return performance_monitor.get_metrics_summary()


def get_quality_summary() -> Dict[str, Any]:
    """Get quality summary."""
    return quality_metrics.get_quality_summary()
