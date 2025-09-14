"""
Advanced error handling and logging system for Sketch2House3D.
Provides structured error handling, logging, and error reporting.
"""

import logging
import traceback
import json
import time
from typing import Any, Dict, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import sys
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    INPUT_VALIDATION = "input_validation"
    MODEL_INFERENCE = "model_inference"
    GEOMETRY_PROCESSING = "geometry_processing"
    EXPORT_ERROR = "export_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    CACHE_ERROR = "cache_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    input_data_hash: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ErrorReport:
    """Structured error report."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    context: ErrorContext
    resolution_suggestions: List[str]
    retry_after: Optional[int] = None  # seconds


class ErrorHandler:
    """Handles errors and generates structured reports."""
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 enable_remote_logging: bool = False,
                 remote_endpoint: Optional[str] = None):
        """
        Initialize error handler.
        
        Args:
            log_file: Path to log file
            enable_remote_logging: Whether to enable remote logging
            remote_endpoint: Remote logging endpoint
        """
        self.log_file = log_file
        self.enable_remote_logging = enable_remote_logging
        self.remote_endpoint = remote_endpoint
        
        # Setup logging
        self._setup_logging()
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recent_errors': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Set log level
        logger.setLevel(logging.INFO)
    
    def handle_error(self, 
                    exception: Exception,
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    resolution_suggestions: Optional[List[str]] = None) -> ErrorReport:
        """
        Handle an error and generate a structured report.
        
        Args:
            exception: The exception that occurred
            context: Error context
            severity: Error severity
            category: Error category
            resolution_suggestions: Suggested resolutions
            
        Returns:
            Error report
        """
        # Generate error ID
        error_id = str(uuid.uuid4())
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=traceback.format_exc(),
            context=context,
            resolution_suggestions=resolution_suggestions or [],
            retry_after=self._get_retry_after(exception, category)
        )
        
        # Log error
        self._log_error(error_report)
        
        # Update statistics
        self._update_error_stats(error_report)
        
        # Send to remote logging if enabled
        if self.enable_remote_logging:
            self._send_to_remote(error_report)
        
        return error_report
    
    def _get_retry_after(self, exception: Exception, category: ErrorCategory) -> Optional[int]:
        """Get retry delay for error."""
        # Network errors - retry after 5 seconds
        if category == ErrorCategory.NETWORK_ERROR:
            return 5
        
        # System errors - retry after 30 seconds
        if category == ErrorCategory.SYSTEM_ERROR:
            return 30
        
        # Model inference errors - retry after 10 seconds
        if category == ErrorCategory.MODEL_INFERENCE:
            return 10
        
        # No retry for other errors
        return None
    
    def _log_error(self, error_report: ErrorReport):
        """Log error to file and console."""
        log_message = f"""
Error ID: {error_report.error_id}
Severity: {error_report.severity.value}
Category: {error_report.category.value}
Operation: {error_report.context.operation}
Message: {error_report.message}
Exception: {error_report.exception_type}: {error_report.exception_message}
Traceback:
{error_report.traceback}
Context: {json.dumps(asdict(error_report.context), indent=2)}
Resolution Suggestions: {error_report.resolution_suggestions}
"""
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _update_error_stats(self, error_report: ErrorReport):
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        
        # Update category stats
        category = error_report.category.value
        if category not in self.error_stats['errors_by_category']:
            self.error_stats['errors_by_category'][category] = 0
        self.error_stats['errors_by_category'][category] += 1
        
        # Update severity stats
        severity = error_report.severity.value
        if severity not in self.error_stats['errors_by_severity']:
            self.error_stats['errors_by_severity'][severity] = 0
        self.error_stats['errors_by_severity'][severity] += 1
        
        # Add to recent errors
        self.error_stats['recent_errors'].append({
            'error_id': error_report.error_id,
            'timestamp': error_report.timestamp,
            'severity': severity,
            'category': category,
            'operation': error_report.context.operation
        })
        
        # Keep only last 100 errors
        if len(self.error_stats['recent_errors']) > 100:
            self.error_stats['recent_errors'] = self.error_stats['recent_errors'][-100:]
    
    def _send_to_remote(self, error_report: ErrorReport):
        """Send error report to remote logging service."""
        if not self.remote_endpoint:
            return
        
        try:
            import requests
            
            # Convert to JSON
            report_data = asdict(error_report)
            report_data['severity'] = error_report.severity.value
            report_data['category'] = error_report.category.value
            
            # Send to remote endpoint
            response = requests.post(
                self.remote_endpoint,
                json=report_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send error report to remote: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not send error report to remote: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_stats.copy()
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.error_stats['recent_errors'][-count:]
    
    def clear_error_stats(self):
        """Clear error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recent_errors': []
        }


class ErrorRecovery:
    """Handles error recovery and fallback strategies."""
    
    def __init__(self, error_handler: ErrorHandler):
        """Initialize error recovery."""
        self.error_handler = error_handler
    
    def retry_with_backoff(self, 
                          func: Callable,
                          max_retries: int = 3,
                          backoff_factor: float = 2.0,
                          context: Optional[ErrorContext] = None) -> Any:
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for delay
            context: Error context
            
        Returns:
            Function result or raises last exception
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate delay
                    delay = backoff_factor ** attempt
                    
                    # Log retry attempt
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                    
                    # Wait before retry
                    time.sleep(delay)
                else:
                    # Max retries reached
                    if context:
                        self.error_handler.handle_error(
                            e, context, ErrorSeverity.HIGH, ErrorCategory.SYSTEM_ERROR
                        )
                    raise e
        
        # This should never be reached
        raise last_exception
    
    def fallback_strategy(self, 
                         primary_func: Callable,
                         fallback_func: Callable,
                         context: Optional[ErrorContext] = None) -> Any:
        """
        Try primary function, fallback to secondary if it fails.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function
            context: Error context
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func()
        except Exception as e:
            logger.warning(f"Primary function failed, trying fallback: {e}")
            
            if context:
                self.error_handler.handle_error(
                    e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_ERROR
                )
            
            try:
                return fallback_func()
            except Exception as fallback_e:
                logger.error(f"Fallback function also failed: {fallback_e}")
                
                if context:
                    self.error_handler.handle_error(
                        fallback_e, context, ErrorSeverity.HIGH, ErrorCategory.SYSTEM_ERROR
                    )
                
                raise fallback_e


def error_handler(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 resolution_suggestions: Optional[List[str]] = None):
    """
    Decorator for error handling.
    
    Args:
        severity: Error severity
        category: Error category
        resolution_suggestions: Suggested resolutions
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context
                context = ErrorContext(
                    operation=func.__name__,
                    metadata={'args': str(args), 'kwargs': str(kwargs)}
                )
                
                # Handle error
                error_report = global_error_handler.handle_error(
                    e, context, severity, category, resolution_suggestions
                )
                
                # Re-raise exception
                raise e
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                default_return: Any = None,
                context: Optional[ErrorContext] = None) -> Any:
    """
    Safely execute function with error handling.
    
    Args:
        func: Function to execute
        default_return: Default return value on error
        context: Error context
        
    Returns:
        Function result or default return value
    """
    try:
        return func()
    except Exception as e:
        if context:
            global_error_handler.handle_error(
                e, context, ErrorSeverity.LOW, ErrorCategory.UNKNOWN
            )
        return default_return


# Global error handler instance
global_error_handler = ErrorHandler()
global_error_recovery = ErrorRecovery(global_error_handler)


# Convenience functions
def handle_error(exception: Exception, 
                operation: str,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                category: ErrorCategory = ErrorCategory.UNKNOWN) -> ErrorReport:
    """Handle an error with default context."""
    context = ErrorContext(operation=operation)
    return global_error_handler.handle_error(exception, context, severity, category)


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics."""
    return global_error_handler.get_error_stats()


def retry_with_backoff(func: Callable, max_retries: int = 3) -> Any:
    """Retry function with backoff."""
    return global_error_recovery.retry_with_backoff(func, max_retries)
