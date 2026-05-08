# common/logging.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
# Note: Ensure bitpredict.common.constants exists or replace with your hardcoded string
try:
    from bitpredict.common.constants import LOG_DIR_NAME
except ImportError:
    LOG_DIR_NAME = "logs"

# ============================================================================
# READ CONFIGURATION FROM ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
ENABLE_ROTATION = os.getenv("ENABLE_ROTATION", "true").lower() == "true"
MAX_LOG_SIZE = int(os.getenv("MAX_LOG_SIZE", str(5 * 1024 * 1024)))  # Default 5MB
BACKUP_COUNT = int(os.getenv("BACKUP_COUNT", "3"))  # Keep 3 backup files

# LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s [%(run_mode)s] - %(name)s - %(levelname)s - %(message)s")
DATE_FORMAT = os.getenv("DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Single root-level logs directory
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / LOG_DIR_NAME

# ============================================================================
# CORE LOGGING LOGIC
# ============================================================================

_root_configured = False
PROJECT_NAMESPACE = "bitpredict"

# Add this class before setup_logging
class RunModeFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.run_mode = ""

    def filter(self, record):
        # record.run_mode = self.run_mode if self.run_mode else "-"
        record.run_mode = getattr(self, "run_mode", "-") or "-"
        return True

# Global filter instance
run_mode_filter = RunModeFilter()

def set_run_mode(mode: str):
    run_mode_filter.run_mode = mode

def setup_logging(app_name: str = "app", run_mode: str = ""):
    """
    REQUIRED: Call this ONCE at the start of your main script (entry point).
    Example: setup_logging("data_module")
    
    This configures the parent logger and determines which file all logs 
    for this session will go into.
    """
    global _root_configured
    
    # We configure the top-level 'bitpredict' logger
    root_logger = logging.getLogger(PROJECT_NAMESPACE)
    
    if not _root_configured:
        root_logger.setLevel(LOG_LEVEL)
        root_logger.handlers.clear()
        # Prevent logs from being sent to the system root logger (avoid double logging)
        root_logger.propagate = False 
        
        # Attach run_mode filter
        root_logger.addFilter(run_mode_filter)
        
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        
        # 1. FILE HANDLER (Single file for the entire execution)
        if LOG_TO_FILE:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOG_DIR / f"{app_name}.log"
            
            if ENABLE_ROTATION:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_handler.namer = rotate_namer
            else:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setFormatter(formatter)
            file_handler.addFilter(run_mode_filter)   # <-- ADD THIS
            root_logger.addHandler(file_handler)
        
        # 2. CONSOLE HANDLER
        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(run_mode_filter)   # <-- ADD THIS
            root_logger.addHandler(console_handler)
            
        _root_configured = True
        root_logger.debug(f"Logging initialized. Output file: {app_name}.log")
    
    return root_logger

def get_logger(module_name: str):
    """
    Use this in every module: logger = get_logger(__name__)
    
    This returns a child logger that inherits handlers from the project root.
    """
    # Ensures the name is bitpredict.your_module
    if not module_name.startswith(PROJECT_NAMESPACE):
        full_name = f"{PROJECT_NAMESPACE}.{module_name}"
    else:
        full_name = module_name
        
    return logging.getLogger(full_name)

# ============================================================================
# UTILS
# ============================================================================

def set_global_level(level: int):
    root_logger = logging.getLogger(PROJECT_NAMESPACE)
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

def get_log_info():
    return {
        "log_level": logging.getLevelName(LOG_LEVEL),
        "log_directory": str(LOG_DIR),
        "configured": _root_configured,
        "namespace": PROJECT_NAMESPACE
    }

def rotate_namer(name: str) -> str:
    """
    Convert: data.bars.log.1 -> data.bars.1.log
    """
    base, ext = os.path.splitext(name)      # data.bars.log , .1
    base2, ext2 = os.path.splitext(base)    # data.bars , .log
    return f"{base2}{ext}{ext2}"
