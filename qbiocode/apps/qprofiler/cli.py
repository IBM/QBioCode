"""
CLI wrapper for QProfiler that provides default config path.
"""
import sys
import os

def main():
    """
    Wrapper for qprofiler CLI that adds default config-dir if not specified.
    """
    # Get the directory where qprofiler is installed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_dir = os.path.join(script_dir, 'configs')
    
    # Check if --config-dir is already in arguments
    has_config_dir = any('--config-dir' in arg for arg in sys.argv)
    has_config_path = any('--config-path' in arg for arg in sys.argv)
    
    # If no config-dir or config-path specified, add the default one
    if not has_config_dir and not has_config_path:
        # Use absolute path for filesystem access
        sys.argv.insert(1, f'--config-dir={os.path.abspath(default_config_dir)}')
    
    # Import and run the actual main function
    from apps.qprofiler.qprofiler import main as qprofiler_main
    qprofiler_main()

if __name__ == '__main__':
    main()
