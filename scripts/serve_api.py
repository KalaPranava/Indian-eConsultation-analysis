"""
Script for serving the FastAPI application.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main():
    """Main function to serve the API."""
    parser = argparse.ArgumentParser(description="Serve the Indian E-Consultation Analysis API")
    
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["critical", "error", "warning", "info", "debug"],
                        help="Log level")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--ssl_cert", type=str, default=None,
                        help="Path to SSL certificate file")
    parser.add_argument("--ssl_key", type=str, default=None,
                        help="Path to SSL private key file")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Indian E-Consultation Analysis API Server")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set environment variables for the app
    os.environ["CONFIG_PATH"] = args.config_path
    
    # SSL configuration
    ssl_kwargs = {}
    if args.ssl_cert and args.ssl_key:
        ssl_kwargs.update({
            "ssl_certfile": args.ssl_cert,
            "ssl_keyfile": args.ssl_key
        })
        logger.info("SSL enabled")
    
    try:
        # Run the server
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,  # Reload mode only works with 1 worker
            reload=args.reload,
            log_level=args.log_level,
            access_log=True,
            **ssl_kwargs
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()