"""
Development server script for the Fraud Detection API.
"""
import logging
import os
from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create CLI app
app = typer.Typer(help="Fraud Detection API development server")

def main(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    env_file: str = ".env",
):
    """Run the development server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to reload on file changes
        env_file: Path to environment file
    """
    # Load environment variables
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"Environment file {env_path} not found")
    
    # Set demo mode
    if os.getenv("DEMO_MODE", "").lower() == "true":
        logger.info("Running in demo mode")
    
    # Run server
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def dev(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    env_file: str = ".env",
):
    """Run the development server with auto-reload.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to reload on file changes
        env_file: Path to environment file
    """
    main(host, port, reload, env_file)


@app.command()
def prod(
    host: str = "0.0.0.0",
    port: int = int(os.getenv("PORT", "8000")),
    env_file: str = ".env",
):
    """Run the production server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        env_file: Path to environment file
    """
    main(host, port, False, env_file)


@app.command()
def create_tables():
    """Create database tables (for development)."""
    logger.info("Creating database tables")
    # In a real implementation, this would create tables in Supabase
    # For this demo, we'll just log that it would happen
    logger.info("Tables would be created in Supabase")


@app.command()
def seed_data():
    """Seed database with sample data (for development)."""
    logger.info("Seeding database with sample data")
    # In a real implementation, this would seed data in Supabase
    # For this demo, we'll just log that it would happen
    logger.info("Data would be seeded in Supabase")


if __name__ == "__main__":
    app() 