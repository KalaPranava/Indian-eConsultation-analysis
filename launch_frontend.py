"""
Direct launcher for the frontend dashboard
Run this if the batch file doesn't work
"""
import os
import sys
from pathlib import Path

def main():
    print("=" * 50)
    print("🚀 Indian E-Consultation Analysis Dashboard")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Frontend directory: {frontend_dir}")
    
    # Check if frontend files exist
    serve_file = frontend_dir / "serve_frontend.py"
    if not serve_file.exists():
        print(f"❌ ERROR: Could not find {serve_file}")
        print("Make sure you're in the correct directory")
        input("Press Enter to exit...")
        return
    
    print("✅ Frontend files found")
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    print(f"📂 Changed to: {os.getcwd()}")
    
    # Import and run the frontend server
    try:
        print("🌐 Starting frontend server...")
        print("📖 Frontend will be available at: http://localhost:3000")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Import the frontend server module
        sys.path.insert(0, str(frontend_dir))
        from serve_frontend import start_frontend_server
        
        # Start the server
        start_frontend_server(port=3000, directory=frontend_dir)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative method...")
        
        # Fallback: use subprocess
        import subprocess
        try:
            subprocess.run([sys.executable, "serve_frontend.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running server: {e}")
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("👋 Goodbye!")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()