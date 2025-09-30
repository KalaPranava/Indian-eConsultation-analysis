"""
Simple HTTP server to serve the frontend dashboard
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

def start_frontend_server(port=3000, directory=None):
    """Start a simple HTTP server for the frontend"""
    
    if directory is None:
        directory = Path(__file__).parent
    
    os.chdir(directory)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers to allow API calls
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print(f"🌐 Frontend Dashboard Server starting...")
            print(f"📁 Serving directory: {directory}")
            print(f"🔗 Local URL: http://localhost:{port}")
            print(f"🔗 Network URL: http://127.0.0.1:{port}")
            print(f"🚀 Opening browser...")
            print(f"⚠️  Make sure your API server is running on http://127.0.0.1:8000")
            print("=" * 60)
            
            # Open browser automatically
            webbrowser.open(f'http://localhost:{port}')
            
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 10048:  # Port already in use on Windows
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_frontend_server(port + 1, directory)
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    # Get the frontend directory path
    frontend_dir = Path(__file__).parent
    start_frontend_server(3000, frontend_dir)