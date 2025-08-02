from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import urllib.parse
import json
from http import HTTPStatus
import socket
from threading import Thread
import mimetypes
from pathlib import Path

# Add the root directory to the Python path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Configure MIME types
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('application/octet-stream', '.onnx')
mimetypes.add_type('application/vnd.apple.mpegurl', '.m3u8')
mimetypes.add_type('video/MP2T', '.ts')

# Import the Flask app
from api.app import app as flask_app

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR / 'static'), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        try:
            # Parse the path
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            print(f"\n[DEBUG] Requested path: {path}")
            
            # Handle API routes
            if path.startswith('/api/'):
                self.proxy_to_flask()
                return
                
            # Handle root path
            if path == '/':
                path = '/index.html'
            
            # Check if file exists in static, models, or stream directories
            for base_path, url_prefix in [
                (BASE_DIR / 'static', '/static'),
                (BASE_DIR / 'models', '/models'),
                (BASE_DIR / 'stream', '/stream')
            ]:
                if path.startswith(url_prefix):
                    # Remove the URL prefix and any leading slashes
                    relative_path = path[len(url_prefix):].lstrip('/')
                    # Handle nested paths correctly
                    file_path = base_path / relative_path
                    file_path = file_path.resolve()
                    
                    # Security check: prevent directory traversal
                    try:
                        file_path.relative_to(BASE_DIR)
                    except ValueError:
                        print(f"[SECURITY] Attempted path traversal: {file_path}")
                        self.send_error(403, "Access denied")
                        return
                    
                    print(f"[DEBUG] Trying to serve: {file_path}")
                    
                    if file_path.exists() and file_path.is_file():
                        print(f"[DEBUG] Found file: {file_path}")
                        try:
                            self.send_file(file_path)
                            return
                        except Exception as e:
                            print(f"[ERROR] Error serving file {file_path}: {str(e)}")
                            self.send_error(500, f"Error serving file: {str(e)}")
                            return
            
            # If file not found, try to serve from root (for backward compatibility)
            print(f"[DEBUG] File not found in any base directory, trying root: {path}")
            if path.startswith('/static/'):
                # For static files, try to serve directly from static directory
                relative_path = path[8:]  # Remove '/static/' prefix
                file_path = (BASE_DIR / 'static' / relative_path).resolve()
                if file_path.exists() and file_path.is_file():
                    print(f"[DEBUG] Found file via static fallback: {file_path}")
                    try:
                        self.send_file(file_path)
                        return
                    except Exception as e:
                        print(f"[ERROR] Error serving file via static fallback {file_path}: {str(e)}")
            
            # If still not found, try the default handler
            self.path = path
            return super().do_GET()
            
        except Exception as e:
            error_msg = f"Error processing request {self.path}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.send_error(500, error_msg)
    
    def send_file(self, file_path):
        """Helper method to send a file with proper MIME type"""
        try:
            print(f"[DEBUG] Sending file: {file_path}")
            print(f"[DEBUG] File exists: {file_path.exists()}")
            print(f"[DEBUG] File is file: {file_path.is_file()}")
            
            with open(str(file_path), 'rb') as f:  # Explicitly convert to string for Windows
                fs = os.fstat(f.fileno())
                ext = os.path.splitext(file_path.name)[1].lower()
                content_type = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
                
                print(f"[DEBUG] Content-Type: {content_type}")
                
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(fs.st_size))
                self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                
                # Send the file in chunks
                self.copyfile(f, self.wfile)
                print("[DEBUG] File sent successfully")
                
        except Exception as e:
            error_msg = f"Error sending file {file_path}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)

    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy_to_flask()
        else:
            self.send_error(404, "Not Found")

    def proxy_to_flask(self):
        # Get request data
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else None
        
        # Parse query parameters
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        # Create a test client for the Flask app
        with flask_app.test_request_context(
            path=parsed_path.path,
            method=self.command,
            data=post_data,
            query_string=query_params,
            headers=dict(self.headers)
        ):
            try:
                # Process the request with Flask
                response = flask_app.full_dispatch_request()
                
                # Send the response
                self.send_response(response.status_code)
                for header, value in response.headers.items():
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(response.get_data())
                
            except Exception as e:
                self.send_error(500, str(e))

def run_flask():
    # Run Flask on a different port (5001) to avoid conflict with the main server
    flask_app.run(port=5001, debug=False, use_reloader=False, host='127.0.0.1')

def run_http_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print('Starting HTTP server at http://localhost:8000')
    httpd.serve_forever()

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start the main HTTP server in the main thread
    run_http_server()
