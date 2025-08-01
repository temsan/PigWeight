from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

if __name__ == '__main__':
    # Установим текущую директорию как корень для HTTP сервера
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    server = HTTPServer(('localhost', 8000), CORSRequestHandler)
    print('Starting server at http://localhost:8000')
    server.serve_forever()
