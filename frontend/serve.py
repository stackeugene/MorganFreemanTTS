from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

os.chdir('frontend')
server = HTTPServer(('localhost', 8080), CORSRequestHandler)
print("Serving frontend at http://localhost:8080")
server.serve_forever()