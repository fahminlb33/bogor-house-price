import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

hostName = "0.0.0.0"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        req = urllib.request.Request(
            urllib.parse.urljoin("https://picture.rumah123.com/", self.path),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
        )

        print(req.full_url)
        with urllib.request.urlopen(req) as f:
            binary_content = f.read()

            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()

            self.wfile.write(binary_content)

if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
