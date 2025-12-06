#!/usr/bin/env python3
"""
Serveur HTTP simple pour servir la landing page Hooksmith AI.
Usage: python3 serve.py [port]
"""

import http.server
import socketserver
import sys
import os
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Ajouter les headers CORS pour permettre l'appel API depuis le navigateur
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        # Log personnalisé
        print(f"[{self.log_date_time_string()}] {format % args}")

if __name__ == "__main__":
    # Changer vers le répertoire du script
    os.chdir(Path(__file__).parent)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Serveur démarré sur http://localhost:{PORT}")
        print(f"📁 Répertoire: {os.getcwd()}")
        print(f"🌐 Ouvrez http://localhost:{PORT} dans votre navigateur")
        print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Serveur arrêté")
            sys.exit(0)

