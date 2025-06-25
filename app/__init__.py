# app/__init__.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Utiliser init_routes au lieu d'api_bp
    from app.routes import init_routes
    init_routes(app)
    
    return app
