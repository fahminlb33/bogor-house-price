from flask import Flask

from infrastructure.database import db
from infrastructure.predictor import load_model


def create_app():
    # create Flask app
    app = Flask(__name__,
                instance_relative_config=True,
                static_url_path="/assets",
                static_folder='public')

    # load default config
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI="sqlite:///test.db",
        MODEL_PATH=
        "/home/fahmi/projects/project-rumah-regresi/models/random_forest.joblib"
    )

    # configure database
    app.logger.info('Connecting to database...')
    db.init_app(app)

    # load model
    app.logger.info('Loading model...')
    load_model(app.config["MODEL_PATH"])

    # load blueprints
    import controllers

    app.logger.info('Registering blueprints...')
    app.register_blueprint(controllers.home_router)
    app.register_blueprint(controllers.about_router)
    app.register_blueprint(controllers.ask_ai_router)
    app.register_blueprint(controllers.predictions_router)

    return app
