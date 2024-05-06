import os
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask

from appv2.utils.shared import get_settings, cache


def create_app():
  # create flask app
  app = Flask(
      __name__,
      static_folder="public",
      static_url_path="/assets",
      template_folder="templates")

  # load configuration
  settings = get_settings()
  app.config.from_object(settings)

  # initialize cache
  cache.init_app(app)

  # initialize database
  from appv2.utils.db import db_session

  @app.teardown_appcontext
  def shutdown_session(exception=None):
    db_session.remove()

  # register blueprints
  from appv2.routers import AboutRouter, DashboardRouter, ChatAPIRouter, StatisticsAPIRouter, PredictionsAPIRouter
  app.register_blueprint(AboutRouter)
  app.register_blueprint(DashboardRouter)
  app.register_blueprint(ChatAPIRouter)
  app.register_blueprint(StatisticsAPIRouter)
  app.register_blueprint(PredictionsAPIRouter)

  return app
