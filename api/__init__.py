from flask import Blueprint

PREFIX = 'sentiment'

sentiment_blueprint = Blueprint(PREFIX, __name__, url_prefix='/sentiment')

from api import sentiment
